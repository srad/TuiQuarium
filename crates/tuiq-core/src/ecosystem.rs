//! Energy-based ecosystem: metabolism, feeding, hunting, death.
//! Feeding roles are emergent from morphology — no fixed trophic categories.

use std::collections::HashSet;

use hecs::{Entity, World};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::calibration::EcologyCalibration;
use crate::components::{Position, Velocity};
use crate::environment::Environment;
use crate::genome::CreatureGenome;
use crate::needs::Needs;
use crate::phenotype::{DerivedPhysics, FeedingCapability};
use crate::spatial::SpatialGrid;
use crate::EntityInfoMap;

/// Energy state for a living creature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Energy {
    pub current: f32,
    pub max: f32,
}

impl Energy {
    pub fn new(max: f32) -> Self {
        Self {
            current: max * 0.8,
            max,
        }
    }

    pub fn new_with(current: f32, max: f32) -> Self {
        Self {
            current: current.min(max),
            max,
        }
    }

    pub fn fraction(&self) -> f32 {
        (self.current / self.max).clamp(0.0, 1.0)
    }
}

/// Monod/Michaelis-Menten style saturation for limiting resources.
///
/// Research note: resource uptake in ecological growth models is commonly
/// represented as a saturating response to resource concentration rather than
/// as a hard threshold (Monod, 1949).
fn saturation_limit(resource: f32, half_sat: f32) -> f32 {
    if resource <= 0.0 {
        0.0
    } else {
        resource / (resource + half_sat.max(1e-3))
    }
}

/// Age of a creature in simulation ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Age {
    pub ticks: u64,
    pub max_ticks: u64,
}

/// Marker component for producer entities (plants, food pellets).
/// Creatures do NOT have this — their feeding behavior is determined by FeedingCapability.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Producer;

/// Marker component for detritus entities (dead creature remains).
/// Detritus is a producer that decays faster and can be grazed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Detritus;

/// Tank-wide dissolved nutrient state and suspended phytoplankton load.
///
/// Research note: submerged macrophytes are frequently co-limited by nitrogen and
/// phosphorus rather than by light alone (Mebane et al., 2021), and nutrient-driven
/// phytoplankton/periphyton shading is a major indirect stressor (Yu et al., 2018).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientPool {
    pub dissolved_n: f32,
    pub dissolved_p: f32,
    pub sediment_n: f32,
    pub sediment_p: f32,
    pub phytoplankton_load: f32,
}

impl Default for NutrientPool {
    fn default() -> Self {
        Self {
            dissolved_n: 72.0,
            dissolved_p: 12.0,
            sediment_n: 220.0,
            sediment_p: 44.0,
            phytoplankton_load: 0.15,
        }
    }
}

impl NutrientPool {
    pub fn tick(&mut self, env: &Environment, dt: f32, calibration: &EcologyCalibration) {
        let runtime_targets =
            crate::ecology_equilibrium::ReducedEquilibriumModel::default_runtime_nutrient_targets();
        let n_target = runtime_targets.dissolved_n_target.max(1e-3);
        let p_target = runtime_targets.dissolved_p_target.max(1e-3);
        let n_deficit = ((n_target - self.dissolved_n) / n_target).clamp(0.0, 1.0);
        let p_deficit = ((p_target - self.dissolved_p) / p_target).clamp(0.0, 1.0);

        // Research note: shallow aquatic systems can recycle a meaningful share
        // of benthic nutrients back into the water column on ecological
        // timescales, so the founder-web baseline should not behave like a
        // near-closed sediment sink from the first day.
        let release_n_rate = 0.0006 + 0.0016 * n_deficit;
        let release_p_rate = 0.0006 + 0.0014 * p_deficit;
        let release_n = (self.sediment_n * release_n_rate * dt).min(self.sediment_n);
        let release_p = (self.sediment_p * release_p_rate * dt).min(self.sediment_p);
        self.sediment_n -= release_n;
        self.sediment_p -= release_p;
        self.dissolved_n += release_n;
        self.dissolved_p += release_p;
        self.dissolved_n += (0.0004 + 0.0040 * n_deficit) * dt;
        self.dissolved_p += (0.00005 + 0.0009 * p_deficit) * dt;

        // Nitrogen fixation analogue: cyanobacteria-like background process
        // slowly restores dissolved N when it drops below a critical threshold.
        // This prevents irreversible ecosystem crashes from nitrogen depletion.
        let n_fixation_threshold = (runtime_targets.dissolved_n_target * 0.65).max(2.5);
        const N_FIXATION_RATE: f32 = 0.10;
        if self.dissolved_n < n_fixation_threshold {
            let deficit_ratio = 1.0 - self.dissolved_n / n_fixation_threshold;
            self.dissolved_n += N_FIXATION_RATE * deficit_ratio * dt;
        }

        if self.dissolved_n > runtime_targets.dissolved_n_upper {
            let excess_n = self.dissolved_n - runtime_targets.dissolved_n_upper;
            let denitrification_n = excess_n * 0.030 * dt;
            let export_n = excess_n * 0.018 * dt;
            self.dissolved_n = (self.dissolved_n - denitrification_n - export_n).max(0.0);
        }

        // Research note: shallow lakes do export/bury phosphorus under high-load
        // conditions via precipitation, burial, and outflow. Without a soft cap,
        // the simplified loop drifts into unrealistically extreme dissolved P.
        if self.dissolved_p > runtime_targets.dissolved_p_upper {
            let excess_p = self.dissolved_p - runtime_targets.dissolved_p_upper;
            let burial_p = excess_p * 0.0120 * dt;
            let export_p = excess_p * 0.0060 * dt;
            self.dissolved_p = (self.dissolved_p - burial_p - export_p).max(0.0);
            self.sediment_p += burial_p;
        }

        self.dissolved_n = self.dissolved_n.max(0.0);
        self.dissolved_p = self.dissolved_p.max(0.0);

        let nutrient_pressure = (self.dissolved_n / 95.0 + self.dissolved_p / 18.0) * 0.5;
        let bloom_target = if matches!(
            env.active_event.as_ref().map(|e| e.kind),
            Some(crate::environment::EventKind::AlgaeBloom)
        ) {
            0.38
        } else {
            0.015 + nutrient_pressure.clamp(0.0, 1.0) * 0.22
        };
        self.phytoplankton_load += (bloom_target - self.phytoplankton_load)
            * (0.12 * calibration.phytoplankton_shading_multiplier * dt);
        self.phytoplankton_load = self.phytoplankton_load.clamp(0.0, 0.85);
    }

    pub fn take(&mut self, n: f32, p: f32) -> (f32, f32) {
        let taken_n = n.min(self.dissolved_n);
        let taken_p = p.min(self.dissolved_p);
        self.dissolved_n -= taken_n;
        self.dissolved_p -= taken_p;
        (taken_n, taken_p)
    }

    pub fn recycle(&mut self, n: f32, p: f32) {
        // Research note: the microbial loop can return a large share of dead
        // organic matter to dissolved nutrients quickly, but phosphorus is more
        // often retained in the benthic pool than nitrogen in shallow systems.
        self.dissolved_n += n * 0.30;
        self.dissolved_p += p * 0.20;
        self.sediment_n += n * 0.70;
        self.sediment_p += p * 0.80;
    }
}

/// Rasterized light field used to avoid pairwise plant-plant shading checks.
///
/// Research note: submerged macrophyte performance is dominated by depth/light
/// attenuation and canopy access to brighter water (He et al., 2019; Yu et al., 2018).
#[derive(Debug, Clone)]
pub struct LightField {
    width: usize,
    height: usize,
    opacity: Vec<f32>,
    shade_prefix: Vec<f32>,
}

impl LightField {
    pub fn new(width: u16, height: u16) -> Self {
        let width = width.max(1) as usize;
        let height = height.max(1) as usize;
        Self {
            width,
            height,
            opacity: vec![0.0; width * height],
            shade_prefix: vec![0.0; width * height],
        }
    }

    pub fn rebuild(&mut self, world: &World) {
        self.opacity.fill(0.0);
        self.shade_prefix.fill(0.0);

        for (pos, bbox, genome, state) in &mut world.query::<(
            &Position,
            &crate::components::BoundingBox,
            &crate::genome::ProducerGenome,
            &crate::components::ProducerState,
        )>() {
            let leaf_fill = (state.leaf_biomass / genome.active_target_biomass()).clamp(0.15, 1.2);
            let crown_h = (bbox.h.max(1.0) * (0.35 + genome.branching * 0.25)).ceil() as i32;
            let crown_w = (bbox.w.max(1.0) * (0.65 + genome.branching * 0.35)).ceil() as i32;
            let x0 = pos.x.floor() as i32;
            let y0 = pos.y.floor() as i32;
            let opacity =
                0.05 + genome.leaf_area * 0.14 + genome.branching * 0.08 + leaf_fill * 0.10;

            for yy in y0.max(0)..(y0 + crown_h).min(self.height as i32) {
                for xx in x0.max(0)..(x0 + crown_w).min(self.width as i32) {
                    let idx = yy as usize * self.width + xx as usize;
                    self.opacity[idx] += opacity;
                }
            }
        }

        for x in 0..self.width {
            let mut cumulative = 0.0;
            for y in 0..self.height {
                let idx = y * self.width + x;
                self.shade_prefix[idx] = cumulative;
                cumulative += self.opacity[idx];
            }
        }
    }

    pub fn sample_light(
        &self,
        x: f32,
        y: f32,
        crown_width: f32,
        ambient_light: f32,
        tank_height: f32,
        nutrients: &NutrientPool,
        calibration: &EcologyCalibration,
    ) -> f32 {
        let ix0 = x.floor().clamp(0.0, (self.width.saturating_sub(1)) as f32) as usize;
        let ix1 = (x + crown_width.max(1.0))
            .ceil()
            .clamp(0.0, self.width as f32) as usize;
        let iy = y.floor().clamp(0.0, (self.height.saturating_sub(1)) as f32) as usize;

        let mut shade = 0.0;
        let mut count = 0.0;
        for ix in ix0..ix1.max(ix0 + 1) {
            shade += self.shade_prefix[iy * self.width + ix.min(self.width - 1)];
            count += 1.0;
        }
        let avg_shade = if count > 0.0 { shade / count } else { 0.0 };

        let depth_fraction = (y / tank_height.max(1.0)).clamp(0.0, 1.0);
        let attenuation = 0.45
            + nutrients.phytoplankton_load * 1.3 * calibration.phytoplankton_shading_multiplier;
        let depth_transmittance = (-attenuation * depth_fraction).exp();
        let canopy_transmittance = (-avg_shade).exp();

        (ambient_light * depth_transmittance * canopy_transmittance).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct MetabolismFlux {
    pub detritus_n: f32,
    pub detritus_p: f32,
    pub consumer_maintenance: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ProducerEcologyFlux {
    pub net_primary_production: f32,
    pub labile_detritus_energy: f32,
    pub pelagic_consumer_intake: f32,
}

#[derive(Debug, Clone, Default)]
pub struct FeedingSummary {
    pub dead: Vec<Entity>,
    pub total_assimilation: f32,
}

/// Reproductive reserve threshold for one successful offspring event.
///
/// Research note: offspring production scales with adult body size and
/// complexity-linked investment cost, so reproduction should require larger
/// buffers in larger-bodied consumers.
fn consumer_simple_lineage_credit(physics: &DerivedPhysics, genome: &CreatureGenome) -> f32 {
    let simplicity = (1.0 - genome.complexity).clamp(0.0, 1.0);
    let low_mass = ((1.35 - physics.body_mass.max(0.1)) / 1.35).clamp(0.0, 1.0);
    simplicity * 0.035 + low_mass * 0.025
}

fn consumer_soft_population_cap(tank_w: f32, tank_h: f32) -> f32 {
    crate::ecology_equilibrium::ReducedEquilibriumModel::default_startup_targets(
        tank_w.max(1.0).round() as u16,
        tank_h.max(1.0).round() as u16,
    )
    .soft_population_cap
    .max(1) as f32
}

pub fn consumer_reproductive_threshold(physics: &DerivedPhysics, genome: &CreatureGenome) -> f32 {
    let simplicity_credit = consumer_simple_lineage_credit(physics, genome);
    (0.17 + physics.body_mass.max(0.1).powf(0.75) * 0.13 + genome.complexity * 0.11
        - simplicity_credit)
        .max(0.11)
}

pub fn consumer_reproduction_reserve_threshold(
    physics: &DerivedPhysics,
    genome: &CreatureGenome,
) -> f32 {
    let simplicity = (1.0 - genome.complexity).clamp(0.0, 1.0);
    let low_mass = ((1.35 - physics.body_mass.max(0.1)) / 1.35).clamp(0.0, 1.0);
    (0.45 + genome.complexity * 0.05 + physics.body_mass.max(0.1).powf(0.18) * 0.015
        - simplicity * 0.035
        - low_mass * 0.020)
        .clamp(0.40, 0.56)
}

/// Drain energy from creature metabolism and decay producers that are not plants.
/// Returns nutrient flux released from detritus decay.
pub fn metabolism_system(
    world: &mut World,
    dt: f32,
    env: &Environment,
    tank_height: f32,
    calibration: &EcologyCalibration,
) -> MetabolismFlux {
    let temp_mod = env.temperature_modifier();
    // Night stress: creatures burn more energy in darkness (thermoregulation, vigilance)
    let night_stress = 1.0 + 0.3 * (1.0 - env.light_level).max(0.0);
    let mut flux = MetabolismFlux::default();

    for (energy, physics, pos, is_detritus, is_producer, plant_genome) in world.query_mut::<(
        &mut Energy,
        &DerivedPhysics,
        Option<&Position>,
        Option<&Detritus>,
        Option<&Producer>,
        Option<&crate::genome::ProducerGenome>,
    )>() {
        let y = pos.map(|p| p.y).unwrap_or(0.0);

        if is_detritus.is_some() {
            // Detritus decays at a fixed rate
            let decay = 0.5 * dt * 0.12;
            energy.current -= decay;
            flux.detritus_n += decay * 0.45;
            flux.detritus_p += decay * 0.09;
        } else if plant_genome.is_some() {
            continue;
        } else if is_producer.is_some() {
            energy.current -= physics.base_metabolism * dt * 0.25;
        } else {
            // Creature: metabolism scaled by depth, temperature, and day/night cycle
            let depth_mod = env.metabolism_at_depth(y, tank_height);
            let cost = physics.base_metabolism
                * dt
                * 0.25
                * depth_mod
                * temp_mod
                * night_stress
                * calibration.consumer_metabolism_multiplier;
            energy.current -= cost;
            flux.consumer_maintenance += cost.max(0.0);
        }
    }

    flux
}

/// Rooted macrophyte growth, plus a lightweight pelagic phytoplankton loop.
///
/// Research note: this pass combines continuous depth attenuation and canopy access
/// (He et al., 2019), nutrient co-limitation (Mebane et al., 2021), and indirect
/// shading stress from attached growth/phytoplankton (Yu et al., 2018).
pub fn producer_ecology_system(
    world: &mut World,
    dt: f32,
    env: &Environment,
    tank_height: f32,
    light_field: &LightField,
    nutrients: &mut NutrientPool,
    calibration: &EcologyCalibration,
) -> ProducerEcologyFlux {
    let temp_mod = env.temperature_modifier();
    let mut flux = ProducerEcologyFlux::default();

    // Research note: shallow-lake primary production is split between rooted
    // macrophytes and suspended phytoplankton. The phytoplankton pool is kept
    // coarse-grained here, but it still takes up dissolved nutrients, turns
    // light into pelagic production, and leaks carbon back into the detrital
    // loop instead of acting only as a shading scalar.
    let pelagic_light = (env.light_level
        * (-(0.10 + nutrients.phytoplankton_load * 0.65) * 0.18).exp())
    .clamp(0.0, 1.0);
    let pelagic_n_limit = saturation_limit(nutrients.dissolved_n, 14.0);
    let pelagic_p_limit = saturation_limit(nutrients.dissolved_p, 3.0);
    let pelagic_nutrient_limit = pelagic_n_limit.min(pelagic_p_limit).clamp(0.0, 1.0);
    let pelagic_growth = nutrients.phytoplankton_load
        * (0.07 + pelagic_light * pelagic_nutrient_limit * 0.26)
        * temp_mod
        * calibration.producer_growth_multiplier
        * dt;
    let pelagic_turnover = nutrients.phytoplankton_load
        * (0.025 + (1.0 - pelagic_light) * 0.04 + nutrients.phytoplankton_load * 0.02)
        * dt;
    let pelagic_required_n =
        pelagic_growth * 0.08 * calibration.producer_nutrient_demand_multiplier;
    let pelagic_required_p =
        pelagic_growth * 0.012 * calibration.producer_nutrient_demand_multiplier;
    let (pelagic_n_taken, pelagic_p_taken) = nutrients.take(pelagic_required_n, pelagic_required_p);
    let pelagic_uptake_limit = if pelagic_required_n > 0.0 && pelagic_required_p > 0.0 {
        (pelagic_n_taken / pelagic_required_n)
            .min(pelagic_p_taken / pelagic_required_p)
            .clamp(0.0, 1.0)
    } else {
        1.0
    };
    let realized_pelagic_growth = pelagic_growth * pelagic_uptake_limit;
    nutrients.recycle(pelagic_turnover * 0.05, pelagic_turnover * 0.008);
    nutrients.phytoplankton_load = (nutrients.phytoplankton_load + realized_pelagic_growth * 0.14
        - pelagic_turnover * 0.90)
        .clamp(0.0, 0.85);
    flux.net_primary_production += realized_pelagic_growth * 0.45;
    flux.labile_detritus_energy += realized_pelagic_growth * 0.75 + pelagic_turnover * 0.55;

    // Research note: a productive lake supports some secondary production from
    // consumers exploiting suspended phytoplankton directly, not only from
    // cropped macrophytes. This keeps the pelagic pathway from becoming a dead
    // end in otherwise producer-rich conditions.
    let mut pelagic_grazing = 0.0;
    let pelagic_energy_per_load =
        (light_field.width.max(1) * light_field.height.max(1)) as f32 * 0.015;
    let mut pelagic_energy_budget =
        nutrients.phytoplankton_load.powi(2) * pelagic_energy_per_load * 4.0;
    for (_entity, energy, physics, feeding, needs, state) in world.query_mut::<(
        Entity,
        &mut Energy,
        &DerivedPhysics,
        &FeedingCapability,
        &mut Needs,
        Option<&mut crate::components::ConsumerState>,
    )>() {
        if feeding.is_producer || feeding.graze_skill < 0.32 || needs.hunger < 0.12 {
            continue;
        }
        if energy.current <= 0.0 {
            continue;
        }

        let size_bonus = (1.20 / physics.body_mass.max(0.12).powf(0.24)).clamp(0.85, 1.90);
        let pelagic_size_gate = if physics.body_mass <= 1.2 {
            1.0
        } else {
            (1.2 / physics.body_mass.max(1.2))
                .powf(1.35)
                .clamp(0.18, 0.55)
        };
        let detritivore_bias =
            (feeding.graze_skill * (1.0 - feeding.hunt_skill).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let filter_affinity = (feeding.graze_skill * 1.12
            + (1.0 - (physics.body_mass / 1.6).clamp(0.0, 1.0)) * 0.10
            + detritivore_bias * 0.06
            - feeding.hunt_skill * 0.10)
            .clamp(0.0, 1.55);
        if filter_affinity <= 0.0 {
            continue;
        }

        let appetite =
            (energy.max - energy.current).max(0.0) * (0.35 + needs.hunger.clamp(0.0, 1.0) * 0.65);
        if appetite <= 0.0 {
            continue;
        }

        let suspension_food = saturation_limit(nutrients.phytoplankton_load, 0.08).clamp(0.0, 1.0);
        if suspension_food <= 0.01 || pelagic_energy_budget <= 1e-4 {
            continue;
        }
        let intake_capacity = physics.base_metabolism.max(0.05)
            * (0.28 + filter_affinity * 1.12)
            * suspension_food
            * size_bonus
            * pelagic_size_gate
            * (0.85 + appetite / energy.max.max(1e-3) * 0.75)
            * dt
            * 1.90;
        let assimilated = appetite
            .min(intake_capacity.max(0.0))
            .min(pelagic_energy_budget);
        if assimilated <= 0.0 {
            continue;
        }

        pelagic_energy_budget = (pelagic_energy_budget - assimilated).max(0.0);
        energy.current = (energy.current + assimilated).min(energy.max);
        needs.hunger = (needs.hunger - assimilated / energy.max.max(1e-3) * 0.95).max(0.0);
        if let Some(state) = state {
            state.recent_assimilation =
                (state.recent_assimilation + assimilated / energy.max.max(1e-3)).min(1.5);
        }
        pelagic_grazing += assimilated;
    }
    let pelagic_load_removed = if pelagic_energy_per_load > 1e-4 {
        pelagic_grazing / pelagic_energy_per_load
    } else {
        0.0
    };
    nutrients.phytoplankton_load =
        (nutrients.phytoplankton_load - pelagic_load_removed).clamp(0.0, 0.85);
    flux.pelagic_consumer_intake += pelagic_grazing;

    for (pos, energy, physics, genome, state, age) in world.query_mut::<(
        &Position,
        &mut Energy,
        &DerivedPhysics,
        &crate::genome::ProducerGenome,
        &mut crate::components::ProducerState,
        &Age,
    )>() {
        state.seed_cooldown = (state.seed_cooldown - dt).max(0.0);
        state.clonal_cooldown = (state.clonal_cooldown - dt).max(0.0);

        let structural_target = genome.support_target_biomass();
        let leaf_target = genome.active_target_biomass();
        let belowground_target =
            structural_target * (0.22 + genome.hardiness * 0.22 + genome.clonal_spread * 0.18);
        let meristem_target = 0.10 + genome.clonal_spread * 0.30 + genome.hardiness * 0.12;
        let root_access = (0.22
            + genome.hardiness * 0.18
            + genome.clonal_spread * 0.12
            + (state.belowground_reserve / belowground_target.max(0.05)).clamp(0.0, 1.2) * 0.28)
            .clamp(0.18, 0.72);
        let crown_width = 1.0 + genome.branching * 2.5;
        let light = light_field.sample_light(
            pos.x,
            pos.y,
            crown_width,
            env.light_level,
            tank_height,
            nutrients,
            calibration,
        );

        // Research note: primary production and nutrient uptake are modeled as
        // saturating resource responses instead of threshold switches
        // (Monod, 1949; Mebane et al., 2021).
        let light_half_sat =
            0.12 / (0.7 + genome.photosynthesis_rate * 0.35 + genome.hardiness * 0.20);
        let light_limit = saturation_limit(light, light_half_sat).clamp(0.0, 1.0);
        let n_half = 18.0 / genome.nutrient_affinity.max(0.2);
        let p_half = 4.0 / genome.nutrient_affinity.max(0.2);
        // Research note: attached producer colonies can supplement water-column
        // uptake with substrate nutrient access, especially when storage biomass
        // and local spread structures are well developed.
        let effective_n = nutrients.dissolved_n + nutrients.sediment_n * root_access * 0.08;
        let effective_p = nutrients.dissolved_p + nutrients.sediment_p * root_access * 0.05;
        let n_limit = saturation_limit(effective_n, n_half);
        let p_limit = saturation_limit(effective_p, p_half);
        let nutrient_limit = n_limit.min(p_limit).clamp(0.0, 1.0);

        let nutrient_pressure = ((n_limit + p_limit) * 0.5).clamp(0.0, 1.0);
        let epiphyte_input = (0.004 + nutrients.phytoplankton_load * 0.018)
            * light_limit
            * (0.6 + nutrient_pressure * 0.4)
            * (1.0 - genome.epiphyte_resistance * 0.65);
        let epiphyte_loss = 0.020 + genome.hardiness * 0.03;
        state.epiphyte_load =
            (state.epiphyte_load + (epiphyte_input - epiphyte_loss) * dt).clamp(0.0, 1.5);
        let epiphyte_shade = (1.0 - state.epiphyte_load * 0.45).clamp(0.35, 1.0);

        // Research note: tissue maintenance follows allometric scaling rather
        // than stage-specific constants (Brown et al., 2004).
        let reserve_ratio = energy.fraction();
        let photosynthetic_capacity = (-physics.base_metabolism)
            * state.leaf_biomass.max(0.05).powf(0.75)
            * (0.85 + genome.leaf_area * 0.35)
            * temp_mod
            * calibration.producer_growth_multiplier;
        let potential_assimilation =
            photosynthetic_capacity * light_limit * nutrient_limit * epiphyte_shade;

        let potential_gain = potential_assimilation.max(0.0) * dt;
        let required_n = potential_gain * 0.11 * calibration.producer_nutrient_demand_multiplier;
        let required_p = potential_gain * 0.015 * calibration.producer_nutrient_demand_multiplier;
        let dissolved_fraction = (1.0 - root_access * 0.45).clamp(0.45, 0.90);
        let (dissolved_n, dissolved_p) = nutrients.take(
            required_n * dissolved_fraction,
            required_p * dissolved_fraction,
        );
        let remaining_n = (required_n - dissolved_n).max(0.0);
        let remaining_p = (required_p - dissolved_p).max(0.0);
        let sediment_n = remaining_n.min(nutrients.sediment_n.min(required_n * root_access));
        let sediment_p = remaining_p.min(nutrients.sediment_p.min(required_p * root_access));
        nutrients.sediment_n -= sediment_n;
        nutrients.sediment_p -= sediment_p;
        let taken_n = dissolved_n + sediment_n;
        let taken_p = dissolved_p + sediment_p;
        let uptake_limit = if required_n > 0.0 && required_p > 0.0 {
            (taken_n / required_n)
                .min(taken_p / required_p)
                .clamp(0.0, 1.0)
        } else {
            1.0
        };
        let gross_photo = potential_assimilation * uptake_limit;

        let resource_stress = (1.0 - light_limit.min(nutrient_limit)).powi(2);
        let age_ratio = (age.ticks as f32 / age.max_ticks.max(1) as f32).max(0.0);
        let senescence = age_ratio.powf(4.0);

        // Research note: rather than a fixed lifespan kill-switch, tissue loss
        // increases smoothly with age and chronic resource stress, which is closer
        // to macrophyte turnover than deterministic death on a timer.
        let baseline_turnover = 0.08 / age.max_ticks.max(1) as f32;
        let leaf_turnover = state.leaf_biomass
            * baseline_turnover
            * (0.10 + 0.70 * resource_stress + 0.55 * senescence)
            * calibration.producer_turnover_multiplier;
        let structural_turnover = state.structural_biomass
            * baseline_turnover
            * (0.04 + 0.18 * senescence + (1.0 - reserve_ratio) * 0.12)
            * calibration.producer_turnover_multiplier;
        let belowground_turnover = state.belowground_reserve
            * baseline_turnover
            * (0.02 + 0.10 * senescence + resource_stress * 0.05)
            * calibration.producer_turnover_multiplier;

        state.leaf_biomass = (state.leaf_biomass - leaf_turnover).max(0.02);
        state.structural_biomass = (state.structural_biomass - structural_turnover).max(0.04);
        state.belowground_reserve = (state.belowground_reserve - belowground_turnover).max(0.03);
        nutrients.recycle(
            leaf_turnover * 0.11 + structural_turnover * 0.07 + belowground_turnover * 0.05,
            leaf_turnover * 0.020 + structural_turnover * 0.012 + belowground_turnover * 0.008,
        );

        let maintenance = (
            // Research note: established submerged producers should retain a
            // modest positive carbon balance in clear water, so baseline tissue
            // maintenance stays well below gross photosynthetic capacity until
            // stress pushes respiration and repair costs upward.
            photosynthetic_capacity
                * (0.06 + resource_stress * (0.04 - genome.hardiness * 0.015).max(0.018))
                + 0.0035 * state.structural_biomass.powf(0.75)
        ) * (0.75 + temp_mod * 0.25)
            * calibration.producer_maintenance_multiplier;

        if reserve_ratio < 0.22 && state.belowground_reserve > 0.08 {
            // Research note: perennial submerged macrophytes draw on stored
            // rhizome/carbohydrate reserves during dark or stressful periods,
            // buffering transient negative carbon balance rather than dying
            // immediately when soluble reserves are low.
            let translocation = (energy.max * 0.010 * dt)
                .max(state.belowground_reserve * 0.035 * dt)
                .min(state.belowground_reserve * 0.25);
            state.belowground_reserve = (state.belowground_reserve - translocation).max(0.02);
            energy.current = (energy.current + translocation).min(energy.max);
        }

        let net = gross_photo - maintenance;
        let stress_gain = resource_stress * (1.0 - genome.hardiness * 0.35) + senescence * 0.20;
        let stress_relief = reserve_ratio * (0.20 + genome.hardiness * 0.25);
        state.stress_time =
            (state.stress_time + (stress_gain - stress_relief) * dt).clamp(0.0, 8.0);

        if net > 0.0 {
            let gain = net * dt;
            let exudate_release = gain * (0.22 + genome.branching * 0.06 + light_limit * 0.06);
            let retained_gain = (gain - exudate_release).max(0.0);
            // Research note: active aquatic producers release a share of fixed
            // carbon as labile organic matter that supports the microbial loop,
            // so not all positive production should stay locked inside producer
            // biomass in the unified founder-web baseline.
            flux.net_primary_production += retained_gain;
            flux.labile_detritus_energy += exudate_release;

            let reserve_target = 0.42 + genome.reserve_allocation * 0.18;
            // Research note: plants below their soluble reserve target divert a
            // larger share of current carbon gain into fast reserve recovery,
            // while already well-provisioned plants route more gain into tissue
            // growth and belowground storage instead of refilling the same pool
            // at nearly the same rate.
            let reserve_deficit = (reserve_target - reserve_ratio).max(0.0);
            let reserve_bias = (0.06 + reserve_deficit * 0.60).clamp(0.06, 0.58);
            let reserve_gain = retained_gain * reserve_bias;
            let biomass_gain = retained_gain - reserve_gain;
            energy.current = (energy.current + reserve_gain).min(energy.max);

            let struct_gap = (structural_target - state.structural_biomass).max(0.0);
            let leaf_gap = (leaf_target - state.leaf_biomass).max(0.0);
            let below_gap = (belowground_target - state.belowground_reserve).max(0.0);
            let gap_total = struct_gap + leaf_gap + below_gap;
            if gap_total > 0.0 {
                // Research note: submerged perennials maintain belowground storage
                // and meristem banks that support post-grazing recovery rather than
                // routing all biomass gain directly into aboveground shoots.
                let to_leaf = biomass_gain * (leaf_gap / gap_total) * (0.40 + light_limit * 0.50);
                let to_below = biomass_gain
                    * (below_gap / gap_total)
                    * (0.25 + genome.hardiness * 0.25 + genome.clonal_spread * 0.20);
                let to_struct = (biomass_gain - to_leaf - to_below).max(0.0);
                state.leaf_biomass = (state.leaf_biomass + to_leaf).min(leaf_target * 1.1);
                state.structural_biomass =
                    (state.structural_biomass + to_struct).min(structural_target * 1.1);
                state.belowground_reserve =
                    (state.belowground_reserve + to_below).min(belowground_target * 1.25);
            } else {
                energy.current = (energy.current + biomass_gain * 0.5).min(energy.max);
            }
            state.meristem_bank = (state.meristem_bank
                + gain * (0.018 + genome.clonal_spread * 0.020))
                .min(meristem_target);
        } else {
            let loss = (-net) * dt;
            energy.current = (energy.current - loss).max(0.0);

            // Research note: short producer stress events should deplete soluble
            // reserve first; rapid canopy collapse is more plausible only after
            // reserves are already low, not on the first bad nutrient day.
            let catabolism_pressure = ((0.30 - reserve_ratio) / 0.30).clamp(0.0, 1.0);
            let leaf_loss = loss
                * catabolism_pressure
                * (0.08 + (1.0 - genome.hardiness) * 0.08)
                * (1.0 + state.stress_time * 0.03);
            state.leaf_biomass = (state.leaf_biomass - leaf_loss).max(0.02);

            if reserve_ratio < 0.16 {
                state.structural_biomass = (state.structural_biomass - loss * 0.02).max(0.04);
                state.belowground_reserve = (state.belowground_reserve - loss * 0.02).max(0.03);
            }
            state.meristem_bank = (state.meristem_bank - loss * 0.010).max(0.0);
        }

        if state.leaf_biomass < leaf_target * 0.35
            && state.belowground_reserve > 0.08
            && state.meristem_bank > 0.02
            && light_limit > 0.20
            && nutrient_limit > 0.18
        {
            // Research note: clonal/perennial aquatic plants can regrow shoots from
            // stored belowground reserves after defoliation when light and nutrients
            // remain sufficient (Li et al., 2015; Ren et al., 2024).
            let regrowth = ((leaf_target * 0.55 - state.leaf_biomass).max(0.0))
                .min(state.belowground_reserve * (0.10 + state.meristem_bank * 0.08) * dt);
            state.belowground_reserve = (state.belowground_reserve - regrowth * 0.85).max(0.02);
            state.leaf_biomass = (state.leaf_biomass + regrowth).min(leaf_target * 0.95);
            state.meristem_bank = (state.meristem_bank - regrowth * 0.08).max(0.0);
        }

        if state.leaf_biomass <= 0.03
            && state.structural_biomass <= 0.05
            && state.belowground_reserve <= 0.04
            && energy.current < energy.max * 0.05
        {
            energy.current = 0.0;
        }
    }

    flux
}

/// Update consumer maturation and reproductive buffering from energetic state.
///
/// Research note: consumer reproduction is modeled as a function of maturation
/// delay and sustained energetic surplus rather than a clock-like urge, which is
/// closer to real consumer-resource systems than timer-driven spawning.
pub fn consumer_life_history_system(world: &mut World, dt: f32, tank_width: f32, tank_height: f32) {
    let consumer_count = (&mut world.query::<&CreatureGenome>()).into_iter().count() as f32;
    let soft_cap = consumer_soft_population_cap(tank_width, tank_height);
    let crowding = consumer_count / soft_cap.max(1.0);
    let crowding_pressure = ((crowding - 0.60) / 0.40).clamp(0.0, 1.0);

    for (energy, age, genome, physics, needs, state) in world.query_mut::<(
        &mut Energy,
        &Age,
        &CreatureGenome,
        &DerivedPhysics,
        &mut Needs,
        &mut crate::components::ConsumerState,
    )>() {
        state.brood_cooldown = (state.brood_cooldown - dt).max(0.0);
        state.recent_assimilation *= (1.0 - 0.05 * dt).max(0.0);
        if state.maturity_progress >= 1.0 {
            state.matured_once = true;
            state.maturity_progress = 1.0;
        }

        let reserve_ratio = energy.fraction();
        let age_fraction = (age.ticks as f32 / age.max_ticks.max(1) as f32).clamp(0.0, 1.35);
        let senescence = ((age_fraction - 0.46) / 0.34).clamp(0.0, 1.0);
        // Research note: age at maturity rises with body size, but much more
        // weakly than total lifespan in aquatic ectotherms, so maturation is
        // tracked on its own allometric timescale instead of as a fixed share
        // of lifespan (Gillooly et al., 2002).
        let maturation_ticks = 5_600.0
            * (0.78 + physics.body_mass.max(0.1).powf(0.32) * 0.38 + genome.complexity * 0.48)
                .clamp(0.72, 1.48);
        let age_signal = (age.ticks as f32 / maturation_ticks.max(1.0)).clamp(0.0, 1.4);
        let condition_signal = ((reserve_ratio - 0.45) / 0.35).clamp(0.0, 1.0);
        let generation_pace =
            (1.16 - genome.complexity * 0.44 - physics.body_mass.max(0.1).powf(0.28) * 0.18)
                .clamp(0.58, 1.10)
                * (1.0 - senescence * 0.36);

        let maturity_gain = age_signal
            * generation_pace
            * (0.16 + condition_signal * 0.36 + state.recent_assimilation.min(0.30) * 1.0)
            * (1.0 - senescence * 0.60)
            * dt
            * 0.0052;
        if state.matured_once {
            state.maturity_progress = 1.0;
        } else {
            let maturity_loss = if reserve_ratio < 0.28 {
                dt * 0.0044
            } else {
                0.0
            } + dt * senescence * 0.0021;
            state.maturity_progress =
                (state.maturity_progress + maturity_gain - maturity_loss).clamp(0.0, 1.0);
            if state.maturity_progress >= 1.0 {
                state.matured_once = true;
                state.maturity_progress = 1.0;
            }
        }

        let small_lineage_credit = consumer_simple_lineage_credit(physics, genome);
        let reserve_gate = consumer_reproduction_reserve_threshold(physics, genome);
        let reserve_target = (reserve_ratio
            * (0.62 + state.recent_assimilation.min(0.45) * 0.52)
            * (1.0 - senescence * 0.22))
            .clamp(0.0, 1.0);
        state.reserve_buffer += (reserve_target - state.reserve_buffer) * dt * 0.26;

        let adultness = if state.is_adult() {
            1.0
        } else {
            state.maturity_progress
        };
        let threshold = consumer_reproductive_threshold(physics, genome);
        let surplus_floor = (0.52 - small_lineage_credit * 1.5).clamp(0.40, 0.52);
        let surplus_span = (0.23 - small_lineage_credit * 0.65).clamp(0.16, 0.23);
        let surplus_condition = ((state.reserve_buffer - surplus_floor) / surplus_span)
            .clamp(0.0, 1.0)
            * (1.0 - crowding_pressure * 0.25);
        let hunger_relief = (1.0 - needs.hunger).clamp(0.0, 1.0);
        let assimilation_target = (0.050 - small_lineage_credit * 0.22).clamp(0.036, 0.050);
        let assimilation_gate = (state.recent_assimilation / assimilation_target).clamp(0.0, 1.0);
        let adult_gate = (0.92 - small_lineage_credit * 0.40).clamp(0.88, 0.92);
        let healthy_adult = state.is_adult()
            && state.reserve_buffer >= reserve_gate * 0.96
            && needs.hunger < 0.70
            && state.recent_assimilation >= assimilation_target * 0.70;

        if adultness >= adult_gate
            && state.brood_cooldown <= 0.0
            && reserve_ratio > (reserve_gate - 0.02).max(0.40)
            && needs.hunger < 0.76
        {
            let buffer_gain = genome.behavior.reproduction_rate
                * dt
                * 0.0072
                * adultness
                * generation_pace
                * surplus_condition
                * hunger_relief
                * (0.34 + assimilation_gate * 0.66)
                * (1.0 - senescence * 0.52);
            state.reproductive_buffer =
                (state.reproductive_buffer + buffer_gain).min(threshold * 1.45);
        } else {
            let passive_decay = if healthy_adult && crowding_pressure < 0.30 && senescence < 0.30
            {
                0.00015
            } else {
                0.0007
            };
            state.reproductive_buffer = (state.reproductive_buffer - dt * passive_decay).max(0.0);
        }

        if reserve_ratio < 0.38 || needs.hunger > 0.74 {
            state.reproductive_buffer = (state.reproductive_buffer - dt * 0.0036).max(0.0);
        }
        let low_assimilation_stress = if state.recent_assimilation < 0.024 {
            let stress = ((0.024 - state.recent_assimilation) / 0.024).clamp(0.0, 1.0);
            state.reserve_buffer =
                (state.reserve_buffer - dt * (0.0010 + stress * 0.0022)).max(0.0);
            needs.hunger = (needs.hunger + dt * (0.0007 + stress * 0.0012)).min(1.0);
            stress
        } else {
            0.0
        };
        if senescence > 0.0 {
            state.reproductive_buffer =
                (state.reproductive_buffer - dt * (0.0016 + senescence * 0.0044)).max(0.0);
            needs.hunger = (needs.hunger + dt * senescence * 0.0030).min(1.0);
        }
        if crowding_pressure > 0.0 {
            state.reproductive_buffer =
                (state.reproductive_buffer - dt * crowding_pressure * 0.0014).max(0.0);
            needs.hunger = (needs.hunger + dt * crowding_pressure * 0.0010).min(1.0);
        }

        let urge = if state.brood_cooldown > 0.0 {
            0.0
        } else {
            (state.reproductive_buffer / threshold.max(1e-3)).clamp(0.0, 1.25)
        };
        needs.reproduction = (adultness
            * urge
            * (0.48 + reserve_ratio * 0.52)
            * (1.0 - needs.hunger * 0.26)
            * (1.0 - senescence * 0.38))
            .clamp(0.0, 1.0);

        let reproductive_load = (state.reproductive_buffer / threshold.max(1e-3)).clamp(0.0, 1.0);
        let starvation_overhead = if state.recent_assimilation < 0.010 {
            low_assimilation_stress * 0.0009
        } else {
            0.0
        };
        let somatic_overhead = (adultness * 0.0009
            + senescence * 0.0052
            + reproductive_load * 0.0006
            + starvation_overhead
            + crowding_pressure * (0.0018 + adultness * 0.0022))
            * dt;
        if somatic_overhead > 0.0 {
            energy.current = (energy.current - energy.max * somatic_overhead).max(0.0);
        }
    }
}

/// Advance age of all creatures.
pub fn age_system(world: &mut World) {
    for age in world.query_mut::<&mut Age>() {
        age.ticks += 1;
    }
}

/// Check predator-prey interactions using emergent feeding capabilities.
/// A creature can eat another entity if:
/// - The prey's body_mass < predator's max_prey_mass
/// - For producers: grazing skill > 0.2
/// - For mobile prey: hunt_skill > 0.3 AND (pursuit speed advantage OR ambush)
///
/// Two predation strategies (Pianka, 1966; Huey & Pianka, 1981):
/// - **Pursuit**: predator must be fast enough to overtake prey (speed gate)
/// - **Ambush**: stationary, camouflaged predator strikes nearby prey (Webb, 1984)
struct EaterInfo {
    entity: Entity,
    x: f32,
    y: f32,
    max_speed: f32,
    actual_speed: f32,
    sensory_range: f32,
    max_prey_mass: f32,
    hunt_skill: f32,
    graze_skill: f32,
    hunger: f32,
    camouflage: f32,
    body_mass: f32,
}

pub fn hunting_check(
    world: &World,
    grid: &SpatialGrid,
    entity_map: &EntityInfoMap,
) -> Vec<(Entity, Entity)> {
    // Collect all creatures with feeding capability (potential eaters)
    let eaters: Vec<EaterInfo> = {
        let mut v = Vec::new();
        for (entity, pos, physics, feeding, vel) in &mut world.query::<(
            Entity,
            &Position,
            &DerivedPhysics,
            &FeedingCapability,
            &Velocity,
        )>() {
            if !feeding.is_producer {
                let hunger = world.get::<&Needs>(entity).map(|n| n.hunger).unwrap_or(1.0);
                let actual_speed = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
                v.push(EaterInfo {
                    entity,
                    x: pos.x,
                    y: pos.y,
                    max_speed: physics.max_speed,
                    actual_speed,
                    sensory_range: physics.sensory_range,
                    max_prey_mass: feeding.max_prey_mass,
                    hunt_skill: feeding.hunt_skill,
                    graze_skill: feeding.graze_skill,
                    hunger,
                    camouflage: physics.camouflage,
                    body_mass: physics.body_mass,
                });
            }
        }
        v
    };

    // Compute kills in parallel — each eater finds at most one prey
    eaters
        .par_iter()
        .filter_map(|eater| {
            if eater.hunger < 0.15 {
                return None;
            }
            let neighbors = grid.neighbors(eater.x, eater.y, eater.sensory_range);
            for &prey_entity in &neighbors {
                if prey_entity == eater.entity {
                    continue;
                }

                let prey = match entity_map.get(&prey_entity) {
                    Some(p) => p,
                    None => continue,
                };

                // Research note: grazers can crop tissue from a producer
                // colony without being able to consume the entire colony as
                // a single prey item, so whole-body prey-size limits should
                // only apply to mobile prey.
                if !prey.is_producer && prey.body_mass > eater.max_prey_mass {
                    continue;
                }

                let dx = eater.x - prey.x;
                let dy = eater.y - prey.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if prey.is_producer {
                    if eater.graze_skill < 0.2 || eater.hunger < 0.15 {
                        continue;
                    }
                    if dist < 6.0 {
                        return Some((eater.entity, prey_entity));
                    }
                } else {
                    if eater.hunt_skill < 0.3 || eater.hunger < 0.25 {
                        continue;
                    }

                    // Pianka (1966): ambush predation — stationary, camouflaged
                    // predators bypass the pursuit speed requirement.
                    let speed_ratio = eater.actual_speed / eater.max_speed.max(0.01);
                    let ambush_factor = (1.0 - speed_ratio.min(1.0)) * eater.camouflage;

                    if ambush_factor > 0.35 {
                        // Webb (1984): strike distance scales with body size
                        let strike_dist = 2.0 + eater.body_mass.powf(0.33) * 2.0;
                        if dist < strike_dist {
                            return Some((eater.entity, prey_entity));
                        }
                    } else {
                        // Pursuit: must be fast enough to catch prey
                        if eater.max_speed < prey.max_speed * 0.8 {
                            continue;
                        }
                        let strike_dist = 2.5 + eater.body_mass.powf(0.33) * 1.0;
                        if dist < strike_dist {
                            return Some((eater.entity, prey_entity));
                        }
                    }
                }
            }
            None
        })
        .collect()
}

/// Apply feeding interactions.
/// - **Producers** (plants/food): partial grazing — drain energy, plant survives and
///   regenerates via photosynthesis. Only dies naturally if overgrazed to zero.
/// - **Mobile prey**: killed on capture — energy transferred, prey marked for removal.
pub fn apply_kills(world: &mut World, kills: &[(Entity, Entity)], dt: f32) -> FeedingSummary {
    let mut dead = Vec::new();
    let mut total_assimilation = 0.0;
    for &(pred, prey) in kills {
        let is_producer = world.get::<&Producer>(prey).is_ok();

        if is_producer {
            // Grazing efficiency scales with complexity — complex digestive systems
            // extract more nutrition from food
            let complexity = world
                .get::<&CreatureGenome>(pred)
                .map(|g| g.complexity)
                .unwrap_or(0.0);
            let efficiency = 0.6 + 0.4 * complexity.max(0.1);
            let hunger = world
                .get::<&Needs>(pred)
                .map(|n| n.hunger)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let (pred_current, pred_max) = world
                .get::<&Energy>(pred)
                .map(|e| (e.current, e.max))
                .unwrap_or((0.0, 1.0));
            let pred_physics = world
                .get::<&DerivedPhysics>(pred)
                .ok()
                .map(|physics| (*physics).clone())
                .unwrap_or_default();
            let energy_deficit = (pred_max - pred_current).max(0.0);
            let appetite = energy_deficit * hunger;

            if appetite <= 0.0 {
                continue;
            }

            // Research note: intake is demand-limited and saturating rather than a
            // fixed prey damage fraction, which is closer to Holling's functional
            // response framing for consumer-resource interactions (Holling, 1959).
            //
            // Handling note: intake and tissue removal must scale with elapsed time,
            // otherwise herbivory becomes an artifact of tick frequency rather than
            // of encounter rate and consumer physiology.
            // Research note: ingestion capacity should scale with metabolic
            // demand/handling, not with metabolic demand times body mass all
            // over again; otherwise the smallest consumers are artificially
            // starved by a mass^1.5 intake rule.
            let intake_capacity = pred_physics.base_metabolism.max(0.05)
                * (0.85 + pred_physics.body_mass.max(0.1).powf(0.25) * 0.45)
                * (0.8 + hunger * 1.4)
                * dt
                * 2.5;
            let desired_gain = appetite.min(intake_capacity.max(0.05));
            let caloric_gain;

            if let Ok(genome) = world.get::<&crate::genome::ProducerGenome>(prey) {
                // Research note: herbivory on submerged plants often removes leaf tissue
                // first, and grazers can indirectly help plants by reducing attached growth
                // (Ren et al., 2024). We therefore defoliate and scrub epiphytes before
                // touching structural biomass.
                let nutritional_value = genome.nutritional_value;
                let rooted = world.get::<&crate::components::RootedMacrophyte>(prey).is_ok();
                let leaf_refuge_floor = if rooted {
                    (genome.active_target_biomass() * 0.18).max(0.08)
                } else {
                    0.0
                };
                drop(genome);

                let raw_demand = desired_gain / (efficiency * nutritional_value.max(0.2));
                let leaf_removed = if let Ok(mut state) =
                    world.get::<&mut crate::components::ProducerState>(prey)
                {
                    let leaf_energy_density = 6.0;
                    let grazeable_leaf = if rooted {
                        (state.leaf_biomass - leaf_refuge_floor).max(0.0)
                    } else {
                        state.leaf_biomass
                    };
                    let max_leaf_take = grazeable_leaf * (0.12 + 0.25 * hunger) * dt * 1.6;
                    let amount = grazeable_leaf.min((raw_demand / leaf_energy_density).min(max_leaf_take));
                    state.leaf_biomass -= amount;
                    if rooted {
                        state.leaf_biomass = state.leaf_biomass.max(leaf_refuge_floor);
                    }
                    state.epiphyte_load = (state.epiphyte_load - (0.06 + 0.10 * hunger)).max(0.0);
                    amount
                } else {
                    0.0
                };

                let reserve_removed = if rooted {
                    0.0
                } else if let Ok(mut prey_energy) = world.get::<&mut Energy>(prey) {
                    let used_from_leaf = leaf_removed * 6.0;
                    let reserve_need = (raw_demand - used_from_leaf).max(0.0);
                    // Research note: producer-derived dissolved/labile organic
                    // carbon can fuel heterotrophic microbial loops (Liu et al.,
                    // 2020), so simple consumers should be able to harvest more
                    // than hard tissue alone when grazing active producer colonies.
                    let max_take = prey_energy
                        .current
                        .min(prey_energy.max * (0.16 + 0.18 * hunger) * dt * 3.0);
                    let amount = reserve_need.min(max_take);
                    prey_energy.current -= amount;
                    amount
                } else {
                    0.0
                };

                caloric_gain = leaf_removed * 8.0 * nutritional_value
                    + reserve_removed * nutritional_value * 2.8;
            } else {
                // Partial grazing of non-plant producers (food / detritus).
                let detritus_bonus = if world.get::<&Detritus>(prey).is_ok() {
                    world
                        .get::<&FeedingCapability>(pred)
                        .map(|feeding| {
                                1.0 + (feeding.graze_skill * (1.0 - feeding.hunt_skill).clamp(0.0, 1.0))
                                .clamp(0.0, 1.0)
                                * 0.24
                        })
                        .unwrap_or(1.0)
                } else {
                    1.0
                };
                let graze_amount = world
                    .get::<&Energy>(prey)
                    .map(|e| {
                        desired_gain.min(
                            e.current
                                .min(e.max * (0.15 + 0.26 * hunger) * dt * 2.5 * detritus_bonus),
                        )
                    })
                    .unwrap_or(0.0);

                if graze_amount <= 0.0 {
                    continue;
                }

                if let Ok(mut prey_energy) = world.get::<&mut Energy>(prey) {
                    prey_energy.current -= graze_amount;
                }
                caloric_gain = graze_amount;
            }

            if let Ok(mut pred_energy) = world.get::<&mut Energy>(pred) {
                let assimilated = caloric_gain * efficiency;
                pred_energy.current = (pred_energy.current + assimilated).min(pred_energy.max);
                total_assimilation += assimilated;
                if let Ok(mut state) = world.get::<&mut crate::components::ConsumerState>(pred) {
                    state.recent_assimilation = (state.recent_assimilation
                        + assimilated / pred_energy.max.max(1e-3))
                    .min(1.5);
                }
            }
        } else {
            // Predation: kill mobile prey, transfer energy
            let prey_caloric_value = world.get::<&Energy>(prey).map(|e| e.max).unwrap_or(10.0);

            if let Ok(mut pred_energy) = world.get::<&mut Energy>(pred) {
                let assimilated = prey_caloric_value * 0.8;
                pred_energy.current = (pred_energy.current + assimilated).min(pred_energy.max);
                total_assimilation += assimilated;
                if let Ok(mut state) = world.get::<&mut crate::components::ConsumerState>(pred) {
                    state.recent_assimilation = (state.recent_assimilation
                        + assimilated / pred_energy.max.max(1e-3))
                    .min(1.5);
                }
            }

            // Set prey energy to 0 so death_system removes it
            if let Ok(mut prey_energy) = world.get::<&mut Energy>(prey) {
                prey_energy.current = 0.0;
            }

            dead.push(prey);
        }

        if let Ok(mut needs) = world.get::<&mut Needs>(pred) {
            let satiation = world
                .get::<&Energy>(pred)
                .map(|e| (e.current / e.max.max(1e-3)).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            needs.hunger = (needs.hunger - satiation * 0.65).max(0.0);
        }
    }
    FeedingSummary {
        dead,
        total_assimilation,
    }
}

/// Result of the death system: how many creatures vs non-creatures died.
pub struct DeathResult {
    pub creature_deaths: u64,
    pub producer_deaths: u64,
    pub total_removed: u64,
    /// Positions and energy of dead creatures for nutrient cycling (detritus spawning).
    pub dead_creature_info: Vec<(f32, f32, f32)>,
    /// Plant biomass returned to the dissolved/sediment nutrient pools.
    pub recycled_plant_nutrients: (f32, f32),
}

/// Remove dead entities (energy <= 0 or age exceeded).
pub fn death_system(world: &mut World) -> DeathResult {
    let mut dead = HashSet::new();

    for (entity, energy) in &mut world.query::<(Entity, &Energy)>() {
        if energy.current <= 0.0 {
            if world.get::<&crate::genome::ProducerGenome>(entity).is_ok() {
                let can_resprout = world
                    .get::<&crate::components::ProducerState>(entity)
                    .map(|state| {
                        state.belowground_reserve > 0.05
                            || state.meristem_bank > 0.02
                            || state.structural_biomass > 0.08
                    })
                    .unwrap_or(false);
                if can_resprout {
                    continue;
                }
            }
            dead.insert(entity);
        }
    }
    for (entity, age) in &mut world.query::<(Entity, &Age)>() {
        // Research note: plants are not removed on a deterministic age threshold.
        // Senescence is handled in producer_ecology_system as gradual turnover.
        if world.get::<&crate::genome::ProducerGenome>(entity).is_ok() {
            continue;
        }
        if age.ticks >= age.max_ticks {
            dead.insert(entity); // HashSet deduplicates automatically
        }
    }

    let mut creature_deaths = 0u64;
    let mut producer_deaths = 0u64;
    let mut dead_creature_info = Vec::new();
    let mut recycled_plant_nutrients = (0.0, 0.0);

    for &entity in &dead {
        if world.get::<&crate::genome::CreatureGenome>(entity).is_ok() {
            creature_deaths += 1;
            // Collect position and energy for detritus spawning
            let pos = world.get::<&Position>(entity).ok().map(|p| (p.x, p.y));
            let max_e = world
                .get::<&Energy>(entity)
                .ok()
                .map(|e| e.max)
                .unwrap_or(0.0);
            if let Some((x, y)) = pos {
                dead_creature_info.push((x, y, max_e));
            }
        } else if let (Ok(genome), Ok(state)) = (
            world.get::<&crate::genome::ProducerGenome>(entity),
            world.get::<&crate::components::ProducerState>(entity),
        ) {
            producer_deaths += 1;
            let biomass = state.total_biomass() + genome.producer_mass();
            recycled_plant_nutrients.0 += biomass * 0.35;
            recycled_plant_nutrients.1 += biomass * 0.08;
        }
    }

    let total_removed = dead.len() as u64;

    for entity in &dead {
        let _ = world.despawn(*entity);
    }

    DeathResult {
        creature_deaths,
        producer_deaths,
        total_removed,
        dead_creature_info,
        recycled_plant_nutrients,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn default_env() -> Environment {
        Environment::default()
    }

    #[test]
    fn test_metabolism_drains_energy() {
        let mut world = World::new();
        let physics = DerivedPhysics {
            base_metabolism: 1.0,
            max_speed: 5.0,
            acceleration: 5.0,
            turn_radius: 1.0,
            drag_coefficient: 0.5,
            body_mass: 1.0,
            max_energy: 50.0,
            visual_profile: 0.5,
            camouflage: 0.5,
            sensory_range: 10.0,
        };
        let e = world.spawn((Energy::new(50.0), physics, Position { x: 10.0, y: 10.0 }));

        let initial = world.get::<&Energy>(e).unwrap().current;
        metabolism_system(
            &mut world,
            1.0,
            &default_env(),
            24.0,
            &EcologyCalibration::default(),
        );
        let after = world.get::<&Energy>(e).unwrap().current;
        assert!(
            after < initial,
            "Energy should decrease: {} -> {}",
            initial,
            after
        );
    }

    #[test]
    fn test_death_removes_zero_energy() {
        let mut world = World::new();
        let e = world.spawn((Energy {
            current: 0.0,
            max: 50.0,
        },));
        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 1);
        assert!(
            world.get::<&Energy>(e).is_err(),
            "Entity should be despawned"
        );
    }

    #[test]
    fn test_death_removes_old_age() {
        let mut world = World::new();
        let _e = world.spawn((
            Energy::new(50.0),
            Age {
                ticks: 1000,
                max_ticks: 500,
            },
        ));
        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 1);
    }

    #[test]
    fn test_energy_fraction() {
        let e = Energy {
            current: 25.0,
            max: 50.0,
        };
        assert!((e.fraction() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_consumer_adulthood_does_not_revert_after_maturation() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(7);
        let genome = crate::genome::CreatureGenome::minimal_cell(&mut rng);
        let physics = crate::phenotype::derive_physics(&genome);
        let entity = world.spawn((
            Energy::new_with(physics.max_energy * 0.22, physics.max_energy),
            Age {
                ticks: 18_000,
                max_ticks: 24_000,
            },
            genome,
            physics,
            crate::needs::Needs::default(),
            crate::components::ConsumerState {
                reserve_buffer: 0.24,
                maturity_progress: 1.0,
                matured_once: false,
                reproductive_buffer: 0.05,
                brood_cooldown: 0.0,
                recent_assimilation: 0.04,
            },
        ));

        consumer_life_history_system(&mut world, 4.0, 48.0, 16.0);

        let state = world
            .get::<&crate::components::ConsumerState>(entity)
            .expect("consumer should still exist after lifecycle update");
        assert!(state.matured_once);
        assert!(state.is_adult());
        assert!(
            (state.maturity_progress - 1.0).abs() < f32::EPSILON,
            "adult maturity should remain pinned at 1.0 after first maturation, got {:.3}",
            state.maturity_progress,
        );
    }

    #[test]
    fn test_producer_ecology_recovers_depleted_reserve_faster_than_full_reserve() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = crate::genome::ProducerGenome::minimal_producer(&mut rng);
        genome.photosynthesis_rate = 1.10;
        genome.reserve_allocation = 0.60;
        genome.hardiness = 0.50;
        genome.nutrient_affinity = 0.60;
        let physics = crate::phenotype::derive_producer_physics(&genome);
        let state = crate::components::ProducerState {
            structural_biomass: genome.support_target_biomass() * 0.9,
            leaf_biomass: genome.active_target_biomass() * 0.9,
            belowground_reserve: genome.support_target_biomass() * 0.28,
            meristem_bank: 0.35,
            epiphyte_load: 0.0,
            seed_cooldown: 0.0,
            clonal_cooldown: 0.0,
            stress_time: 0.0,
            propagule_kind: None,
        };
        let env = Environment::default();
        let mut nutrients = NutrientPool::default();
        let mut light_field = LightField::new(20, 20);

        let depleted = world.spawn((
            Position { x: 5.0, y: 2.0 },
            crate::components::BoundingBox { w: 2.0, h: 3.0 },
            Energy {
                current: physics.max_energy * 0.3,
                max: physics.max_energy,
            },
            physics.clone(),
            genome.clone(),
            state.clone(),
            Age {
                ticks: 0,
                max_ticks: 20_000,
            },
        ));
        let full = world.spawn((
            Position { x: 10.0, y: 2.0 },
            crate::components::BoundingBox { w: 2.0, h: 3.0 },
            Energy {
                current: physics.max_energy * 0.85,
                max: physics.max_energy,
            },
            physics,
            genome,
            state,
            Age {
                ticks: 0,
                max_ticks: 20_000,
            },
        ));

        light_field.rebuild(&world);
        let before_depleted = world.get::<&Energy>(depleted).unwrap().fraction();
        let before_full = world.get::<&Energy>(full).unwrap().fraction();

        for _ in 0..120 {
            producer_ecology_system(
                &mut world,
                0.05,
                &env,
                20.0,
                &light_field,
                &mut nutrients,
                &EcologyCalibration::default(),
            );
        }

        let gained_depleted = world.get::<&Energy>(depleted).unwrap().fraction() - before_depleted;
        let gained_full = world.get::<&Energy>(full).unwrap().fraction() - before_full;

        assert!(
            gained_depleted > 0.0,
            "Depleted producer should gain energy (net of maintenance)"
        );
        assert!(
            gained_depleted > gained_full + 0.001,
            "Reserve-deficient plant should recover faster than already-full plant: {:.4} vs {:.4}",
            gained_depleted,
            gained_full,
        );
    }

    #[test]
    fn test_detritus_decays() {
        let mut world = World::new();
        let physics = DerivedPhysics {
            base_metabolism: 0.5,
            body_mass: 0.1,
            max_energy: 10.0,
            ..Default::default()
        };

        let e = world.spawn((
            Energy {
                current: 10.0,
                max: 10.0,
            },
            physics,
            Detritus,
            Position { x: 5.0, y: 5.0 },
        ));

        let initial = world.get::<&Energy>(e).unwrap().current;
        metabolism_system(
            &mut world,
            1.0,
            &default_env(),
            24.0,
            &EcologyCalibration::default(),
        );
        let after = world.get::<&Energy>(e).unwrap().current;
        assert!(
            after < initial,
            "Detritus should decay: {} -> {}",
            initial,
            after,
        );
    }

    #[test]
    fn test_death_returns_creature_info_for_detritus() {
        let mut world = World::new();
        let genome = crate::genome::CreatureGenome::minimal_cell(&mut StdRng::seed_from_u64(42));
        let physics = crate::phenotype::derive_physics(&genome);
        world.spawn((
            Position { x: 10.0, y: 5.0 },
            Energy {
                current: 0.0,
                max: 50.0,
            },
            genome,
            physics,
        ));

        let result = death_system(&mut world);
        assert_eq!(result.creature_deaths, 1);
        assert_eq!(result.dead_creature_info.len(), 1);
        let (x, y, max_e) = result.dead_creature_info[0];
        assert!((x - 10.0).abs() < 0.01);
        assert!((y - 5.0).abs() < 0.01);
        assert!((max_e - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_depth_affects_photosynthesis() {
        let mut world = World::new();
        let physics = DerivedPhysics {
            base_metabolism: -0.25,
            body_mass: 0.1,
            max_energy: 15.0,
            ..Default::default()
        };

        // Surface producer (y=2, top 8% of 24-high tank)
        let surface = world.spawn((
            Energy {
                current: 7.5,
                max: 15.0,
            },
            physics.clone(),
            Position { x: 5.0, y: 2.0 },
        ));
        // Deep plant (y=20, bottom 83% of 24-high tank)
        let deep = world.spawn((
            Energy {
                current: 7.5,
                max: 15.0,
            },
            physics,
            Position { x: 5.0, y: 20.0 },
        ));

        metabolism_system(
            &mut world,
            1.0,
            &default_env(),
            24.0,
            &EcologyCalibration::default(),
        );

        let surface_e = world.get::<&Energy>(surface).unwrap().current;
        let deep_e = world.get::<&Energy>(deep).unwrap().current;

        assert!(
            surface_e > deep_e,
            "Surface producer should gain more energy than deep: {:.4} vs {:.4}",
            surface_e,
            deep_e,
        );
    }

    // -- Helper to build an EntityInfoMap entry for prey --
    fn prey_info(
        x: f32,
        y: f32,
        body_mass: f32,
        max_speed: f32,
        is_producer: bool,
    ) -> crate::EntityInfo {
        crate::EntityInfo {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            body_mass,
            max_speed,
            is_producer,
            is_boid: false,
            max_prey_mass: 0.0,
            hunt_skill: 0.0,
            graze_skill: 0.0,
        }
    }

    #[test]
    fn test_grazing_drains_producer_partially() {
        use std::collections::HashMap;

        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(42);
        let genome = crate::genome::ProducerGenome::minimal_producer(&mut rng);
        let initial_state = crate::components::ProducerState {
            structural_biomass: genome.support_target_biomass() * 0.8,
            // Keep foliage sparse so grazer demand exhausts leaf tissue and reaches reserve.
            leaf_biomass: genome.active_target_biomass() * 0.12,
            belowground_reserve: genome.support_target_biomass() * 0.18,
            meristem_bank: 0.22,
            epiphyte_load: 0.4,
            seed_cooldown: 0.0,
            clonal_cooldown: 0.0,
            stress_time: 0.0,
            propagule_kind: None,
        };

        let plant = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: 100.0,
                max: 100.0,
            },
            Producer,
            genome.clone(),
            initial_state.clone(),
        ));

        let creature = world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy {
                current: 20.0,
                max: 100.0,
            },
            DerivedPhysics {
                max_speed: 5.0,
                sensory_range: 15.0,
                body_mass: 1.0,
                base_metabolism: 0.9,
                ..Default::default()
            },
            FeedingCapability {
                max_prey_mass: 5.0,
                graze_skill: 0.5,
                is_producer: false,
                ..Default::default()
            },
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
            Velocity { vx: 1.0, vy: 0.0 },
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(plant, prey_info(10.0, 10.0, 0.1, 0.0, true));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert_eq!(kills.len(), 1, "Should find one grazing interaction");

        let dead = apply_kills(&mut world, &kills, 0.05);
        assert!(
            dead.dead.is_empty(),
            "Producer should not be killed by grazing"
        );

        let plant_energy = world.get::<&Energy>(plant).unwrap().current;
        let plant_state = world
            .get::<&crate::components::ProducerState>(plant)
            .unwrap();
        assert!(plant_energy > 0.0, "Producer should survive grazing");
        assert!(
            plant_state.leaf_biomass < initial_state.leaf_biomass,
            "Leaf biomass should be reduced by grazing"
        );
        assert!(
            plant_state.epiphyte_load < initial_state.epiphyte_load,
            "Grazing should also scrub some epiphyte load"
        );
        assert!(
            plant_energy < 100.0,
            "Reserve should also be grazed when appetite exceeds leaf-only intake, got {}",
            plant_energy
        );

        let creature_energy = world.get::<&Energy>(creature).unwrap().current;
        assert!(
            creature_energy > 20.0,
            "Creature should gain energy from grazing, got {}",
            creature_energy
        );
    }

    #[test]
    fn test_predation_kills_mobile_prey() {
        use std::collections::HashMap;

        let mut world = World::new();

        // Mobile prey — small, slow creature (no Producer marker)
        let prey = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: 40.0,
                max: 50.0,
            },
        ));

        // Predator — fast, high hunt_skill
        let predator = world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy {
                current: 30.0,
                max: 200.0,
            },
            DerivedPhysics {
                max_speed: 8.0,
                sensory_range: 15.0,
                body_mass: 3.0,
                ..Default::default()
            },
            FeedingCapability {
                max_prey_mass: 5.0,
                hunt_skill: 0.6,
                is_producer: false,
                ..Default::default()
            },
            Velocity { vx: 5.0, vy: 0.0 },
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(prey, prey_info(10.0, 10.0, 0.5, 3.0, false));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert_eq!(kills.len(), 1, "Should find one predation interaction");

        let dead = apply_kills(&mut world, &kills, 0.05);
        assert_eq!(dead.dead.len(), 1, "Prey should be marked dead");
        assert_eq!(dead.dead[0], prey);

        let prey_energy = world.get::<&Energy>(prey).unwrap().current;
        assert!(
            prey_energy <= 0.0,
            "Prey energy should be 0 after predation, got {}",
            prey_energy
        );

        // predator gains prey.max * 0.8 = 50.0 * 0.8 = 40.0 → 30.0 + 40.0 = 70.0
        let pred_energy = world.get::<&Energy>(predator).unwrap().current;
        assert!(
            (pred_energy - 70.0).abs() < 0.01,
            "Predator should gain 80% of prey's max energy: expected ~70, got {}",
            pred_energy
        );
    }

    #[test]
    fn test_low_hunt_skill_cannot_catch_prey() {
        use std::collections::HashMap;

        let mut world = World::new();

        let prey = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: 40.0,
                max: 50.0,
            },
        ));

        // hunt_skill = 0.1, below the 0.3 threshold
        world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy {
                current: 30.0,
                max: 100.0,
            },
            DerivedPhysics {
                max_speed: 8.0,
                sensory_range: 15.0,
                body_mass: 3.0,
                ..Default::default()
            },
            FeedingCapability {
                max_prey_mass: 5.0,
                hunt_skill: 0.1,
                is_producer: false,
                ..Default::default()
            },
            Velocity { vx: 5.0, vy: 0.0 },
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(prey, prey_info(10.0, 10.0, 0.5, 3.0, false));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert!(
            kills.is_empty(),
            "Low hunt_skill creature should not catch prey, got {} kills",
            kills.len()
        );
    }

    #[test]
    fn test_slow_predator_cannot_catch_fast_prey() {
        use std::collections::HashMap;

        let mut world = World::new();

        let prey = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: 40.0,
                max: 50.0,
            },
        ));

        // pred_speed 5.0 < prey.max_speed 10.0 * 0.8 = 8.0
        world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy {
                current: 30.0,
                max: 100.0,
            },
            DerivedPhysics {
                max_speed: 5.0,
                sensory_range: 15.0,
                body_mass: 3.0,
                ..Default::default()
            },
            FeedingCapability {
                max_prey_mass: 5.0,
                hunt_skill: 0.6,
                is_producer: false,
                ..Default::default()
            },
            Velocity { vx: 3.0, vy: 0.0 },
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(prey, prey_info(10.0, 10.0, 0.5, 10.0, false));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert!(
            kills.is_empty(),
            "Slow predator should not catch fast prey, got {} kills",
            kills.len()
        );
    }

    #[test]
    fn test_creature_metabolism_slower_at_depth() {
        let mut world = World::new();
        let physics = DerivedPhysics {
            base_metabolism: 2.0,
            max_speed: 5.0,
            body_mass: 1.0,
            ..Default::default()
        };

        // Surface creature at y=5 (10% of 50-height tank → metabolism_at_depth = 1.0)
        let surface = world.spawn((
            Energy::new(100.0),
            physics.clone(),
            Position { x: 10.0, y: 5.0 },
        ));

        // Deep creature at y=45 (90% of 50-height tank → metabolism_at_depth = 0.85)
        let deep = world.spawn((Energy::new(100.0), physics, Position { x: 10.0, y: 45.0 }));

        let initial = world.get::<&Energy>(surface).unwrap().current;

        metabolism_system(
            &mut world,
            1.0,
            &default_env(),
            50.0,
            &EcologyCalibration::default(),
        );

        let loss_surface = initial - world.get::<&Energy>(surface).unwrap().current;
        let loss_deep = initial - world.get::<&Energy>(deep).unwrap().current;

        assert!(
            loss_surface > loss_deep,
            "Surface creature should lose MORE energy: surface lost {:.4}, deep lost {:.4}",
            loss_surface,
            loss_deep
        );

        let ratio = loss_deep / loss_surface;
        assert!(
            (ratio - 0.85).abs() < 0.01,
            "Deep metabolism should be 85% of surface: ratio = {:.4}",
            ratio
        );
    }

    #[test]
    fn test_grazing_requires_minimum_skill() {
        use std::collections::HashMap;

        let mut world = World::new();

        let plant = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: 100.0,
                max: 100.0,
            },
            Producer,
        ));

        // graze_skill = 0.1, below the 0.2 threshold
        world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy {
                current: 20.0,
                max: 100.0,
            },
            DerivedPhysics {
                max_speed: 5.0,
                sensory_range: 15.0,
                body_mass: 1.0,
                ..Default::default()
            },
            FeedingCapability {
                max_prey_mass: 5.0,
                graze_skill: 0.1,
                is_producer: false,
                ..Default::default()
            },
            Velocity { vx: 1.0, vy: 0.0 },
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(plant, prey_info(10.0, 10.0, 0.1, 0.0, true));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert!(
            kills.is_empty(),
            "Creature with low graze_skill should not graze, got {} kills",
            kills.len()
        );
    }

    #[test]
    fn test_night_metabolism_stress() {
        // Day world: light_level = 1.0 → night_stress = 1.0
        let mut world_day = World::new();
        let physics = DerivedPhysics {
            base_metabolism: 1.0,
            max_speed: 5.0,
            acceleration: 5.0,
            turn_radius: 1.0,
            drag_coefficient: 0.5,
            body_mass: 1.0,
            max_energy: 100.0,
            visual_profile: 0.5,
            camouflage: 0.5,
            sensory_range: 10.0,
        };
        let e_day = world_day.spawn((
            Energy::new(100.0),
            physics.clone(),
            Position { x: 50.0, y: 5.0 },
        ));
        let initial_day = world_day.get::<&Energy>(e_day).unwrap().current;

        let mut env_day = Environment::default();
        env_day.light_level = 1.0;
        metabolism_system(
            &mut world_day,
            1.0,
            &env_day,
            50.0,
            &EcologyCalibration::default(),
        );
        let after_day = world_day.get::<&Energy>(e_day).unwrap().current;
        let loss_day = initial_day - after_day;

        // Night world: light_level = 0.0 → night_stress = 1.3
        let mut world_night = World::new();
        let e_night =
            world_night.spawn((Energy::new(100.0), physics, Position { x: 50.0, y: 5.0 }));
        let initial_night = world_night.get::<&Energy>(e_night).unwrap().current;

        let mut env_night = Environment::default();
        env_night.light_level = 0.0;
        metabolism_system(
            &mut world_night,
            1.0,
            &env_night,
            50.0,
            &EcologyCalibration::default(),
        );
        let after_night = world_night.get::<&Energy>(e_night).unwrap().current;
        let loss_night = initial_night - after_night;

        assert!(loss_day > 0.0, "Day creature should lose energy");
        assert!(
            loss_night > loss_day,
            "Night creature should lose more energy than day"
        );

        let ratio = loss_night / loss_day;
        assert!(
            (ratio - 1.3).abs() < 0.05,
            "Night energy loss should be ~30% higher than day: ratio={:.3}",
            ratio,
        );
    }

    #[test]
    fn test_temperature_affects_creature_metabolism() {
        let physics = DerivedPhysics {
            base_metabolism: 1.0,
            max_speed: 5.0,
            acceleration: 5.0,
            turn_radius: 1.0,
            drag_coefficient: 0.5,
            body_mass: 1.0,
            max_energy: 100.0,
            visual_profile: 0.5,
            camouflage: 0.5,
            sensory_range: 10.0,
        };

        // Normal temperature (25°C → temp_modifier = 1.0)
        let mut world_normal = World::new();
        let e_normal = world_normal.spawn((
            Energy::new(100.0),
            physics.clone(),
            Position { x: 50.0, y: 5.0 },
        ));
        let initial_normal = world_normal.get::<&Energy>(e_normal).unwrap().current;

        let mut env_normal = Environment::default();
        env_normal.temperature = 25.0;
        env_normal.light_level = 1.0; // eliminate night stress
        metabolism_system(
            &mut world_normal,
            1.0,
            &env_normal,
            50.0,
            &EcologyCalibration::default(),
        );
        let loss_normal = initial_normal - world_normal.get::<&Energy>(e_normal).unwrap().current;

        // Cold snap (20°C → temp_modifier = 1.0 + (20-25)*0.02 = 0.9)
        let mut world_cold = World::new();
        let e_cold = world_cold.spawn((Energy::new(100.0), physics, Position { x: 50.0, y: 5.0 }));
        let initial_cold = world_cold.get::<&Energy>(e_cold).unwrap().current;

        let mut env_cold = Environment::default();
        env_cold.temperature = 20.0;
        env_cold.light_level = 1.0;
        metabolism_system(
            &mut world_cold,
            1.0,
            &env_cold,
            50.0,
            &EcologyCalibration::default(),
        );
        let loss_cold = initial_cold - world_cold.get::<&Energy>(e_cold).unwrap().current;

        assert!(loss_normal > 0.0, "Normal temp creature should lose energy");
        assert!(loss_cold > 0.0, "Cold creature should still lose energy");
        assert!(
            loss_cold < loss_normal,
            "Cold creature should burn LESS energy (cold-blooded slows down): cold={:.4} vs normal={:.4}",
            loss_cold, loss_normal,
        );

        let ratio = loss_cold / loss_normal;
        assert!(
            (ratio - 0.9).abs() < 0.05,
            "Cold metabolism should be ~90% of normal: ratio={:.3}",
            ratio,
        );
    }

    #[test]
    fn test_temperature_affects_photosynthesis() {
        let plant_physics = DerivedPhysics {
            base_metabolism: -0.5, // negative = photosynthetic producer
            body_mass: 0.1,
            max_energy: 50.0,
            ..Default::default()
        };

        // Normal temperature (25°C)
        let mut world_normal = World::new();
        let p_normal = world_normal.spawn((
            Energy {
                current: 20.0,
                max: 50.0,
            },
            plant_physics.clone(),
            Position { x: 5.0, y: 2.0 },
        ));
        let initial_normal = world_normal.get::<&Energy>(p_normal).unwrap().current;

        let mut env_normal = Environment::default();
        env_normal.temperature = 25.0;
        env_normal.light_level = 1.0;
        metabolism_system(
            &mut world_normal,
            1.0,
            &env_normal,
            50.0,
            &EcologyCalibration::default(),
        );
        let gain_normal = world_normal.get::<&Energy>(p_normal).unwrap().current - initial_normal;

        // Cold snap (20°C → temp_modifier = 0.9)
        let mut world_cold = World::new();
        let p_cold = world_cold.spawn((
            Energy {
                current: 20.0,
                max: 50.0,
            },
            plant_physics,
            Position { x: 5.0, y: 2.0 },
        ));
        let initial_cold = world_cold.get::<&Energy>(p_cold).unwrap().current;

        let mut env_cold = Environment::default();
        env_cold.temperature = 20.0;
        env_cold.light_level = 1.0;
        metabolism_system(
            &mut world_cold,
            1.0,
            &env_cold,
            50.0,
            &EcologyCalibration::default(),
        );
        let gain_cold = world_cold.get::<&Energy>(p_cold).unwrap().current - initial_cold;

        assert!(
            gain_normal > 0.0,
            "Producer should gain energy via photosynthesis"
        );
        assert!(gain_cold > 0.0, "Cold producer should still gain energy");
        assert!(
            gain_cold < gain_normal,
            "Cold producer should gain LESS energy: cold={:.4} vs normal={:.4}",
            gain_cold,
            gain_normal,
        );
    }

    #[test]
    fn test_complex_creature_survives_night_better() {
        use crate::genome::CreatureGenome;
        use crate::phenotype::derive_physics;

        let mut rng = StdRng::seed_from_u64(42);
        let mut genome_simple = CreatureGenome::random(&mut rng);
        genome_simple.complexity = 0.0;
        genome_simple.art.body_size = 1.0;
        genome_simple.art.body_elongation = 0.5;
        genome_simple.behavior.speed_factor = 1.0;
        genome_simple.behavior.metabolism_factor = 1.0;

        let mut genome_complex = genome_simple.clone();
        genome_complex.complexity = 0.5;

        let physics_simple = derive_physics(&genome_simple);
        let physics_complex = derive_physics(&genome_complex);

        // Complex creatures should have lower base metabolism
        assert!(
            physics_complex.base_metabolism < physics_simple.base_metabolism,
            "Complex creature should have lower base_metabolism: {:.4} vs {:.4}",
            physics_complex.base_metabolism,
            physics_simple.base_metabolism,
        );

        // Run both through night metabolism
        let mut world_simple = World::new();
        let max_e = 100.0;
        let e_simple = world_simple.spawn((
            Energy {
                current: max_e,
                max: max_e,
            },
            physics_simple,
            Position { x: 50.0, y: 5.0 },
        ));

        let mut world_complex = World::new();
        let e_complex = world_complex.spawn((
            Energy {
                current: max_e,
                max: max_e,
            },
            physics_complex,
            Position { x: 50.0, y: 5.0 },
        ));

        let mut env_night = Environment::default();
        env_night.light_level = 0.0; // full night stress

        metabolism_system(
            &mut world_simple,
            1.0,
            &env_night,
            50.0,
            &EcologyCalibration::default(),
        );
        metabolism_system(
            &mut world_complex,
            1.0,
            &env_night,
            50.0,
            &EcologyCalibration::default(),
        );

        let loss_simple = max_e - world_simple.get::<&Energy>(e_simple).unwrap().current;
        let loss_complex = max_e - world_complex.get::<&Energy>(e_complex).unwrap().current;

        assert!(
            loss_complex < loss_simple,
            "Complex creature should lose less energy at night: complex={:.4} vs simple={:.4}",
            loss_complex,
            loss_simple,
        );

        // Verify the ratio matches complexity_efficiency: 1.0 - 0.25 * 0.5 = 0.875
        let ratio = loss_complex / loss_simple;
        assert!(
            (ratio - 0.875).abs() < 0.05,
            "Energy loss ratio should be ~0.875 (complexity 0.5): got {:.4}",
            ratio,
        );
    }

    #[test]
    fn test_grazing_efficiency_scales_with_complexity() {
        use std::collections::HashMap;

        // Test that complex creatures extract more energy from grazing
        let complexities = [0.0, 0.3, 0.6, 1.0];
        let mut gains = Vec::new();

        for &c in &complexities {
            let mut world = World::new();

            let plant = world.spawn((
                Position { x: 10.0, y: 10.0 },
                Energy {
                    current: 100.0,
                    max: 100.0,
                },
                Producer,
            ));

            let mut genome = CreatureGenome::minimal_cell(&mut StdRng::seed_from_u64(42));
            genome.complexity = c;

            let creature = world.spawn((
                Position { x: 11.0, y: 10.0 },
                Energy {
                    current: 20.0,
                    max: 100.0,
                },
                DerivedPhysics {
                    max_speed: 5.0,
                    sensory_range: 15.0,
                    body_mass: 1.0,
                    ..Default::default()
                },
                FeedingCapability {
                    max_prey_mass: 5.0,
                    graze_skill: 0.5,
                    is_producer: false,
                    ..Default::default()
                },
                genome,
                Velocity { vx: 1.0, vy: 0.0 },
                Needs {
                    hunger: 1.0,
                    ..Default::default()
                },
            ));

            let mut grid = SpatialGrid::new(5.0);
            grid.rebuild(&world);

            let mut entity_map: EntityInfoMap = HashMap::new();
            entity_map.insert(plant, prey_info(10.0, 10.0, 0.1, 0.0, true));

            let kills = hunting_check(&world, &grid, &entity_map);
            apply_kills(&mut world, &kills, 0.05);

            let gain = world.get::<&Energy>(creature).unwrap().current - 20.0;
            gains.push((c, gain));
        }

        // Each higher complexity should yield more energy
        for i in 1..gains.len() {
            assert!(
                gains[i].1 > gains[i - 1].1,
                "Higher complexity should yield more grazing energy: \
                 complexity {:.1} got {:.2}, complexity {:.1} got {:.2}",
                gains[i].0,
                gains[i].1,
                gains[i - 1].0,
                gains[i - 1].1,
            );
        }
    }

    #[test]
    fn test_grazing_efficiency_floor_is_sixty_percent() {
        // Grazing efficiency = 0.6 + 0.4 * complexity.max(0.1)
        // At complexity=0.0 (clamped to 0.1): 0.6 + 0.04 = 0.64
        // At complexity=1.0: 0.6 + 0.4 = 1.0
        let eff_simple = 0.6 + 0.4 * 0.0_f32.max(0.1);
        let eff_complex = 0.6 + 0.4 * 1.0_f32.max(0.1);
        assert!(
            (eff_simple - 0.64).abs() < 1e-5,
            "Simple efficiency should be 0.64: {eff_simple}"
        );
        assert!(
            (eff_complex - 1.0).abs() < 1e-5,
            "Complex efficiency should be 1.0: {eff_complex}"
        );
        // Verify monotonically increasing
        for cx in [0.0_f32, 0.1, 0.2, 0.5, 0.8, 1.0] {
            let eff = 0.6 + 0.4 * cx.max(0.1);
            assert!(
                eff >= 0.64 && eff <= 1.0,
                "Efficiency out of range at cx={cx}: {eff}"
            );
        }
    }

    #[test]
    fn test_soft_external_loading_restores_depleted_nutrients() {
        let mut pool = NutrientPool {
            dissolved_n: 0.0,
            dissolved_p: 0.0,
            sediment_n: 0.0,
            sediment_p: 0.0,
            phytoplankton_load: 0.15,
        };
        let env = default_env();
        let cal = EcologyCalibration::default();
        pool.tick(&env, 1.0, &cal);
        assert!(
            pool.dissolved_n > 0.0,
            "Soft loading should restore some dissolved N: {}",
            pool.dissolved_n
        );
        assert!(
            pool.dissolved_p > 0.0,
            "Soft loading should restore some dissolved P: {}",
            pool.dissolved_p
        );
    }

    #[test]
    fn test_nitrogen_fixation_restores_low_n() {
        let mut pool = NutrientPool {
            dissolved_n: 2.0,
            dissolved_p: 12.0,
            sediment_n: 0.0,
            sediment_p: 0.0,
            phytoplankton_load: 0.15,
        };
        let env = default_env();
        let cal = EcologyCalibration::default();
        let before = pool.dissolved_n;
        pool.tick(&env, 1.0, &cal);
        assert!(
            pool.dissolved_n > before,
            "N fixation should increase dissolved N: {} -> {}",
            before,
            pool.dissolved_n
        );
    }

    #[test]
    fn test_deficit_sensitive_benthic_release_increases_when_poor() {
        let mut pool = NutrientPool::default();
        let env = default_env();
        let cal = EcologyCalibration::default();
        let baseline_sed_p = pool.sediment_p;
        pool.tick(&env, 1.0, &cal);
        let baseline_release_p = baseline_sed_p - pool.sediment_p;

        let mut depleted_pool = NutrientPool {
            dissolved_n: 6.0,
            dissolved_p: 0.2,
            sediment_n: 220.0,
            sediment_p: 44.0,
            phytoplankton_load: 0.15,
        };
        let depleted_sed_p = depleted_pool.sediment_p;
        depleted_pool.tick(&env, 1.0, &cal);
        let depleted_release_p = depleted_sed_p - depleted_pool.sediment_p;

        assert!(
            depleted_release_p > baseline_release_p,
            "P release should increase under dissolved-P deficit: baseline={baseline_release_p:.4} depleted={depleted_release_p:.4}",
        );
        assert!(
            depleted_pool.dissolved_p > 0.2,
            "Low dissolved P should recover under deficit-sensitive release"
        );
    }

    #[test]
    fn test_high_dissolved_p_soft_export_reduces_runaway() {
        let mut pool = NutrientPool {
            dissolved_n: 40.0,
            dissolved_p: 220.0,
            sediment_n: 180.0,
            sediment_p: 60.0,
            phytoplankton_load: 0.20,
        };
        let env = default_env();
        let cal = EcologyCalibration::default();
        let before = pool.dissolved_p;

        pool.tick(&env, 1.0, &cal);

        assert!(
            pool.dissolved_p < before,
            "Soft export should reduce extreme dissolved P: before={before:.2} after={:.2}",
            pool.dissolved_p
        );
    }

    #[test]
    fn test_phytoplankton_contributes_pelagic_primary_production() {
        let mut world = World::new();
        let mut nutrients = NutrientPool::default();
        let light_field = LightField::new(40, 20);
        let env = default_env();
        let flux = producer_ecology_system(
            &mut world,
            0.25,
            &env,
            20.0,
            &light_field,
            &mut nutrients,
            &EcologyCalibration::default(),
        );

        assert!(
            flux.net_primary_production > 0.0,
            "Pelagic phytoplankton should contribute primary production even without rooted plants"
        );
        assert!(
            flux.labile_detritus_energy > 0.0,
            "Pelagic phytoplankton should leak some production into the detrital loop"
        );
        assert!(
            nutrients.phytoplankton_load >= 0.0 && nutrients.phytoplankton_load <= 0.85,
            "Phytoplankton load should stay in the configured bounds"
        );
    }

    #[test]
    fn test_filter_feeders_gain_energy_from_phytoplankton() {
        let mut world = World::new();
        let mut genome = crate::genome::CreatureGenome::minimal_cell(&mut StdRng::seed_from_u64(7));
        genome.behavior.aggression = 0.05;
        genome.behavior.mouth_size = 0.35;
        genome.behavior.hunting_instinct = 0.0;
        genome.behavior.pheromone_sensitivity = 0.55;
        let physics = crate::phenotype::derive_physics(&genome);
        let feeding = crate::phenotype::derive_feeding(&genome, &physics);
        let entity = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: physics.max_energy * 0.35,
                max: physics.max_energy,
            },
            physics,
            feeding,
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
            crate::components::ConsumerState::default(),
        ));

        let mut nutrients = NutrientPool {
            phytoplankton_load: 0.70,
            ..NutrientPool::default()
        };
        let env = default_env();
        let light_field = LightField::new(40, 20);
        let before_energy = world.get::<&Energy>(entity).unwrap().current;
        let before_phy = nutrients.phytoplankton_load;

        let flux = producer_ecology_system(
            &mut world,
            0.25,
            &env,
            20.0,
            &light_field,
            &mut nutrients,
            &EcologyCalibration::default(),
        );

        let after_energy = world.get::<&Energy>(entity).unwrap().current;
        assert!(
            after_energy > before_energy,
            "Filter-feeding grazer should gain energy from phytoplankton: before={before_energy:.3} after={after_energy:.3}"
        );
        assert!(
            nutrients.phytoplankton_load < before_phy,
            "Filter feeding should reduce phytoplankton load: before={before_phy:.3} after={:.3}",
            nutrients.phytoplankton_load
        );
        assert!(
            flux.pelagic_consumer_intake > 0.0,
            "Pelagic consumer intake should be tracked in ecology flux"
        );
    }

    #[test]
    fn test_filter_feeders_do_not_gain_energy_without_phytoplankton() {
        let mut world = World::new();
        let mut genome =
            crate::genome::CreatureGenome::minimal_cell(&mut StdRng::seed_from_u64(11));
        genome.behavior.aggression = 0.05;
        genome.behavior.mouth_size = 0.35;
        genome.behavior.hunting_instinct = 0.0;
        let physics = crate::phenotype::derive_physics(&genome);
        let feeding = crate::phenotype::derive_feeding(&genome, &physics);
        let entity = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy {
                current: physics.max_energy * 0.35,
                max: physics.max_energy,
            },
            physics,
            feeding,
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
            crate::components::ConsumerState::default(),
        ));

        let mut nutrients = NutrientPool {
            phytoplankton_load: 0.0,
            ..NutrientPool::default()
        };
        let env = default_env();
        let light_field = LightField::new(40, 20);
        let before_energy = world.get::<&Energy>(entity).unwrap().current;

        let flux = producer_ecology_system(
            &mut world,
            0.25,
            &env,
            20.0,
            &light_field,
            &mut nutrients,
            &EcologyCalibration::default(),
        );

        let after_energy = world.get::<&Energy>(entity).unwrap().current;
        assert!(
            (after_energy - before_energy).abs() < 1e-4,
            "Filter-feeding grazer should not gain energy without phytoplankton: before={before_energy:.3} after={after_energy:.3}"
        );
        assert!(
            flux.pelagic_consumer_intake <= 1e-5,
            "Pelagic consumer intake should vanish without phytoplankton, got {:.6}",
            flux.pelagic_consumer_intake
        );
    }

    #[test]
    fn test_rooted_macrophyte_grazing_preserves_basal_canopy_and_regrows() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(21);

        let mut plant_genome = crate::genome::ProducerGenome::minimal_producer(&mut rng);
        plant_genome.height_factor = 0.24;
        plant_genome.leaf_area = 0.72;
        plant_genome.branching = 0.42;
        plant_genome.hardiness = 0.55;
        plant_genome.clonal_spread = 0.48;
        plant_genome.nutrient_affinity = 1.0;
        let plant_physics = crate::phenotype::derive_producer_physics(&plant_genome);
        let leaf_target = plant_genome.active_target_biomass();
        let refuge_floor = (leaf_target * 0.18).max(0.08);
        let support_target = plant_genome.support_target_biomass();
        let belowground_target =
            support_target * (0.22 + plant_genome.hardiness * 0.22 + plant_genome.clonal_spread * 0.18);
        let plant = world.spawn((
            Position { x: 10.0, y: 12.0 },
            crate::components::BoundingBox { w: 2.0, h: 3.0 },
            Energy {
                current: plant_physics.max_energy * 0.45,
                max: plant_physics.max_energy,
            },
            plant_physics,
            Producer,
            plant_genome.clone(),
            crate::components::ProducerState {
                structural_biomass: support_target * 0.55,
                leaf_biomass: refuge_floor + 0.03,
                belowground_reserve: belowground_target * 0.85,
                meristem_bank: 0.28,
                epiphyte_load: 0.0,
                seed_cooldown: 0.0,
                clonal_cooldown: 0.0,
                stress_time: 0.0,
                propagule_kind: None,
            },
            Age {
                ticks: 2_000,
                max_ticks: 20_000,
            },
            crate::components::RootedMacrophyte,
        ));

        let mut grazer_genome = crate::genome::CreatureGenome::minimal_cell(&mut rng);
        grazer_genome.behavior.mouth_size = 0.55;
        grazer_genome.behavior.hunting_instinct = 0.0;
        grazer_genome.behavior.aggression = 0.02;
        let grazer_physics = crate::phenotype::derive_physics(&grazer_genome);
        let grazer = world.spawn((
            Energy {
                current: grazer_physics.max_energy * 0.15,
                max: grazer_physics.max_energy,
            },
            grazer_physics,
            grazer_genome,
            Needs {
                hunger: 1.0,
                ..Default::default()
            },
        ));

        let before_energy = world.get::<&Energy>(plant).unwrap().current;
        apply_kills(&mut world, &[(grazer, plant)], 0.5);

        let leaf_after_grazing = world
            .get::<&crate::components::ProducerState>(plant)
            .unwrap()
            .leaf_biomass;
        let energy_after_grazing = world.get::<&Energy>(plant).unwrap().current;
        assert!(
            leaf_after_grazing >= refuge_floor - 1e-4,
            "Rooted plant should keep a basal leaf refuge after grazing: refuge={refuge_floor:.4} leaf={:.4}",
            leaf_after_grazing,
        );
        assert!(
            (energy_after_grazing - before_energy).abs() < 1e-4,
            "Rooted plant reserve energy should not be directly mined once only the refuge remains: before={before_energy:.4} after={energy_after_grazing:.4}",
        );

        let env = default_env();
        let mut nutrients = NutrientPool::default();
        let mut light_field = LightField::new(24, 20);
        light_field.rebuild(&world);
        for _ in 0..40 {
            producer_ecology_system(
                &mut world,
                0.25,
                &env,
                20.0,
                &light_field,
                &mut nutrients,
                &EcologyCalibration::default(),
            );
            light_field.rebuild(&world);
        }

        let regrown_leaf = world
            .get::<&crate::components::ProducerState>(plant)
            .unwrap()
            .leaf_biomass;
        assert!(
            regrown_leaf > leaf_after_grazing + 0.01,
            "Rooted refuge should support regrowth after grazing: grazed={:.4} regrown={regrown_leaf:.4}",
            leaf_after_grazing,
        );
    }

    #[test]
    fn test_high_dissolved_n_soft_export_prevents_runaway() {
        let mut pool = NutrientPool {
            dissolved_n: 400.0,
            dissolved_p: 2.0,
            sediment_n: 220.0,
            sediment_p: 44.0,
            phytoplankton_load: 0.0,
        };
        let env = default_env();
        let cal = EcologyCalibration::default();
        let before = pool.dissolved_n;

        for _ in 0..20 {
            pool.tick(&env, 1.0, &cal);
        }

        assert!(
            pool.dissolved_n < before * 0.4,
            "High dissolved N should trend downward under soft export: before={before:.2} after={:.2}",
            pool.dissolved_n,
        );
    }
}
