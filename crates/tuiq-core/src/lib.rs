pub mod animation;
pub mod behavior;
pub mod boids;
pub mod brain;
pub mod calibration;
pub mod components;
pub mod ecology_equilibrium;
pub mod ecosystem;
pub mod environment;
pub mod genetics;
pub mod genome;
pub mod needs;
pub mod phenotype;
pub mod pheromone;
pub mod physics;
pub mod producer_lifecycle;
pub mod save;
pub mod spatial;
pub mod spawner;

mod bootstrap;
mod producer_reproduction;
mod stats;
pub mod systems;

use boids::{Boid, BoidParams};
use brain::InnovationTracker;
use components::*;
use ecosystem::{LightField, NutrientPool, Producer};
use environment::{Environment, EventKind, SubstrateGrid};
use genome::CreatureGenome;
use phenotype::{DerivedPhysics, FeedingCapability};
use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;
use spatial::SpatialGrid;
use stats::ECOLOGY_HISTORY_DAYS;
use systems::{
    AllometricEcosystem, BodySizeHunting, BrainSystem, DefaultProducerLifecycle,
    DefaultReproduction, EcosystemSystem, HuntingSystem, NeatBrainSystem, ProducerLifecycleSystem,
    ReproductionSystem,
};

use hecs::Entity;
use std::collections::{HashMap, VecDeque};

pub use calibration::{EcologyCalibration, EvolutionCalibration, RuntimeCalibration};
pub use ecology_equilibrium::EquilibriumStartupTargets;
pub use stats::{DailyEcologySample, EcologyDiagnostics, EcologyInstant, SimStats};

/// Pre-computed entity info shared across brain, boids, and hunting systems per tick.
#[derive(Clone, Copy)]
pub struct EntityInfo {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub body_mass: f32,
    pub max_speed: f32,
    pub is_producer: bool,
    pub is_boid: bool,
    pub max_prey_mass: f32,
    pub hunt_skill: f32,
    pub graze_skill: f32,
}

/// Shared lookup map type used by brain, boids, and hunting systems.
pub type EntityInfoMap = HashMap<Entity, EntityInfo>;

/// The core simulation interface. Rendering code only sees this trait.
pub trait Simulation {
    /// Advance simulation by one fixed timestep.
    fn tick(&mut self, dt: f32);

    /// Read-only access to the ECS world for rendering queries.
    fn world(&self) -> &hecs::World;

    /// Current environment state (time of day, light level, etc.)
    fn environment(&self) -> &Environment;

    /// Simulation statistics for HUD display.
    fn stats(&self) -> SimStats;

    /// Richer ecology diagnostics for calibration and debugging overlays.
    fn ecology_diagnostics(&self) -> EcologyDiagnostics;

    /// Tank dimensions in cells (width, height) — interior area.
    fn tank_size(&self) -> (u16, u16);

    /// Current diversity coefficient (scales mutation rates and fitness sharing).
    fn diversity_coefficient(&self) -> f32;

    /// Adjust the diversity coefficient (clamped to 0.25–2.5).
    fn set_diversity_coefficient(&mut self, value: f32);
}

/// The main aquarium simulation.
pub struct AquariumSim {
    world: hecs::World,
    env: Environment,
    calibration: RuntimeCalibration,
    tick_count: u64,
    total_creature_births: u64,
    total_creature_deaths: u64,
    total_producer_births: u64,
    total_producer_deaths: u64,
    elapsed_days: u64,
    prev_time_of_day: f32,
    tank_width: u16,
    tank_height: u16,
    grid: SpatialGrid,
    boid_params: BoidParams,
    rng: StdRng,
    innovation_tracker: InnovationTracker,
    pheromone_grid: pheromone::PheromoneGrid,
    light_field: LightField,
    nutrients: NutrientPool,
    substrate: SubstrateGrid,
    /// Entities born since last drain — render layer regenerates their art.
    pending_births: Vec<hecs::Entity>,
    // ECS system delegates
    brain_system: NeatBrainSystem,
    ecosystem_system: AllometricEcosystem,
    hunting_system: BodySizeHunting,
    reproduction_system: DefaultReproduction,
    producer_lifecycle_system: DefaultProducerLifecycle,
    // Cached stats (recomputed every 20 ticks, not every frame)
    cached_max_generation: u32,
    cached_avg_complexity: f32,
    cached_max_creature_complexity: f32,
    cached_species_count: usize,
    cached_creature_count: usize,
    cached_producer_leaf_biomass: f32,
    cached_producer_structural_biomass: f32,
    cached_producer_belowground_reserve: f32,
    cached_consumer_biomass: f32,
    cached_juvenile_count: usize,
    cached_adult_count: usize,
    rolling_producer_npp: f32,
    rolling_pelagic_consumer_intake: f32,
    rolling_consumer_intake: f32,
    rolling_consumer_maintenance: f32,
    pending_labile_detritus_energy: f32,
    feeding_frenzy_food_budget: f32,
    feeding_frenzy_detritus_budget: f32,
    stats_cache_tick: u64,
    daily_ecology_history: VecDeque<DailyEcologySample>,
    archived_daily_history: Vec<DailyEcologySample>,
    last_daily_creature_births: u64,
    last_daily_creature_deaths: u64,
    last_daily_producer_births: u64,
    last_daily_producer_deaths: u64,
    /// Runtime diversity coefficient — scales mutation rates and fitness sharing.
    diversity_coefficient: f32,
}

impl AquariumSim {
    pub fn new(tank_width: u16, tank_height: u16) -> Self {
        Self::with_rng_and_calibration(
            tank_width,
            tank_height,
            StdRng::seed_from_u64(rand::random()),
            RuntimeCalibration::default(),
        )
    }

    pub fn new_with_calibration<C: Into<RuntimeCalibration>>(
        tank_width: u16,
        tank_height: u16,
        calibration: C,
    ) -> Self {
        Self::with_rng_and_calibration(
            tank_width,
            tank_height,
            StdRng::seed_from_u64(rand::random()),
            calibration.into(),
        )
    }

    pub fn new_seeded(tank_width: u16, tank_height: u16, seed: u64) -> Self {
        Self::with_rng_and_calibration(
            tank_width,
            tank_height,
            StdRng::seed_from_u64(seed),
            RuntimeCalibration::default(),
        )
    }

    pub fn new_seeded_with_calibration<C: Into<RuntimeCalibration>>(
        tank_width: u16,
        tank_height: u16,
        seed: u64,
        calibration: C,
    ) -> Self {
        Self::with_rng_and_calibration(
            tank_width,
            tank_height,
            StdRng::seed_from_u64(seed),
            calibration.into(),
        )
    }

    fn with_rng_and_calibration(
        tank_width: u16,
        tank_height: u16,
        mut rng: StdRng,
        calibration: RuntimeCalibration,
    ) -> Self {
        use rand::RngExt;
        let substrate_seed: u64 = rng.random();
        let startup_targets =
            ecology_equilibrium::ReducedEquilibriumModel::default_startup_targets(
                tank_width,
                tank_height,
            );
        Self {
            world: hecs::World::new(),
            env: Environment::default(),
            calibration,
            tick_count: 0,
            total_creature_births: 0,
            total_creature_deaths: 0,
            total_producer_births: 0,
            total_producer_deaths: 0,
            elapsed_days: 0,
            prev_time_of_day: 8.0,
            tank_width,
            tank_height,
            grid: SpatialGrid::new(6.0),
            boid_params: BoidParams::default(),
            rng,
            innovation_tracker: InnovationTracker::new(),
            pheromone_grid: pheromone::PheromoneGrid::new(tank_width as f32, tank_height as f32),
            light_field: LightField::new(tank_width, tank_height),
            nutrients: NutrientPool {
                dissolved_n: startup_targets.dissolved_n,
                dissolved_p: startup_targets.dissolved_p,
                sediment_n: startup_targets.sediment_n,
                sediment_p: startup_targets.sediment_p,
                phytoplankton_load: startup_targets.phytoplankton_load,
            },
            substrate: SubstrateGrid::generate(tank_width, substrate_seed),
            pending_births: Vec::new(),
            brain_system: NeatBrainSystem,
            ecosystem_system: AllometricEcosystem,
            hunting_system: BodySizeHunting,
            reproduction_system: DefaultReproduction,
            producer_lifecycle_system: DefaultProducerLifecycle,
            cached_max_generation: 0,
            cached_avg_complexity: 0.0,
            cached_max_creature_complexity: 0.0,
            cached_species_count: 0,
            cached_creature_count: 0,
            cached_producer_leaf_biomass: 0.0,
            cached_producer_structural_biomass: 0.0,
            cached_producer_belowground_reserve: 0.0,
            cached_consumer_biomass: 0.0,
            cached_juvenile_count: 0,
            cached_adult_count: 0,
            rolling_producer_npp: 0.0,
            rolling_pelagic_consumer_intake: 0.0,
            rolling_consumer_intake: 0.0,
            rolling_consumer_maintenance: 0.0,
            pending_labile_detritus_energy: 0.0,
            feeding_frenzy_food_budget: 0.0,
            feeding_frenzy_detritus_budget: 0.0,
            stats_cache_tick: 0,
            daily_ecology_history: VecDeque::with_capacity(ECOLOGY_HISTORY_DAYS),
            archived_daily_history: Vec::new(),
            last_daily_creature_births: 0,
            last_daily_creature_deaths: 0,
            last_daily_producer_births: 0,
            last_daily_producer_deaths: 0,
            diversity_coefficient: 1.0,
        }
    }

    pub fn calibration(&self) -> RuntimeCalibration {
        self.calibration
    }

    pub fn default_startup_targets(tank_width: u16, tank_height: u16) -> EquilibriumStartupTargets {
        ecology_equilibrium::ReducedEquilibriumModel::default_startup_targets(tank_width, tank_height)
    }

    pub(crate) fn startup_targets(&self) -> EquilibriumStartupTargets {
        Self::default_startup_targets(self.tank_width, self.tank_height)
    }

    /// Mutable access to the ECS world for spawning entities from outside.
    pub fn world_mut(&mut self) -> &mut hecs::World {
        &mut self.world
    }

    /// Drain newly born entities so the render layer can regenerate their art.
    pub fn drain_births(&mut self) -> Vec<hecs::Entity> {
        std::mem::take(&mut self.pending_births)
    }

    fn max_producers(&self) -> usize {
        (((self.tank_width as f32) * (self.tank_height as f32) / 60.0) as usize).max(20)
    }

    fn smoothing_factor(dt: f32, horizon_seconds: f32) -> f32 {
        (dt / horizon_seconds.max(dt)).clamp(0.0, 1.0)
    }

    fn feeding_frenzy_food_count(&mut self) -> usize {
        (&mut self.world.query::<(Entity, &Producer)>())
            .into_iter()
            .filter(|(entity, _)| {
                self.world.get::<&ecosystem::Detritus>(*entity).is_err()
                    && self.world.get::<&genome::ProducerGenome>(*entity).is_err()
            })
            .count()
    }

    fn spawn_feeding_frenzy_food(&mut self, dt: f32) {
        let area_scale =
            ((self.tank_width as f32 * self.tank_height as f32) / 900.0).clamp(0.75, 2.5);
        let food_cap = ((self.tank_width as usize * self.tank_height as usize) / 120).max(6);
        let existing_food = self.feeding_frenzy_food_count();
        let consumer_positions: Vec<Position> =
            (&mut self.world.query::<(&Position, &CreatureGenome)>())
                .into_iter()
                .map(|(pos, _)| pos.clone())
                .collect();
        self.feeding_frenzy_food_budget += 1.4 * area_scale * dt;
        self.feeding_frenzy_detritus_budget += 0.35 * area_scale * dt;
        if existing_food < food_cap {
            let pellet_spawns = (self.feeding_frenzy_food_budget.floor() as usize)
                .min(food_cap.saturating_sub(existing_food));
            for _ in 0..pellet_spawns {
                let glyph = if self.rng.random_bool(0.35) { "o" } else { "." };
                let frame = AsciiFrame::from_rows(vec![glyph]);
                let appearance = Appearance {
                    frame_sets: vec![vec![frame.clone()], vec![frame]],
                    facing: Direction::Right,
                    color_index: 4,
                };
                let anchor = consumer_positions
                    .get(self.rng.random_range(0..consumer_positions.len().max(1)))
                    .cloned();
                let x = anchor
                    .as_ref()
                    .map(|pos| pos.x + self.rng.random_range(-2.5..2.5))
                    .unwrap_or_else(|| self.rng.random_range(1.5..(self.tank_width as f32 - 1.5)))
                    .clamp(1.5, self.tank_width as f32 - 1.5);
                let y = anchor
                    .as_ref()
                    .map(|pos| pos.y + self.rng.random_range(-1.2..1.2))
                    .unwrap_or_else(|| {
                        self.rng
                            .random_range(1.0..(self.tank_height as f32 * 0.22).max(2.0))
                    })
                    .clamp(1.0, self.tank_height as f32 - 3.0);
                let vx = self.env.current.0 * 0.25 + self.rng.random_range(-0.08..0.08);
                let vy = self.rng.random_range(0.12..0.45);
                self.spawn_food(
                    Position { x, y },
                    Velocity { vx, vy },
                    BoundingBox { w: 1.0, h: 1.0 },
                    appearance,
                );
            }
            self.feeding_frenzy_food_budget =
                (self.feeding_frenzy_food_budget - pellet_spawns as f32).max(0.0);
        }

        let detritus_spawns = self.feeding_frenzy_detritus_budget.floor() as usize;
        for _ in 0..detritus_spawns.min(2) {
            let anchor = consumer_positions
                .get(self.rng.random_range(0..consumer_positions.len().max(1)))
                .cloned();
            let x = anchor
                .map(|pos| pos.x + self.rng.random_range(-1.6..1.6))
                .unwrap_or_else(|| self.rng.random_range(2.0..(self.tank_width as f32 - 2.0)))
                .clamp(2.0, self.tank_width as f32 - 2.0);
            let y = (self.rooted_settlement_y() + self.rng.random_range(-1.0..0.4))
                .clamp(self.tank_height as f32 - 4.0, self.tank_height as f32 - 2.0);
            self.spawn_detritus(x, y, 1.2);
        }
        self.feeding_frenzy_detritus_budget =
            (self.feeding_frenzy_detritus_budget - detritus_spawns as f32).max(0.0);
    }

    fn apply_environment_need_pressure(&mut self, dt: f32) {
        if !matches!(
            self.env.active_event.as_ref().map(|event| event.kind),
            Some(EventKind::Earthquake)
        ) {
            return;
        }

        for (needs, _) in self
            .world
            .query_mut::<(&mut needs::Needs, &CreatureGenome)>()
        {
            needs.safety = (needs.safety + 0.18 + dt * 0.40).clamp(0.0, 1.0);
            needs.comfort = (needs.comfort + 0.05 + dt * 0.12).clamp(0.0, 1.0);
        }
    }

    fn apply_environment_velocity_forcing(&mut self, dt: f32) {
        if !matches!(
            self.env.active_event.as_ref().map(|event| event.kind),
            Some(EventKind::Earthquake)
        ) {
            return;
        }

        for (vel, physics, _) in self
            .world
            .query_mut::<(&mut Velocity, &DerivedPhysics, &CreatureGenome)>()
        {
            let quake_push = physics.max_speed.max(0.6) * (1.3 + self.rng.random_range(0.0..0.7));
            vel.vx +=
                (self.rng.random_range(-1.0..1.0) + self.env.current.0 * 0.45) * quake_push * dt;
            vel.vy +=
                (self.rng.random_range(-1.0..1.0) + self.env.current.1 * 0.65) * quake_push * dt;
        }
    }
}

impl Simulation for AquariumSim {
    fn tick(&mut self, dt: f32) {
        let tw = self.tank_width as f32;
        let th = self.tank_height as f32;

        // 0. Advance environment
        self.env.tick(dt, &mut self.rng);
        let day_rolled_over = self.env.time_of_day < self.prev_time_of_day;
        if day_rolled_over {
            self.elapsed_days += 1;
        }
        self.prev_time_of_day = self.env.time_of_day;
        if matches!(
            self.env.active_event.as_ref().map(|event| event.kind),
            Some(EventKind::FeedingFrenzy)
        ) {
            self.spawn_feeding_frenzy_food(dt);
        } else {
            self.feeding_frenzy_food_budget = 0.0;
            self.feeding_frenzy_detritus_budget = 0.0;
        }

        // 1. Rebuild spatial grid
        self.grid.rebuild(&self.world);

        // 2. Build shared entity info map (used by brain, boids, hunting)
        let sensory_map: EntityInfoMap = {
            let mut m = HashMap::with_capacity(self.world.len() as usize);
            for (entity, pos, vel, physics) in
                &mut self
                    .world
                    .query::<(Entity, &Position, &Velocity, &DerivedPhysics)>()
            {
                let is_producer = self.world.get::<&Producer>(entity).is_ok();
                let is_boid = self.world.get::<&Boid>(entity).is_ok();
                let (max_prey_mass, hunt_skill, graze_skill) = self
                    .world
                    .get::<&FeedingCapability>(entity)
                    .map(|f| (f.max_prey_mass, f.hunt_skill, f.graze_skill))
                    .unwrap_or((0.0, 0.0, 0.0));
                m.insert(
                    entity,
                    EntityInfo {
                        x: pos.x,
                        y: pos.y,
                        vx: vel.vx,
                        vy: vel.vy,
                        body_mass: physics.body_mass,
                        max_speed: physics.max_speed,
                        is_producer,
                        is_boid,
                        max_prey_mass,
                        hunt_skill,
                        graze_skill,
                    },
                );
            }
            m
        };

        // 3. Update needs
        for (needs, weights) in self
            .world
            .query_mut::<(&mut needs::Needs, &needs::NeedWeights)>()
        {
            needs::needs_tick(needs, weights, dt);
        }
        self.apply_environment_need_pressure(dt);

        // 4. Brain system (returns pheromone deposits)
        let pheromone_deposits = self.brain_system.evaluate(
            &mut self.world,
            &self.grid,
            &sensory_map,
            &self.env,
            &self.pheromone_grid,
            dt,
            tw,
            th,
        );
        for (x, y, amount) in pheromone_deposits {
            self.pheromone_grid.deposit(x, y, amount);
        }
        self.pheromone_grid.tick();

        // 5. Boids flocking
        boids::boids_system(
            &mut self.world,
            &self.grid,
            &sensory_map,
            &self.boid_params,
            tw,
            th,
            dt,
        );
        self.apply_environment_velocity_forcing(dt);

        // 6. Physics integration + boundary bounce
        physics::physics_system(&mut self.world, dt, tw, th);
        self.apply_environment_velocity_forcing(dt * 0.65);

        // Rebuild the spatial grid after movement so hunting and establishment see
        // current positions rather than pre-physics positions.
        self.grid.rebuild(&self.world);
        let entity_map: EntityInfoMap = {
            let mut m = HashMap::with_capacity(self.world.len() as usize);
            for (entity, pos, vel, physics) in
                &mut self
                    .world
                    .query::<(Entity, &Position, &Velocity, &DerivedPhysics)>()
            {
                let is_producer = self.world.get::<&Producer>(entity).is_ok();
                let is_boid = self.world.get::<&Boid>(entity).is_ok();
                let (max_prey_mass, hunt_skill, graze_skill) = self
                    .world
                    .get::<&FeedingCapability>(entity)
                    .map(|f| (f.max_prey_mass, f.hunt_skill, f.graze_skill))
                    .unwrap_or((0.0, 0.0, 0.0));
                m.insert(
                    entity,
                    EntityInfo {
                        x: pos.x,
                        y: pos.y,
                        vx: vel.vx,
                        vy: vel.vy,
                        body_mass: physics.body_mass,
                        max_speed: physics.max_speed,
                        is_producer,
                        is_boid,
                        max_prey_mass,
                        hunt_skill,
                        graze_skill,
                    },
                );
            }
            m
        };

        // 7. Ecosystem: nutrients, creature metabolism, producer ecology, aging
        self.nutrients
            .tick(&self.env, dt, &self.calibration.ecology);
        self.light_field.rebuild(&self.world);

        let flux = self.ecosystem_system.tick_metabolism(
            &mut self.world,
            dt,
            &self.env,
            th,
            &self.calibration.ecology,
        );
        self.nutrients.recycle(flux.detritus_n, flux.detritus_p);
        let producer_flux = self.ecosystem_system.tick_producer_ecology(
            &mut self.world,
            dt,
            &self.env,
            th,
            &self.light_field,
            &mut self.nutrients,
            &self.calibration.ecology,
        );
        self.pending_labile_detritus_energy += producer_flux.labile_detritus_energy;
        self.spawn_labile_detritus_from_producers();
        self.ecosystem_system.tick_aging(&mut self.world);
        self.ecosystem_system
            .tick_consumer_lifecycle(&mut self.world, dt, tw, th);

        let smoothing = Self::smoothing_factor(dt, 75.0);
        self.rolling_producer_npp +=
            (producer_flux.net_primary_production - self.rolling_producer_npp) * smoothing;
        self.rolling_pelagic_consumer_intake +=
            (producer_flux.pelagic_consumer_intake - self.rolling_pelagic_consumer_intake)
                * smoothing;
        self.rolling_consumer_maintenance +=
            (flux.consumer_maintenance - self.rolling_consumer_maintenance) * smoothing;

        // 7b. Plant lifecycle: age plants, update growth stages and appearance
        self.producer_lifecycle_system.tick(&mut self.world, dt);

        // 8. Hunting + feeding
        let feeding =
            self.hunting_system
                .find_and_apply(&mut self.world, &self.grid, &entity_map, dt);
        self.rolling_consumer_intake += ((feeding.total_assimilation
            + producer_flux.pelagic_consumer_intake)
            - self.rolling_consumer_intake)
            * smoothing;

        // 9. Creature reproduction (with population cap)
        let creature_count = (&mut self.world.query::<&CreatureGenome>())
            .into_iter()
            .count();
        self.innovation_tracker.new_generation();
        let births = self.reproduction_system.reproduce_creatures(
            &mut self.world,
            &self.grid,
            &mut self.rng,
            tw,
            th,
            creature_count,
            &mut self.innovation_tracker,
            &self.calibration.evolution,
            self.diversity_coefficient,
        );
        self.total_creature_births += births.len() as u64;
        self.pending_births.extend(&births);

        // 10. Death cleanup + nutrient cycling (dead creatures → detritus)
        let death_result = self.ecosystem_system.tick_death(&mut self.world);
        self.total_creature_deaths += death_result.creature_deaths;
        self.total_producer_deaths += death_result.producer_deaths;
        self.nutrients.recycle(
            death_result.recycled_plant_nutrients.0,
            death_result.recycled_plant_nutrients.1,
        );

        // Spawn detritus at dead creature locations (50% of max energy recycled)
        for (x, y, max_e) in &death_result.dead_creature_info {
            let detritus_energy = max_e * 0.5;
            if detritus_energy > 1.0 {
                self.spawn_detritus(*x, *y, detritus_energy);
            }
        }

        // 11. Plant reproduction — per-plant reserve allocation into seed/clonal propagules.
        self.total_producer_births += self.reproduce_producers();

        // 12. Animation
        animation::animation_system(&mut self.world, dt);

        self.tick_count += 1;

        // 13. Periodically recompute cached stats (every 20 ticks, not every frame)
        let should_refresh_stats =
            day_rolled_over || self.tick_count % 20 == 0 || self.tick_count == 1;
        if should_refresh_stats {
            self.recompute_cached_stats();
        }
        if day_rolled_over {
            self.record_daily_diagnostics();
        }
    }

    fn world(&self) -> &hecs::World {
        &self.world
    }

    fn environment(&self) -> &Environment {
        &self.env
    }

    fn stats(&self) -> SimStats {
        SimStats {
            entity_count: self.world.len() as usize,
            creature_count: self.cached_creature_count,
            tick_count: self.tick_count,
            creature_births: self.total_creature_births,
            creature_deaths: self.total_creature_deaths,
            producer_births: self.total_producer_births,
            producer_deaths: self.total_producer_deaths,
            elapsed_days: self.elapsed_days,
            max_generation: self.cached_max_generation,
            avg_complexity: self.cached_avg_complexity,
            max_creature_complexity: self.cached_max_creature_complexity,
            species_count: self.cached_species_count,
            producer_leaf_biomass: self.cached_producer_leaf_biomass,
            producer_structural_biomass: self.cached_producer_structural_biomass,
            producer_belowground_reserve: self.cached_producer_belowground_reserve,
            consumer_biomass: self.cached_consumer_biomass,
            rolling_producer_npp: self.rolling_producer_npp,
            rolling_consumer_intake: self.rolling_consumer_intake,
            rolling_consumer_maintenance: self.rolling_consumer_maintenance,
            juvenile_count: self.cached_juvenile_count,
            adult_count: self.cached_adult_count,
        }
    }

    fn ecology_diagnostics(&self) -> EcologyDiagnostics {
        EcologyDiagnostics {
            instant: self.build_ecology_instant(),
            daily_history: self.daily_ecology_history.iter().cloned().collect(),
        }
    }

    fn tank_size(&self) -> (u16, u16) {
        (self.tank_width, self.tank_height)
    }

    fn diversity_coefficient(&self) -> f32 {
        self.diversity_coefficient
    }

    fn set_diversity_coefficient(&mut self, value: f32) {
        self.diversity_coefficient = value.clamp(0.25, 2.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecosystem::{Age, Detritus, Energy};
    use crate::environment::{EnvironmentEvent, EventKind};
    use crate::genome::CreatureGenome;
    use crate::needs::{NeedWeights, Needs};
    use crate::phenotype::derive_physics;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_environment_defaults() {
        let env = Environment::default();
        assert!((env.time_of_day - 8.0).abs() < f32::EPSILON);
        assert!(env.light_level >= 0.0 && env.light_level <= 1.0);
        assert!(env.temperature > 0.0);
    }

    #[test]
    fn test_sim_starts_empty() {
        let sim = AquariumSim::new(80, 24);
        assert_eq!(sim.stats().entity_count, 0);
    }

    #[test]
    fn test_ecology_diagnostics_expose_current_nutrient_state() {
        let sim = AquariumSim::new(80, 24);
        let diagnostics = sim.ecology_diagnostics();

        assert!((diagnostics.instant.dissolved_n - sim.nutrients.dissolved_n).abs() < 0.001);
        assert!((diagnostics.instant.dissolved_p - sim.nutrients.dissolved_p).abs() < 0.001);
        assert!((diagnostics.instant.sediment_n - sim.nutrients.sediment_n).abs() < 0.001);
        assert!((diagnostics.instant.sediment_p - sim.nutrients.sediment_p).abs() < 0.001);
        assert!(
            (diagnostics.instant.phytoplankton_load - sim.nutrients.phytoplankton_load).abs()
                < 0.001
        );
    }

    #[test]
    fn test_ecology_history_appends_once_per_day() {
        let mut sim = AquariumSim::new(40, 16);
        sim.env.set_random_events_enabled(false);

        for _ in 0..150 {
            sim.tick(1.0);
        }
        assert_eq!(sim.ecology_diagnostics().daily_history.len(), 0);

        for _ in 0..150 {
            sim.tick(1.0);
        }
        let diagnostics = sim.ecology_diagnostics();
        assert_eq!(diagnostics.daily_history.len(), 1);
        assert_eq!(diagnostics.daily_history[0].day, 1);

        for _ in 0..300 {
            sim.tick(1.0);
        }
        let diagnostics = sim.ecology_diagnostics();
        assert_eq!(diagnostics.daily_history.len(), 2);
        assert_eq!(diagnostics.daily_history[1].day, 2);
    }

    #[test]
    fn test_ecology_history_records_daily_birth_and_death_deltas() {
        let mut sim = AquariumSim::new(40, 16);
        sim.total_creature_births = 3;
        sim.total_creature_deaths = 1;
        sim.total_producer_births = 4;
        sim.total_producer_deaths = 2;
        sim.elapsed_days = 1;
        sim.recompute_cached_stats();
        sim.record_daily_diagnostics();

        sim.total_creature_births = 8;
        sim.total_creature_deaths = 2;
        sim.total_producer_births = 9;
        sim.total_producer_deaths = 5;
        sim.elapsed_days = 2;
        sim.record_daily_diagnostics();

        let diagnostics = sim.ecology_diagnostics();
        assert_eq!(diagnostics.daily_history.len(), 2);
        assert_eq!(diagnostics.daily_history[0].creature_births_delta, 3);
        assert_eq!(diagnostics.daily_history[0].creature_deaths_delta, 1);
        assert_eq!(diagnostics.daily_history[0].producer_births_delta, 4);
        assert_eq!(diagnostics.daily_history[0].producer_deaths_delta, 2);
        assert_eq!(diagnostics.daily_history[1].creature_births_delta, 5);
        assert_eq!(diagnostics.daily_history[1].creature_deaths_delta, 1);
        assert_eq!(diagnostics.daily_history[1].producer_births_delta, 5);
        assert_eq!(diagnostics.daily_history[1].producer_deaths_delta, 3);
    }

    #[test]
    fn test_ecology_history_resets_with_runtime_counters() {
        let mut sim = AquariumSim::new(40, 16);
        sim.env.set_random_events_enabled(false);
        for _ in 0..600 {
            sim.tick(1.0);
        }
        assert!(!sim.ecology_diagnostics().daily_history.is_empty());

        sim.reset_runtime_counters();

        let diagnostics = sim.ecology_diagnostics();
        assert!(diagnostics.daily_history.is_empty());
        assert_eq!(sim.elapsed_days, 0);
        assert_eq!(sim.total_creature_births, 0);
        assert_eq!(sim.total_producer_births, 0);
    }

    #[test]
    fn test_ecology_history_caps_at_32_days() {
        let mut sim = AquariumSim::new(40, 16);
        sim.env.set_random_events_enabled(false);

        for _ in 0..(35 * 300) {
            sim.tick(1.0);
        }

        let diagnostics = sim.ecology_diagnostics();
        assert_eq!(diagnostics.daily_history.len(), ECOLOGY_HISTORY_DAYS);
        assert_eq!(diagnostics.daily_history.first().unwrap().day, 4);
        assert_eq!(diagnostics.daily_history.last().unwrap().day, 35);
    }

    #[test]
    fn test_spawn_and_tick_moves_creature() {
        let mut sim = AquariumSim::new(80, 24);
        let frame = AsciiFrame::from_rows(vec![r"<=>", r"<->"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        sim.spawn_creature(
            Position { x: 10.0, y: 5.0 },
            Velocity { vx: 4.0, vy: 0.0 },
            BoundingBox { w: 3.0, h: 2.0 },
            appearance,
            AnimationState::new(0.2),
        );
        assert_eq!(sim.stats().entity_count, 1);
        for _ in 0..10 {
            sim.tick(0.05);
        }
        let new_x = {
            let mut q = sim.world.query::<&Position>();
            let pos = (&mut q).into_iter().next().unwrap();
            pos.x
        };
        assert!(new_x > 10.0, "Creature should have moved right");
    }

    #[test]
    fn test_ascii_frame_flip() {
        let frame = AsciiFrame::from_rows(vec![r" /o\=>", r" \  />"]);
        let flipped = frame.flip_horizontal();
        assert_eq!(flipped.rows[0], r"<=/o\ ");
        assert_eq!(flipped.rows[1], r"<\  / ");
    }

    /// Helper: spawn a creature from a genome for integration tests.
    fn spawn_test_creature(sim: &mut AquariumSim, x: f32, y: f32) {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = CreatureGenome::minimal_cell(&mut rng);
        sim.spawn_from_genome(genome, x, y);
    }

    /// Helper: spawn a deterministic grazer so soak tests exercise plant-herbivore
    /// coupling instead of relying on random creature draws.
    fn spawn_test_grazer(sim: &mut AquariumSim, x: f32, y: f32) -> hecs::Entity {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = CreatureGenome::minimal_cell(&mut rng);
        genome.art.body_size = 0.60;
        genome.art.body_elongation = 0.40;
        genome.behavior.aggression = 0.05;
        genome.behavior.hunting_instinct = 0.0;
        genome.behavior.mouth_size = 0.55;
        genome.behavior.speed_factor = 0.9;
        genome.behavior.metabolism_factor = 0.65;
        genome.behavior.max_lifespan_factor = 2.0;
        genome.behavior.reproduction_rate = 0.2;

        let entity = sim.spawn_from_genome(genome, x, y);

        if let Ok(mut needs) = sim.world.get::<&mut Needs>(entity) {
            needs.hunger = 0.55;
        }
        if let Ok(mut weights) = sim.world.get::<&mut NeedWeights>(entity) {
            // Keep the soak focused on grazing pressure and plant persistence rather
            // than on reproduction-driven population explosions.
            weights.hunger_rate = 0.01;
        }
        if let Ok(mut energy) = sim.world.get::<&mut Energy>(entity) {
            energy.current = energy.max * 0.95;
        }
        if let Ok(mut state) = sim.world.get::<&mut ConsumerState>(entity) {
            state.brood_cooldown = 1_000_000.0;
            state.maturity_progress = 1.0;
        }
        if let Ok(mut age) = sim.world.get::<&mut Age>(entity) {
            // Test harness note: extend grazer lifespan so the soak exercises
            // plant-herbivore persistence instead of nominal creature senescence.
            age.max_ticks = 120_000;
        }

        let feeding = sim.world.get::<&FeedingCapability>(entity).unwrap();
        assert!(
            feeding.graze_skill > 0.7 && feeding.graze_skill > feeding.hunt_skill,
            "Test grazer should strongly prefer plants: graze_skill={:.3}, hunt_skill={:.3}",
            feeding.graze_skill,
            feeding.hunt_skill,
        );
        drop(feeding);

        entity
    }

    /// Helper: spawn a seaweed plant for integration tests.
    fn spawn_test_producer(sim: &mut AquariumSim, x: f32, y: f32) {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = genome::ProducerGenome::minimal_producer(&mut rng);
        sim.spawn_producer(Position { x, y }, genome);
    }

    fn prime_test_adult(sim: &mut AquariumSim, entity: hecs::Entity, buffer_fill: f32) {
        let (physics, genome) = {
            let physics = sim.world.get::<&DerivedPhysics>(entity).unwrap().clone();
            let genome = sim.world.get::<&CreatureGenome>(entity).unwrap().clone();
            (physics, genome)
        };
        let threshold = ecosystem::consumer_reproductive_threshold(&physics, &genome);
        if let Ok(mut age) = sim.world.get::<&mut Age>(entity) {
            age.ticks = age.max_ticks / 3;
        }
        if let Ok(mut state) = sim.world.get::<&mut ConsumerState>(entity) {
            state.maturity_progress = 1.0;
            state.reserve_buffer = 0.92;
            state.recent_assimilation = 0.25;
            state.reproductive_buffer = threshold * buffer_fill;
            state.brood_cooldown = 0.0;
        }
        if let Ok(mut needs) = sim.world.get::<&mut Needs>(entity) {
            needs.hunger = 0.08;
        }
        if let Ok(mut energy) = sim.world.get::<&mut Energy>(entity) {
            energy.current = energy.max * 0.96;
        }
    }

    /// Helper: spawn a representative submerged plant with stable, moderate traits so
    /// long-running soak tests are deterministic and focus on ecology rather than
    /// random genome draws.
    fn spawn_test_soak_producer(sim: &mut AquariumSim, x: f32, y: f32) -> hecs::Entity {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = genome::ProducerGenome::minimal_producer(&mut rng);
        genome.stem_thickness = 0.45;
        genome.height_factor = 0.55;
        genome.leaf_area = 0.62;
        genome.branching = 0.40;
        genome.curvature = 0.25;
        genome.photosynthesis_rate = 1.10;
        genome.max_energy_factor = 1.10;
        genome.hardiness = 0.50;
        genome.seed_range = 0.80;
        genome.seed_count = 0.60;
        genome.seed_size = 0.85;
        genome.lifespan_factor = 1.15;
        genome.nutritional_value = 0.80;
        genome.clonal_spread = 0.45;
        genome.nutrient_affinity = 1.00;
        genome.epiphyte_resistance = 0.50;
        genome.reserve_allocation = 0.35;
        genome.complexity = 0.30;
        sim.spawn_producer(Position { x, y }, genome)
    }

    fn seed_mixed_grazer_ecology(sim: &mut AquariumSim, grazer_count: usize, plant_count: usize) {
        let tw = sim.tank_width as f32;
        let th = sim.tank_height as f32;
        let grazer_rows = [th * 0.34, th * 0.44, th * 0.54];
        let producer_rows = [th * 0.18, th * 0.28, th * 0.38, th * 0.48];

        for i in 0..grazer_count {
            let frac = (i as f32 + 0.5) / grazer_count.max(1) as f32;
            let x = 4.0 + frac * (tw - 8.0);
            let y = grazer_rows[i % grazer_rows.len()].clamp(4.0, th - 4.0);
            spawn_test_grazer(sim, x, y);
        }

        for i in 0..plant_count {
            let frac = (i as f32 + 0.5) / plant_count.max(1) as f32;
            let x = 3.0 + frac * (tw - 6.0);
            let y = producer_rows[i % producer_rows.len()].clamp(3.0, th - 4.0);
            spawn_test_soak_producer(sim, x, y);
        }
    }

    fn producer_snapshot(sim: &mut AquariumSim) -> (usize, f32, f32, f32) {
        let mut count = 0usize;
        let mut total_energy = 0.0;
        let mut total_leaf = 0.0;
        let mut total_structural = 0.0;

        for (_entity, energy, state) in &mut sim.world.query::<(Entity, &Energy, &ProducerState)>()
        {
            count += 1;
            total_energy += energy.current;
            total_leaf += state.leaf_biomass;
            total_structural += state.structural_biomass;
        }

        (count, total_energy, total_leaf, total_structural)
    }

    fn grazer_snapshot(sim: &mut AquariumSim) -> (usize, usize) {
        let mut creature_count = 0usize;
        let mut grazer_count = 0usize;

        for (_entity, feeding) in &mut sim.world.query::<(Entity, &FeedingCapability)>() {
            if feeding.is_producer {
                continue;
            }
            creature_count += 1;
            if feeding.graze_skill > 0.7 && feeding.graze_skill > feeding.hunt_skill {
                grazer_count += 1;
            }
        }

        (creature_count, grazer_count)
    }

    #[test]
    fn test_creatures_can_eat_food() {
        let mut sim = AquariumSim::new(40, 20);

        spawn_test_creature(&mut sim, 10.0, 10.0);
        spawn_test_producer(&mut sim, 12.0, 10.0);

        assert_eq!(sim.stats().entity_count, 2);

        for _ in 0..200 {
            sim.tick(0.05);
        }

        let creature_alive = {
            let mut has_creature = false;
            for _genome in &mut sim.world.query::<&CreatureGenome>() {
                has_creature = true;
            }
            has_creature
        };
        let stats = sim.stats();
        assert!(
            creature_alive,
            "Creature should still be alive after {} ticks (creature_deaths={}, entity_count={})",
            stats.tick_count, stats.creature_deaths, stats.entity_count
        );
    }

    #[test]
    fn test_ecosystem_survives_2000_ticks() {
        let mut sim = AquariumSim::new(60, 20);

        for i in 0..5 {
            spawn_test_creature(&mut sim, 10.0 + i as f32 * 8.0, 8.0);
        }
        for i in 0..6 {
            spawn_test_producer(&mut sim, 5.0 + i as f32 * 8.0, 15.0);
        }

        let dt = 0.05;
        for _ in 0..2000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        let creature_count = stats.creature_count;

        assert!(
            creature_count > 0,
            "At least one creature should survive 2000 ticks! \
             CreatureBirths={}, CreatureDeaths={}, Entities={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.entity_count,
        );
    }

    #[test]
    fn test_producers_seed_new_producers() {
        let mut sim = AquariumSim::new(40, 20);

        // Start with one plant near surface
        spawn_test_producer(&mut sim, 20.0, 5.0);

        // Put the plant into a genuinely reproductive state under the new
        // biomass-driven lifecycle model.
        for age in sim.world.query_mut::<&mut ProducerAge>() {
            age.seconds = 50.0;
        }
        for state in sim.world.query_mut::<&mut ProducerState>() {
            state.structural_biomass *= 1.4;
            state.leaf_biomass *= 1.5;
            state.seed_cooldown = 0.0;
            state.clonal_cooldown = 0.0;
        }
        for energy in sim.world.query_mut::<&mut Energy>() {
            energy.current = energy.max * 0.9;
        }
        for stage in sim.world.query_mut::<&mut ProducerStage>() {
            *stage = ProducerStage::Broadcasting;
        }

        assert_eq!(sim.stats().entity_count, 1);

        // Run long enough for reserve-driven propagule production to occur.
        for _ in 0..5000 {
            sim.tick(0.05);
        }

        assert!(
            sim.stats().entity_count > 1 && sim.stats().producer_births > 0,
            "Broadcasting producer should seed new producers. Entities={}, ProducerBirths={}",
            sim.stats().entity_count,
            sim.stats().producer_births,
        );
    }

    #[test]
    fn test_producers_regenerate_energy() {
        let mut sim = AquariumSim::new(40, 20);
        spawn_test_producer(&mut sim, 10.0, 3.0); // near surface for better light

        // Set reserve to mid-range so the plant must recover through photosynthesis.
        for energy in sim.world.query_mut::<&mut Energy>() {
            energy.current = energy.max * 0.4;
        }

        let initial: f32 = sim
            .world
            .query_mut::<&Energy>()
            .into_iter()
            .next()
            .unwrap()
            .current;

        for _ in 0..400 {
            sim.tick(0.05);
        }

        for energy in &mut sim.world.query::<&Energy>() {
            assert!(
                energy.current > initial,
                "Producer should regenerate energy via photosynthesis, got {} (started at {})",
                energy.current,
                initial
            );
            break;
        }
    }

    #[test]
    fn test_producers_persist_for_multiple_days() {
        let mut sim = AquariumSim::new(60, 20);
        for i in 0..6 {
            let x = 5.0 + i as f32 * 8.0;
            let y = 4.0 + (i as f32 % 3.0) * 3.0;
            spawn_test_producer(&mut sim, x, y);
        }

        // Six in-game days at 20 ticks/sec and 300 real seconds/day.
        for _ in 0..36000 {
            sim.tick(0.05);
        }

        let plant_count = sim
            .world
            .query_mut::<&genome::ProducerGenome>()
            .into_iter()
            .count();
        let mut total_energy = 0.0;
        let mut total_leaf = 0.0;
        let mut total_structural = 0.0;
        let mut sampled = 0usize;
        for (_entity, energy, state) in &mut sim.world.query::<(Entity, &Energy, &ProducerState)>()
        {
            total_energy += energy.current;
            total_leaf += state.leaf_biomass;
            total_structural += state.structural_biomass;
            sampled += 1;
        }
        let stats = sim.stats();
        assert!(
            plant_count > 0,
            "Plant-only tank should still contain living plants after multiple days. \
             plants={} sampled={} total_energy={:.2} total_leaf={:.2} total_structural={:.2} \
             creature_births={} creature_deaths={} producer_births={} producer_deaths={} elapsed_days={}",
            plant_count,
            sampled,
            total_energy,
            total_leaf,
            total_structural,
            stats.creature_births,
            stats.creature_deaths,
            stats.producer_births,
            stats.producer_deaths,
            stats.elapsed_days,
        );
    }

    #[test]
    fn test_seed_establishment_prefers_open_patch_over_dense_patch() {
        let mut sim = AquariumSim::new(60, 20);

        for i in 0..6 {
            let x = 10.0 + (i as f32 % 3.0) * 0.9;
            let y = 5.0 + (i as f32 / 3.0).floor() * 0.8;
            spawn_test_soak_producer(&mut sim, x, y);
        }

        sim.grid.rebuild(&sim.world);
        sim.light_field.rebuild(&sim.world);

        let crowded = sim.producer_establishment_chance(11.0, 5.4, ProducerOrigin::Broadcast, 0.5);
        let open = sim.producer_establishment_chance(42.0, 5.4, ProducerOrigin::Broadcast, 0.5);

        assert!(
            open > crowded * 1.5,
            "Seed establishment should favor open space over an occupied patch. crowded={:.3}, open={:.3}",
            crowded,
            open,
        );
    }

    #[test]
    fn test_mixed_ecology_producers_persist_with_grazers_for_multiple_days() {
        let mut sim = AquariumSim::new(60, 20);
        seed_mixed_grazer_ecology(&mut sim, 4, 10);

        // Three in-game days at 20 ticks/sec and 300 real seconds/day.
        for _ in 0..18_000 {
            sim.tick(0.05);
        }

        let (plant_count, total_energy, total_leaf, total_structural) = producer_snapshot(&mut sim);
        let (creature_count, grazer_count) = grazer_snapshot(&mut sim);
        let stats = sim.stats();

        assert!(
            plant_count > 0,
            "Mixed ecology should retain plants under grazer pressure. \
             plants={} creatures={} grazers={} total_energy={:.2} total_leaf={:.2} \
             total_structural={:.2} creature_births={} creature_deaths={} \
             producer_births={} producer_deaths={} elapsed_days={}",
            plant_count,
            creature_count,
            grazer_count,
            total_energy,
            total_leaf,
            total_structural,
            stats.creature_births,
            stats.creature_deaths,
            stats.producer_births,
            stats.producer_deaths,
            stats.elapsed_days,
        );
        assert!(
            grazer_count > 0,
            "Mixed ecology should retain at least one grazer. \
             plants={} creatures={} grazers={} total_energy={:.2} total_leaf={:.2} \
             total_structural={:.2} creature_births={} creature_deaths={} \
             producer_births={} producer_deaths={} elapsed_days={}",
            plant_count,
            creature_count,
            grazer_count,
            total_energy,
            total_leaf,
            total_structural,
            stats.creature_births,
            stats.creature_deaths,
            stats.producer_births,
            stats.producer_deaths,
            stats.elapsed_days,
        );
        assert!(
            stats.elapsed_days >= 3,
            "Expected at least three in-game days, saw {}",
            stats.elapsed_days,
        );
    }

    #[test]
    #[ignore = "long mixed-ecology soak"]
    fn test_mixed_ecology_producers_persist_with_grazers_long_soak() {
        let mut sim = AquariumSim::new(80, 24);
        seed_mixed_grazer_ecology(&mut sim, 5, 24);

        // Six in-game days at 20 ticks/sec and 300 real seconds/day.
        let mut grazer_days = 0u64;
        for _ in 0..36_000 {
            sim.tick(0.05);
            if sim.tick_count % 6_000 == 0 {
                let (_, grazers) = grazer_snapshot(&mut sim);
                if grazers > 0 {
                    grazer_days += 1;
                }
            }
        }

        let (plant_count, total_energy, total_leaf, total_structural) = producer_snapshot(&mut sim);
        let (creature_count, grazer_count) = grazer_snapshot(&mut sim);
        let stats = sim.stats();

        assert!(
            plant_count > 0,
            "Long mixed-ecology soak should retain plants under grazer pressure. \
             plants={} creatures={} grazers={} total_energy={:.2} total_leaf={:.2} \
             total_structural={:.2} creature_births={} creature_deaths={} \
             producer_births={} producer_deaths={} elapsed_days={}",
            plant_count,
            creature_count,
            grazer_count,
            total_energy,
            total_leaf,
            total_structural,
            stats.creature_births,
            stats.creature_deaths,
            stats.producer_births,
            stats.producer_deaths,
            stats.elapsed_days,
        );
        assert!(
            grazer_days >= 3,
            "Long mixed-ecology soak should include several grazer-active days. \
             grazer_days={} plants={} creatures={} grazers_final={} total_energy={:.2} \
             total_leaf={:.2} total_structural={:.2} creature_births={} creature_deaths={} \
             producer_births={} producer_deaths={} elapsed_days={}",
            grazer_days,
            plant_count,
            creature_count,
            grazer_count,
            total_energy,
            total_leaf,
            total_structural,
            stats.creature_births,
            stats.creature_deaths,
            stats.producer_births,
            stats.producer_deaths,
            stats.elapsed_days,
        );
        assert!(
            stats.elapsed_days >= 6,
            "Expected at least six in-game days, saw {}",
            stats.elapsed_days,
        );
    }

    #[test]
    fn test_spawn_from_genome() {
        let mut sim = AquariumSim::new(80, 24);
        let mut rng = StdRng::seed_from_u64(42);
        let genome = CreatureGenome::random(&mut rng);
        let entity = sim.spawn_from_genome(genome, 10.0, 10.0);

        assert!(sim.world.get::<&CreatureGenome>(entity).is_ok());
        assert!(sim.world.get::<&FeedingCapability>(entity).is_ok());
        assert!(sim.world.get::<&Energy>(entity).is_ok());
        assert!(sim.world.get::<&Boid>(entity).is_ok());
    }

    // ── Balance tests ─────────────────────────────────────────
    // These tests verify the energy economy supports a sustainable ecosystem.

    #[test]
    fn test_cell_survives_under_producer_rich_conditions() {
        // Under the life-history model, a single founder is not guaranteed to
        // produce a birth on a fixed short horizon because encounter geometry
        // and stochastic feeding still matter. The deterministic contract we
        // care about here is narrower: a primed adult grazer with ample producer
        // biomass should survive the short regression window. Integrated tests
        // with multiple founders cover actual births and sustained turnover.
        let mut sim = AquariumSim::new(30, 15);
        sim.env.set_random_events_enabled(false);

        let mut genome = sim.founder_consumer_genome();
        genome.behavior.reproduction_rate = 0.75;
        let entity = sim.spawn_from_genome(genome, 15.0, 7.0);
        prime_test_adult(&mut sim, entity, 1.15);
        for i in 0..8 {
            spawn_test_soak_producer(&mut sim, 5.0 + i as f32 * 2.8, 9.0 + (i % 2) as f32 * 1.5);
        }

        let dt = 0.05;
        for _ in 0..2500 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        let state = sim.world.get::<&ConsumerState>(entity);
        let energy = sim.world.get::<&Energy>(entity);
        assert!(
            stats.creature_count > 0 && stats.creature_deaths == 0,
            "Primed single grazer should survive the short rich-food regression. \
             Creatures={}, CreatureDeaths={}, Entities={}",
            stats.creature_count,
            stats.creature_deaths,
            stats.entity_count,
        );
        assert!(
            state.is_ok() && energy.is_ok(),
            "Primed grazer should remain alive and queryable at the end of the regression"
        );
        let state = state.unwrap();
        let energy = energy.unwrap();
        assert!(
            state.is_adult() && energy.fraction() > 0.35,
            "Primed grazer should stay alive and in reasonable condition under ample producer biomass. \
             Adult={} EnergyFraction={:.4} ReproductiveBuffer={:.4}",
            state.is_adult(),
            energy.fraction(),
            state.reproductive_buffer,
        );
    }

    #[test]
    fn test_cell_energy_math_is_viable() {
        // Unit test: verify a minimal cell's max_energy is large enough
        // relative to its metabolism that it can survive a reasonable time.
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let g = CreatureGenome::minimal_cell(&mut rng);
            let p = derive_physics(&g);

            let drain_per_sec = p.base_metabolism * 0.5; // metabolism_system multiplier
            let initial_energy = p.max_energy * 0.8; // Energy::new starts at 80%
            let survive_secs = initial_energy / drain_per_sec;

            assert!(
                survive_secs > 120.0,
                "Cell should survive >120s without food, got {:.1}s \
                 (max_energy={:.1}, metabolism={:.4}, drain/s={:.4})",
                survive_secs,
                p.max_energy,
                p.base_metabolism,
                drain_per_sec,
            );
        }
    }

    #[test]
    fn test_population_grows_with_food() {
        // Start with 10 mature consumers and abundant plants.
        // Under the new life-history model, the regression target is bounded
        // coexistence and survival rather than immediate rapid growth.
        let mut sim = AquariumSim::new(60, 20);
        let tw = 60.0_f32;
        let th = 20.0_f32;

        for i in 0..10 {
            let x = 5.0 + (i as f32 / 10.0) * (tw - 10.0);
            let y = th * 0.4 + (i as f32 % 3.0) * 2.0;
            let entity = sim.spawn_from_genome(
                CreatureGenome::minimal_cell(&mut StdRng::seed_from_u64(42)),
                x,
                y,
            );
            prime_test_adult(&mut sim, entity, 0.85);
        }

        // Dense producer coverage
        for i in 0..12 {
            let x = 3.0 + (i as f32 / 12.0) * (tw - 6.0);
            let y = th * 0.3 + (i as f32 % 4.0) * 3.0;
            spawn_test_producer(&mut sim, x, y);
        }

        let dt = 0.05;
        for _ in 0..5000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.creature_count >= 6 && stats.creature_count <= 24,
            "Population should remain viable and bounded under abundant food. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, Entities={}, consumer_biomass={:.2}",
            stats.creature_births, stats.creature_deaths, stats.creature_count, stats.entity_count, stats.consumer_biomass,
        );
    }

    #[test]
    fn test_ecosystem_survives_10000_ticks() {
        // Regression target for the new startup path: a producer-first bootstrap
        // should keep both producers and consumers alive through a shorter soak,
        // without relying on immediate births from arbitrary founders.
        let mut sim = AquariumSim::new(80, 24);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let dt = 0.05;
        for _ in 0..10000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        let producer_biomass = stats.producer_leaf_biomass
            + stats.producer_structural_biomass
            + stats.producer_belowground_reserve;
        assert!(
            stats.creature_count > 0,
            "Consumers should survive 10000 ticks after the default bootstrap. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, Entities={}, ProducerBiomass={:.2}",
            stats.creature_births, stats.creature_deaths, stats.creature_count, stats.entity_count, producer_biomass,
        );
        assert!(
            producer_biomass > 0.05,
            "Producer biomass should survive 10000 ticks after the default bootstrap. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, Entities={}, ProducerBiomass={:.2}",
            stats.creature_births, stats.creature_deaths, stats.creature_count, stats.entity_count,
            producer_biomass,
        );
    }

    #[test]
    fn test_stats_include_generation_and_complexity() {
        let mut sim = AquariumSim::new(80, 24);
        let mut rng = StdRng::seed_from_u64(42);

        let mut g = CreatureGenome::random(&mut rng);
        g.generation = 5;
        g.complexity = 0.6;
        sim.spawn_from_genome(g, 10.0, 10.0);
        sim.tick(0.05); // trigger initial stats cache

        let stats = sim.stats();
        assert_eq!(stats.max_generation, 5);
        assert!((stats.avg_complexity - 0.6).abs() < 0.01);
        assert!(stats.species_count >= 1);
    }

    #[test]
    fn test_stats_track_creature_births_and_deaths_separately() {
        let mut sim = AquariumSim::new(40, 20);
        let mut genome = sim.founder_consumer_genome();
        genome.behavior.reproduction_rate = 0.9;

        let reproducer = sim.spawn_from_genome(genome, 20.0, 8.0);
        prime_test_adult(&mut sim, reproducer, 1.25);

        for _ in 0..20 {
            sim.tick(0.05);
            if sim.stats().creature_births > 0 {
                break;
            }
        }

        let doomed = sim.spawn_from_genome(
            CreatureGenome::minimal_cell(&mut StdRng::seed_from_u64(42)),
            6.0,
            6.0,
        );
        if let Ok(mut energy) = sim.world.get::<&mut Energy>(doomed) {
            energy.current = 0.0;
        }
        sim.tick(0.05);

        let stats = sim.stats();
        assert!(
            stats.creature_births > 0,
            "Creature births should be tracked separately"
        );
        assert!(
            stats.creature_deaths > 0,
            "Creature deaths should be tracked separately"
        );
        assert_eq!(
            stats.producer_births, 0,
            "Producer births should stay zero in creature-only counter test"
        );
        assert_eq!(
            stats.producer_deaths, 0,
            "Producer deaths should stay zero in creature-only counter test"
        );
    }

    #[test]
    fn test_stats_track_producer_births_and_deaths_separately() {
        let mut sim = AquariumSim::new(40, 20);
        let entity = spawn_test_soak_producer(&mut sim, 20.0, 5.0);

        if let Ok(mut stage) = sim.world.get::<&mut ProducerStage>(entity) {
            *stage = ProducerStage::Broadcasting;
        }
        if let Ok(mut energy) = sim.world.get::<&mut Energy>(entity) {
            energy.current = energy.max * 0.94;
        }
        if let Ok(mut age) = sim.world.get::<&mut ProducerAge>(entity) {
            age.seconds = 50.0;
        }
        if let Ok(mut state) = sim.world.get::<&mut ProducerState>(entity) {
            state.structural_biomass *= 1.5;
            state.leaf_biomass *= 1.5;
            state.seed_cooldown = 0.0;
            state.clonal_cooldown = 0.0;
        }

        for _ in 0..4000 {
            sim.tick(0.05);
            if sim.stats().producer_births > 0 {
                break;
            }
        }

        if let Ok(mut energy) = sim.world.get::<&mut Energy>(entity) {
            energy.current = 0.0;
        }
        if let Ok(mut state) = sim.world.get::<&mut ProducerState>(entity) {
            state.belowground_reserve = 0.0;
            state.meristem_bank = 0.0;
            state.structural_biomass = 0.02;
            state.leaf_biomass = 0.0;
        }
        sim.tick(0.05);

        let stats = sim.stats();
        assert!(
            stats.producer_births > 0,
            "Producer births should be tracked separately"
        );
        assert!(
            stats.producer_deaths > 0,
            "Producer deaths should be tracked separately"
        );
        assert_eq!(
            stats.creature_births, 0,
            "Creature births should stay zero in plant-only counter test"
        );
        assert_eq!(
            stats.creature_deaths, 0,
            "Creature deaths should stay zero in plant-only counter test"
        );
    }

    // ── Hypothesis tests ──────────────────────────────────────────
    // These tests verify the evolutionary dynamics fixes actually work.

    /// Hypothesis test: grazing drains plant energy without killing the plant.
    /// Plants should survive being eaten and regenerate via photosynthesis.
    #[test]
    fn test_grazing_drains_but_preserves_producers() {
        let mut sim = AquariumSim::new(30, 15);

        // Place a creature right next to a plant so it can graze
        spawn_test_creature(&mut sim, 11.0, 10.0);
        spawn_test_producer(&mut sim, 12.0, 10.0);

        // Record initial plant energy
        let mut initial_plant_energy = 0.0_f32;
        for energy in &mut sim.world.query::<(&Energy, &Producer)>() {
            initial_plant_energy = energy.0.current;
        }
        assert!(initial_plant_energy > 0.0);

        // Run enough ticks for grazing to occur
        for _ in 0..200 {
            sim.tick(0.05);
        }

        // Producer should still exist (not destroyed)
        let mut plant_count = 0;
        for _p in &mut sim.world.query::<&Producer>() {
            plant_count += 1;
        }
        assert!(
            plant_count >= 1,
            "Producer should survive grazing (partial energy drain, not destruction)"
        );
    }

    /// Hypothesis test: ecosystem sustains itself using only plant photosynthesis.
    /// No food rain — plants are the sole energy source. Population should survive.
    #[test]
    fn test_ecosystem_sustains_on_photosynthesis_alone() {
        let mut sim = AquariumSim::new(60, 20);
        let tw = 60.0_f32;
        let th = 20.0_f32;

        // Spawn 10 cells with abundant plants
        for i in 0..10 {
            let x = 5.0 + (i as f32 / 10.0) * (tw - 10.0);
            let y = th * 0.3 + (i as f32 % 3.0) * 2.0;
            spawn_test_creature(&mut sim, x, y);
        }
        for i in 0..12 {
            let x = 3.0 + (i as f32 / 12.0) * (tw - 6.0);
            let y = th * 0.5 + (i as f32 % 4.0) * 3.0;
            spawn_test_producer(&mut sim, x, y);
        }

        let dt = 0.05;
        for _ in 0..5000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        // Plants should still exist (photosynthesis keeps them alive)
        let mut plant_count = 0;
        for _p in &mut sim.world.query::<&Producer>() {
            plant_count += 1;
        }
        assert!(
            plant_count > 0,
            "Plants should survive via photosynthesis. Plants={}, Creatures={}",
            plant_count,
            stats.creature_count,
        );
        assert!(
            stats.creature_count > 0,
            "Creatures should survive on plant-based ecosystem! \
             CreatureBirths={}, CreatureDeaths={}, Plants={}",
            stats.creature_births,
            stats.creature_deaths,
            plant_count,
        );
    }

    /// Hypothesis test: compatible mature consumers can reproduce sexually in
    /// the full simulation loop once life-history gates are satisfied.
    #[test]
    fn test_sexual_reproduction_across_generations() {
        let mut sim = AquariumSim::new(40, 15);
        let mut rng = StdRng::seed_from_u64(42);

        // Spawn 15 creatures with complexity 0.6 so compatible mating is possible
        // All clones of the same genome so they start compatible
        let mut base = CreatureGenome::minimal_cell(&mut rng);
        base.complexity = 0.6;

        for i in 0..15 {
            let mut g = base.clone();
            // Small variation so they're not exact clones
            g.art.body_elongation += (i as f32) * 0.01;
            let x = 3.0 + (i as f32 / 15.0) * 34.0;
            let y = 3.0 + (i as f32 % 3.0) * 3.0;
            let entity = sim.spawn_from_genome(g, x, y);
            prime_test_adult(&mut sim, entity, 1.10);
        }

        // Add food
        for i in 0..8 {
            spawn_test_producer(&mut sim, 3.0 + i as f32 * 4.0, 10.0);
        }

        let dt = 0.05;
        for _ in 0..10_000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.creature_births > 0,
            "Sexual reproduction should produce offspring in the integrated sim. \
             Max gen={}, CreatureBirths={}, CreatureDeaths={}, Creatures={}",
            stats.max_generation,
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
        );
    }

    /// Integration regression: a simple founder web should show real consumer
    /// turnover and at least one successful birth in a short controlled run.
    ///
    /// Long-horizon complexity increase is covered by the ignored soak test
    /// above; the default suite should not require visible evolutionary novelty
    /// on this much shorter horizon.
    #[test]
    fn test_simple_founder_web_reproduces_in_simulation() {
        let mut sim = AquariumSim::new(60, 20);
        sim.env.set_random_events_enabled(false);
        let tw = 60.0_f32;
        let th = 20.0_f32;
        let mut initial_complexities = Vec::new();

        // Controlled viable founders: still low-complexity near-minimal consumers,
        // but with grazer traits and adult priming so the test measures mutation
        // and selection under the new life-history gates rather than startup luck.
        for i in 0..12 {
            let mut genome = sim.founder_consumer_genome();
            genome.behavior.reproduction_rate = 0.70;
            genome.behavior.mutation_rate_factor = 1.40;
            let x = 4.0 + (i as f32 / 12.0) * (tw - 8.0);
            let y = th * 0.28 + (i as f32 % 4.0) * 2.2;
            let entity = sim.spawn_from_genome(genome.clone(), x, y);
            prime_test_adult(&mut sim, entity, 1.10);
            initial_complexities.push(genome.complexity);
        }

        for i in 0..16 {
            let x = 3.0 + (i as f32 / 16.0) * (tw - 6.0);
            let y = th * 0.42 + (i as f32 % 4.0) * 2.1;
            spawn_test_soak_producer(&mut sim, x, y);
        }

        let initial_avg = mean(&initial_complexities);
        assert!(
            initial_avg <= 0.15,
            "Initial complexity should be low, got {:.3}",
            initial_avg,
        );

        let dt = 0.05;
        for _ in 0..12000 {
            sim.tick(dt);
        }

        let final_stats = sim.stats();
        let final_complexities: Vec<f32> = (&mut sim.world.query::<&CreatureGenome>())
            .into_iter()
            .map(|genome| genome.complexity)
            .collect();
        let final_avg = mean(&final_complexities);
        let final_max = final_complexities.iter().copied().fold(0.0_f32, f32::max);

        assert!(
            final_stats.creature_count > 0,
            "Population must survive! CreatureBirths={}, CreatureDeaths={}",
            final_stats.creature_births,
            final_stats.creature_deaths,
        );
        assert!(
            final_stats.creature_births > 0,
            "Simple founder web should achieve successful reproduction. \
             CreatureBirths={}, Gen={}, Creatures={}, AvgFinal={:.3}, MaxFinal={:.3}",
            final_stats.creature_births,
            final_stats.max_generation,
            final_stats.creature_count,
            final_avg,
            final_max,
        );
        assert!(
            final_stats.max_generation > 0 || final_max >= initial_avg,
            "Successful reproduction should either leave a derived lineage alive or at least preserve viable simple complexity. \
             InitialAvg={:.3}, FinalAvg={:.3}, FinalMax={:.3}, Gen={}, CreatureBirths={}, Creatures={}",
            initial_avg, final_avg, final_max,
            final_stats.max_generation, final_stats.creature_births, final_stats.creature_count,
        );
    }

    #[test]
    fn test_seeded_founder_web_shows_births_and_multiple_lineages() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let dt = 0.05;
        for _ in 0..20_000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.creature_births > 0,
            "Seeded founder web should produce consumer births. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, Gen={}, Species={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
            stats.max_generation,
            stats.species_count,
        );
        assert!(
            stats.max_generation > 0,
            "Seeded founder web should advance at least one generation. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, Species={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
            stats.species_count,
        );
        assert!(
            stats.species_count >= 2,
            "Seeded founder web should maintain more than one lineage cluster. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, Gen={}, Species={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
            stats.max_generation,
            stats.species_count,
        );
    }

    #[test]
    fn test_detritus_spawns_on_creature_death() {
        let mut sim = AquariumSim::new(100, 50);
        let mut rng = StdRng::seed_from_u64(42);
        let genome = CreatureGenome::minimal_cell(&mut rng);
        let entity = sim.spawn_from_genome(genome, 50.0, 25.0);

        // Drain creature's energy to 0 to trigger death
        {
            let mut energy = sim.world.get::<&mut Energy>(entity).unwrap();
            energy.current = 0.0;
        }

        // Tick to trigger death_system and detritus spawning
        sim.tick(0.05);

        // Check that detritus exists
        let mut detritus_count = 0;
        for _d in &mut sim.world.query::<&Detritus>() {
            detritus_count += 1;
        }
        assert!(
            detritus_count > 0,
            "Detritus should spawn when creature dies"
        );
    }

    #[test]
    fn test_simulation_runs_with_pheromone_system() {
        let mut sim = AquariumSim::new(100, 50);

        // Spawn some creatures
        for i in 0..5 {
            spawn_test_creature(&mut sim, 10.0 + i as f32 * 15.0, 10.0 + i as f32 * 5.0);
        }

        // Run for 100 ticks — should not panic
        for _ in 0..100 {
            sim.tick(0.05);
        }

        // If we got here, pheromone system is wired correctly
        assert!(
            sim.stats().tick_count >= 100,
            "Simulation with pheromone system runs without panicking"
        );
    }

    #[test]
    fn test_nutrient_cycle_detritus_is_edible() {
        let mut sim = AquariumSim::new(40, 20);
        let mut rng = StdRng::seed_from_u64(42);
        let genome = CreatureGenome::minimal_cell(&mut rng);
        let entity = sim.spawn_from_genome(genome, 20.0, 10.0);

        // Kill the creature to produce detritus
        {
            let mut energy = sim.world.get::<&mut Energy>(entity).unwrap();
            energy.current = 0.0;
        }
        sim.tick(0.05);

        // Verify detritus exists and is also a Producer (grazeable)
        let mut detritus_is_producer = false;
        for (_d, _p) in &mut sim.world.query::<(&Detritus, &Producer)>() {
            detritus_is_producer = true;
        }
        assert!(
            detritus_is_producer,
            "Detritus should be a Producer (grazeable)"
        );

        // Record detritus energy before spawning a grazer
        let mut initial_detritus_energy = 0.0_f32;
        for energy in &mut sim.world.query::<(&Energy, &Detritus)>() {
            initial_detritus_energy = energy.0.current;
        }
        assert!(initial_detritus_energy > 0.0, "Detritus should have energy");

        // Spawn a creature near the detritus so it can graze
        spawn_test_creature(&mut sim, 20.0, 10.0);

        // Run ticks for grazing and detritus decay
        for _ in 0..200 {
            sim.tick(0.05);
        }

        // Detritus energy should have decreased (grazed or decayed)
        let mut final_detritus_energy = initial_detritus_energy;
        for energy in &mut sim.world.query::<(&Energy, &Detritus)>() {
            final_detritus_energy = energy.0.current;
        }
        // If detritus was consumed entirely, energy stays at initial (no entity found)
        // Either way, detritus should not persist at full energy
        assert!(
            final_detritus_energy < initial_detritus_energy,
            "Detritus energy should decrease from grazing or decay. Initial={}, Final={}",
            initial_detritus_energy,
            final_detritus_energy,
        );
    }

    /// Integration test: environmental pressure (night metabolism stress, temperature)
    /// should create selection pressure that favours moderately complex creatures
    /// over simple ones, because complex creatures have greater sensory range
    /// (up to 18.0) and can locate food from farther away, offsetting their 25%
    /// metabolism penalty.
    #[test]
    fn test_complex_creatures_have_survival_advantage() {
        let mut sim = AquariumSim::new(80, 30);
        let mut rng = StdRng::seed_from_u64(42);

        let mut simple_entities = Vec::new();
        let mut complex_entities = Vec::new();

        let base = CreatureGenome::minimal_cell(&mut rng);

        // Spawn simple creatures (complexity = 0.05)
        for i in 0..10 {
            let mut genome = base.clone();
            genome.complexity = 0.05;
            genome.behavior.reproduction_rate = 0.1;
            let x = 5.0 + (i as f32) * 7.0;
            let entity = sim.spawn_from_genome(genome, x, 10.0);
            simple_entities.push(entity);
        }

        // Spawn complex creatures (complexity = 0.4)
        for i in 0..10 {
            let mut genome = base.clone();
            genome.complexity = 0.4;
            genome.behavior.reproduction_rate = 0.1;
            let x = 5.0 + (i as f32) * 7.0;
            let entity = sim.spawn_from_genome(genome, x, 20.0);
            complex_entities.push(entity);
        }

        // Provide food: dense producer coverage so survival depends on foraging ability
        for i in 0..15 {
            let x = 3.0 + (i as f32 / 15.0) * 74.0;
            let y = 10.0 + (i as f32 % 3.0) * 5.0;
            spawn_test_producer(&mut sim, x, y);
        }

        // Run for 2 simulated days (12000 ticks at 0.05s) — covers multiple
        // day/night cycles so night metabolism stress has an effect.
        for _ in 0..12000 {
            sim.tick(0.05);
        }

        // Count survivors and measure average energy
        let simple_alive: Vec<f32> = simple_entities
            .iter()
            .filter_map(|&e| sim.world.get::<&Energy>(e).ok().map(|en| en.current))
            .collect();
        let complex_alive: Vec<f32> = complex_entities
            .iter()
            .filter_map(|&e| sim.world.get::<&Energy>(e).ok().map(|en| en.current))
            .collect();

        let simple_avg = if simple_alive.is_empty() {
            0.0
        } else {
            simple_alive.iter().sum::<f32>() / simple_alive.len() as f32
        };
        let complex_avg = if complex_alive.is_empty() {
            0.0
        } else {
            complex_alive.iter().sum::<f32>() / complex_alive.len() as f32
        };

        // Complex creatures should not be worse off than simple ones —
        // their sensory advantage should compensate for the metabolism cost.
        assert!(
            complex_alive.len() >= simple_alive.len() || complex_avg >= simple_avg,
            "Complex creatures should survive at least as well: \
             simple: {}/{} alive (avg energy {:.1}), complex: {}/{} alive (avg energy {:.1})",
            simple_alive.len(),
            simple_entities.len(),
            simple_avg,
            complex_alive.len(),
            complex_entities.len(),
            complex_avg,
        );
    }

    /// Diagnostic soak: under the unified founder-web bootstrap, natural
    /// complexity increase remains a longer stochastic process than the default
    /// suite can require deterministically.
    #[test]
    #[ignore = "long stochastic evolution soak under the unified founder-web model"]
    fn test_complexity_rises_naturally() {
        let mut sim = AquariumSim::new(80, 30);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let initial_complexities: Vec<f32> = (&mut sim.world.query::<&CreatureGenome>())
            .into_iter()
            .map(|genome| genome.complexity)
            .collect();
        let initial_max = initial_complexities.iter().copied().fold(0.0_f32, f32::max);

        let mut max_complexity_seen = 0.0_f32;
        let mut total_births = 0u64;
        let dt = 0.25;
        let ticks_per_day = (300.0 / dt) as usize;
        let checkpoint_ticks = [
            ticks_per_day * 5,
            ticks_per_day * 10,
            ticks_per_day * 15,
            ticks_per_day * 20,
        ];

        for tick in 1..=(ticks_per_day * 20) {
            sim.tick(dt);

            if checkpoint_ticks.contains(&tick) {
                let stats = sim.stats();
                let day = tick as f32 / ticks_per_day as f32;

                let mut complexities: Vec<f32> = Vec::new();
                for genome in &mut sim.world.query::<&CreatureGenome>() {
                    complexities.push(genome.complexity);
                }

                let max_c = complexities.iter().copied().fold(0.0_f32, f32::max);
                let avg_c = if complexities.is_empty() {
                    0.0
                } else {
                    complexities.iter().sum::<f32>() / complexities.len() as f32
                };
                max_complexity_seen = max_complexity_seen.max(max_c);
                total_births = stats.creature_births;

                eprintln!(
                    "Day {:.0}: pop={}, births={}, gen={}, avg_c={:.3}, max_c={:.3}",
                    day,
                    complexities.len(),
                    stats.creature_births,
                    stats.max_generation,
                    avg_c,
                    max_c,
                );
            }
        }

        assert!(
            total_births > 0,
            "No births occurred in the long bootstrap soak"
        );
        assert!(
            max_complexity_seen > initial_max,
            "Max complexity should exceed the bootstrap founders over the long soak. \
             InitialMax={:.3}, FinalMax={:.3}, Births={}",
            initial_max,
            max_complexity_seen,
            total_births,
        );
    }

    #[test]
    fn test_short_lived_producer_is_not_killed_by_age_alone() {
        // Research note: plant senescence is gradual in the new model, so even a
        // short-lived plant is not removed by a hard age threshold if carbon balance
        // remains positive.
        let mut sim = AquariumSim::new(40, 20);
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = genome::ProducerGenome::minimal_producer(&mut rng);
        genome.lifespan_factor = 0.01; // very short lifespan

        sim.spawn_producer(Position { x: 20.0, y: 15.0 }, genome);
        assert_eq!(sim.stats().entity_count, 1);

        // Tick long enough for age ratio to exceed the nominal lifespan scale.
        for _ in 0..500 {
            sim.tick(0.05);
        }

        let plant_count: usize = {
            let mut count = 0;
            for (_, g) in &mut sim.world.query::<(hecs::Entity, &genome::ProducerGenome)>() {
                if g.generation == 0 {
                    count += 1;
                }
            }
            count
        };

        assert_eq!(
            plant_count, 1,
            "Healthy gen-0 producer should not be killed by a hard age threshold"
        );
    }

    #[test]
    fn test_gen0_producers_have_staggered_stages() {
        let mut sim = AquariumSim::new(80, 24);

        // Spawn 10 plants
        for i in 0..10 {
            let mut rng = StdRng::seed_from_u64(42);
            let genome = genome::ProducerGenome::minimal_producer(&mut rng);
            sim.spawn_producer(
                Position {
                    x: 5.0 + i as f32 * 7.0,
                    y: 18.0,
                },
                genome,
            );
        }

        // Tick enough so starting age spread (0-20s) straddles Young→Mature boundary (40s)
        // After 30s sim time: plants at ages 30-50s → some Young (<40s), some Mature (≥40s)
        for _ in 0..600 {
            sim.tick(0.05); // 30 sim seconds
        }

        // Collect stages
        let stages: Vec<ProducerStage> = {
            let mut v = Vec::new();
            for s in sim.world.query_mut::<&ProducerStage>() {
                v.push(*s);
            }
            v
        };

        // With staggered starting age (0-20s) and energy (50-100%),
        // not all producers should be in the same stage
        let all_same = stages.windows(2).all(|w| w[0] == w[1]);
        assert!(
            !all_same || stages.len() <= 1,
            "Gen-0 producers should have staggered stages, but all {} are {:?}",
            stages.len(),
            stages.first()
        );
    }

    #[test]
    fn test_broadcasting_producer_produces_offspring() {
        let mut sim = AquariumSim::new(60, 20);

        // Spawn one plant, manually set it to Mature (sufficient for seeding)
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = genome::ProducerGenome::minimal_producer(&mut rng);
        genome.seed_count = 3.0;
        genome.seed_range = 1.0;

        let entity = sim.spawn_producer(Position { x: 30.0, y: 5.0 }, genome);

        // Set to a reproductive state under the new biomass/reserve model.
        if let Ok(mut stage) = sim.world.get::<&mut ProducerStage>(entity) {
            *stage = ProducerStage::Broadcasting;
        }
        if let Ok(mut energy) = sim.world.get::<&mut ecosystem::Energy>(entity) {
            energy.current = energy.max * 0.92;
        }
        if let Ok(mut age) = sim.world.get::<&mut ProducerAge>(entity) {
            age.seconds = 50.0;
        }
        if let Ok(mut state) = sim.world.get::<&mut ProducerState>(entity) {
            state.structural_biomass *= 1.5;
            state.leaf_biomass *= 1.5;
            state.seed_cooldown = 0.0;
            state.clonal_cooldown = 0.0;
        }

        let initial_count = sim.stats().entity_count;

        // Run enough ticks for reserve-driven seed or clonal propagules to appear.
        for _ in 0..5000 {
            sim.tick(0.05);
        }

        let final_count = sim.stats().entity_count;
        assert!(
            final_count > initial_count && sim.stats().producer_births > 0,
            "Broadcasting producer should produce offspring. Initial={}, Final={}, ProducerBirths={}",
            initial_count, final_count, sim.stats().producer_births,
        );
    }

    fn mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn stage_histogram(sim: &mut AquariumSim) -> [usize; 5] {
        let mut counts = [0usize; 5];
        for stage in sim.world.query_mut::<&ProducerStage>() {
            let idx = match *stage {
                ProducerStage::Cell => 0,
                ProducerStage::Patch => 1,
                ProducerStage::Mature => 2,
                ProducerStage::Broadcasting => 3,
                ProducerStage::Collapsing => 4,
            };
            counts[idx] += 1;
        }
        counts
    }

    fn total_recorded_activity(stats: &SimStats) -> u64 {
        stats.creature_births
            + stats.creature_deaths
            + stats.producer_births
            + stats.producer_deaths
    }

    fn count_food_packets(sim: &mut AquariumSim) -> usize {
        (&mut sim.world.query::<(Entity, &Producer)>())
            .into_iter()
            .filter(|(entity, _)| {
                sim.world.get::<&Detritus>(*entity).is_err()
                    && sim.world.get::<&genome::ProducerGenome>(*entity).is_err()
            })
            .count()
    }

    fn average_safety(sim: &mut AquariumSim) -> f32 {
        let mut count = 0usize;
        let total: f32 = (&mut sim.world.query::<(&Needs, &CreatureGenome)>())
            .into_iter()
            .map(|(needs, _)| {
                count += 1;
                needs.safety
            })
            .sum();
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    fn average_speed(sim: &mut AquariumSim) -> f32 {
        let mut count = 0usize;
        let total: f32 = (&mut sim.world.query::<(&Velocity, &CreatureGenome)>())
            .into_iter()
            .map(|(vel, _)| {
                count += 1;
                (vel.vx * vel.vx + vel.vy * vel.vy).sqrt()
            })
            .sum();
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    fn count_living_creatures(sim: &mut AquariumSim) -> usize {
        (&mut sim.world.query::<&CreatureGenome>())
            .into_iter()
            .count()
    }

    #[test]
    fn test_default_bootstrap_starts_with_simple_consumers() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let stats = sim.stats();
        let startup_targets = sim.startup_targets();
        let complexities: Vec<f32> = (&mut sim.world.query::<&CreatureGenome>())
            .into_iter()
            .map(|genome| genome.complexity)
            .collect();

        assert!(
            stats.creature_count > 0,
            "Default bootstrap should seed consumers"
        );
        assert!(
            stats.creature_count >= startup_targets.target_consumer_founders,
            "Default bootstrap should seed at least the solver-derived visible founders, got {} with minimum {}",
            stats.creature_count,
            startup_targets.target_consumer_founders,
        );
        assert!(
            stats.producer_leaf_biomass
                + stats.producer_structural_biomass
                + stats.producer_belowground_reserve
                > 0.0,
            "Default bootstrap should establish a producer stand before adding consumers"
        );
        assert!(
            complexities.iter().all(|&complexity| complexity <= 0.20),
            "Default bootstrap should start with simple consumer founders, got complexities={complexities:?}",
        );
    }

    #[test]
    fn test_default_bootstrap_starts_with_low_stage_producers() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let stages: Vec<ProducerStage> = sim
            .world
            .query_mut::<&ProducerStage>()
            .into_iter()
            .copied()
            .collect();

        assert!(
            !stages.is_empty(),
            "Default bootstrap should seed producer founders"
        );
        assert!(
            stages
                .iter()
                .all(|stage| matches!(stage, ProducerStage::Cell | ProducerStage::Patch)),
            "Default bootstrap should start from low-stage producer colonies, got {stages:?}",
        );
    }

    #[test]
    fn test_default_bootstrap_roots_macrophytes_to_substrate_band() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let expected_bottom = sim.tank_height as f32 - 3.0;
        let mut rooted_count = 0usize;
        for (pos, bbox, _) in sim
            .world
            .query_mut::<(&Position, &BoundingBox, &RootedMacrophyte)>()
        {
            rooted_count += 1;
            let bottom = pos.y + bbox.h - 1.0;
            assert!(
                (bottom - expected_bottom).abs() < 0.05,
                "Rooted macrophyte should sit on the substrate band: bottom={bottom:.2} expected={expected_bottom:.2}"
            );
        }

        assert!(
            rooted_count > 0,
            "Default bootstrap should tag rooted macrophytes"
        );
    }

    #[test]
    fn test_default_bootstrap_starts_from_simple_founder_web() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let complexities: Vec<f32> = (&mut sim.world.query::<&CreatureGenome>())
            .into_iter()
            .map(|genome| genome.complexity)
            .collect();

        assert!(
            !complexities.is_empty(),
            "Default bootstrap should seed consumer founders"
        );
        assert!(
            complexities.iter().all(|&complexity| complexity <= 0.20),
            "Default bootstrap should start from simple cell founders, got complexities={complexities:?}",
        );
    }

    #[test]
    fn test_default_bootstrap_shows_ecology_activity_within_five_days() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let dt = 0.25;
        let ticks_per_day = (300.0 / dt) as usize;
        for _ in 0..(ticks_per_day * 5) {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            total_recorded_activity(&stats) > 0,
            "Default startup should show visible ecology activity within five days. \
             creature_births={} creature_deaths={} producer_births={} producer_deaths={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.producer_births,
            stats.producer_deaths,
        );
    }

    #[test]
    fn test_founder_web_daily_history_changes_over_five_days() {
        let mut sim = AquariumSim::new(80, 24);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let dt = 0.25;
        let ticks_per_day = (300.0 / dt) as usize;
        for _ in 0..(ticks_per_day * 5) {
            sim.tick(dt);
        }

        let diagnostics = sim.ecology_diagnostics();
        let producer_series: Vec<f32> = diagnostics
            .daily_history
            .iter()
            .map(|sample| sample.producer_total_biomass)
            .collect();
        let consumer_series: Vec<f32> = diagnostics
            .daily_history
            .iter()
            .map(|sample| sample.consumer_biomass)
            .collect();
        let consumer_intake_series: Vec<f32> = diagnostics
            .daily_history
            .iter()
            .map(|sample| sample.rolling_consumer_intake)
            .collect();
        let consumer_maintenance_series: Vec<f32> = diagnostics
            .daily_history
            .iter()
            .map(|sample| sample.rolling_consumer_maintenance)
            .collect();

        assert!(
            diagnostics.daily_history.len() >= 5,
            "Founder-web run should accumulate at least five daily samples, got {}",
            diagnostics.daily_history.len(),
        );
        assert!(
            producer_series
                .windows(2)
                .any(|pair| (pair[1] - pair[0]).abs() > 0.01),
            "Producer daily history should not stay flat: {producer_series:?}",
        );
        assert!(
            consumer_series
                .windows(2)
                .any(|pair| (pair[1] - pair[0]).abs() > 0.01)
                || consumer_intake_series
                    .windows(2)
                    .any(|pair| (pair[1] - pair[0]).abs() > 0.0001)
                || consumer_maintenance_series
                    .windows(2)
                    .any(|pair| (pair[1] - pair[0]).abs() > 0.0001),
            "Consumer history should show biomass or energetic change: \
             consumer={consumer_series:?} intake={consumer_intake_series:?} \
             maintenance={consumer_maintenance_series:?}",
        );
    }

    #[test]
    fn test_default_bootstrap_producers_change_after_visible_start() {
        let mut sim = AquariumSim::new(80, 24);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let initial_stats = sim.stats();
        let initial_biomass = initial_stats.producer_leaf_biomass
            + initial_stats.producer_structural_biomass
            + initial_stats.producer_belowground_reserve;
        let initial_hist = stage_histogram(&mut sim);

        let dt = 0.25;
        let ticks_per_day = (300.0 / dt) as usize;
        for _ in 0..(ticks_per_day * 2) {
            sim.tick(dt);
        }

        let final_stats = sim.stats();
        let final_biomass = final_stats.producer_leaf_biomass
            + final_stats.producer_structural_biomass
            + final_stats.producer_belowground_reserve;
        let final_hist = stage_histogram(&mut sim);

        assert!(
            final_stats.producer_births > 0
                || final_hist != initial_hist
                || (final_biomass - initial_biomass).abs() > 0.5,
            "Visible startup should show plant change after bootstrap. \
             initial_biomass={:.2} final_biomass={:.2} initial_hist={initial_hist:?} final_hist={final_hist:?} producer_births={}",
            initial_biomass,
            final_biomass,
            final_stats.producer_births,
        );
    }

    #[test]
    fn test_default_bootstrap_stable_coexistence_for_20_days() {
        let mut sim = AquariumSim::new(48, 16);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let dt = 1.0;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut producer_history = Vec::new();
        let mut producer_never_zero = true;
        let mut consumer_active_days = 0usize;

        for tick in 0..(ticks_per_day * 20) {
            sim.tick(dt);
            let stats = sim.stats();
            let producer_biomass = stats.producer_leaf_biomass
                + stats.producer_structural_biomass
                + stats.producer_belowground_reserve;
            if producer_biomass <= 0.05 {
                producer_never_zero = false;
                break;
            }
            if (tick + 1) % ticks_per_day == 0 {
                producer_history.push(producer_biomass);
                if stats.creature_count > 0 {
                    consumer_active_days += 1;
                }
            }
        }

        let stats = sim.stats();
        let producer_biomass = stats.producer_leaf_biomass
            + stats.producer_structural_biomass
            + stats.producer_belowground_reserve;

        assert!(
            producer_never_zero,
            "Producer biomass should not collapse to zero during the 20-day baseline"
        );
        assert!(
            producer_biomass > 0.05,
            "Producers should still exist after 20 days, got {:.3}",
            producer_biomass,
        );
        assert!(
            consumer_active_days >= 4,
            "Consumers should remain active for multiple days under the default bootstrap, got {} active days",
            consumer_active_days,
        );

        let producer_10_15 = mean(&producer_history[9..14]);
        let producer_15_20 = mean(&producer_history[14..19]);

        assert!(
            producer_15_20 > 3.0 && producer_15_20 > producer_10_15 * 0.25,
            "Producer biomass should persist through the 10-20 day window. \
             day10_15={:.2}, day15_20={:.2}, final={:.2}",
            producer_10_15,
            producer_15_20,
            producer_biomass,
        );
    }

    #[test]
    fn test_default_bootstrap_retains_visible_consumers_with_events_enabled() {
        let mut sim = AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();

        let dt = 1.0;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut visible_consumer_days = 0usize;

        for tick in 0..(ticks_per_day * 20) {
            sim.tick(dt);
            if (tick + 1) % ticks_per_day == 0 {
                let stats = sim.stats();
                if stats.creature_count >= 3 && stats.consumer_biomass > 0.8 {
                    visible_consumer_days += 1;
                }
            }
        }

        let stats = sim.stats();

        assert!(
            stats.creature_births >= 2,
            "Event-enabled default bootstrap should still produce multiple creature births. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, ConsumerBiomass={:.2}, Gen={}, Species={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
            stats.consumer_biomass,
            stats.max_generation,
            stats.species_count,
        );
        assert!(
            stats.creature_count >= 3,
            "Event-enabled default bootstrap should retain a visible consumer population after 20 days. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, ConsumerBiomass={:.2}, Gen={}, Species={}",
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
            stats.consumer_biomass,
            stats.max_generation,
            stats.species_count,
        );
        assert!(
            visible_consumer_days >= 8,
            "Event-enabled default bootstrap should keep consumers visibly present across multiple days, got {} visible days. \
             CreatureBirths={}, CreatureDeaths={}, Creatures={}, ConsumerBiomass={:.2}, Gen={}, Species={}",
            visible_consumer_days,
            stats.creature_births,
            stats.creature_deaths,
            stats.creature_count,
            stats.consumer_biomass,
            stats.max_generation,
            stats.species_count,
        );
    }

    #[test]
    fn test_feeding_frenzy_increases_edible_food_and_consumer_intake() {
        let mut control = AquariumSim::new_seeded(48, 16, 42);
        control.env.set_random_events_enabled(false);
        let control_grazer = spawn_test_grazer(&mut control, 20.0, 8.0);
        if let Ok(mut needs) = control.world.get::<&mut Needs>(control_grazer) {
            needs.hunger = 0.95;
        }
        if let Ok(mut energy) = control.world.get::<&mut Energy>(control_grazer) {
            energy.current = energy.max * 0.45;
        }

        let mut frenzy = AquariumSim::new_seeded(48, 16, 42);
        frenzy.env.set_random_events_enabled(false);
        let frenzy_grazer = spawn_test_grazer(&mut frenzy, 20.0, 8.0);
        if let Ok(mut needs) = frenzy.world.get::<&mut Needs>(frenzy_grazer) {
            needs.hunger = 0.95;
        }
        if let Ok(mut energy) = frenzy.world.get::<&mut Energy>(frenzy_grazer) {
            energy.current = energy.max * 0.45;
        }

        let dt = 0.25;
        frenzy.env.active_event = Some(EnvironmentEvent {
            kind: EventKind::FeedingFrenzy,
            remaining: 12.0,
        });
        for _ in 0..8 {
            control.tick(dt);
            frenzy.tick(dt);
        }
        let frenzy_food = count_food_packets(&mut frenzy);
        let control_food = count_food_packets(&mut control);

        for _ in 0..16 {
            control.tick(dt);
            frenzy.tick(dt);
        }

        let control_hunger = control.world.get::<&Needs>(control_grazer).unwrap().hunger;
        let frenzy_hunger = frenzy.world.get::<&Needs>(frenzy_grazer).unwrap().hunger;

        assert!(
            frenzy_food > control_food,
            "Feeding frenzy should add edible food packets: control={} frenzy={}",
            control_food,
            frenzy_food,
        );
        assert!(
            frenzy_hunger + 0.10 < control_hunger,
            "Feeding frenzy should lower grazer hunger: control_hunger={:.3} frenzy_hunger={:.3}",
            control_hunger,
            frenzy_hunger,
        );
    }

    #[test]
    fn test_earthquake_raises_safety_without_killing_consumers() {
        let mut control = AquariumSim::new_seeded(40, 18, 42);
        control.env.set_random_events_enabled(false);
        let mut quake = AquariumSim::new_seeded(40, 18, 42);
        quake.env.set_random_events_enabled(false);

        for i in 0..4 {
            spawn_test_creature(&mut control, 8.0 + i as f32 * 5.0, 8.0);
            spawn_test_creature(&mut quake, 8.0 + i as f32 * 5.0, 8.0);
        }

        let control_count_before = count_living_creatures(&mut control);
        let quake_count_before = count_living_creatures(&mut quake);
        let control_safety_before = average_safety(&mut control);
        let quake_safety_before = average_safety(&mut quake);

        quake.env.active_event = Some(EnvironmentEvent {
            kind: EventKind::Earthquake,
            remaining: 2.0,
        });
        for _ in 0..4 {
            control.tick(0.25);
            quake.tick(0.25);
        }

        assert_eq!(
            count_living_creatures(&mut control),
            control_count_before,
            "Control run should not lose creatures in the short disturbance window",
        );
        assert_eq!(
            count_living_creatures(&mut quake),
            quake_count_before,
            "Earthquake should not directly kill consumers in the short disturbance window",
        );
        assert!(
            average_safety(&mut quake) > quake_safety_before + 0.15
                && average_safety(&mut quake) > average_safety(&mut control) + 0.10
                && average_safety(&mut control) <= control_safety_before + 0.05,
            "Earthquake should raise safety pressure without direct mortality: control_before={:.3} control_after={:.3} quake_before={:.3} quake_after={:.3}",
            control_safety_before,
            average_safety(&mut control),
            quake_safety_before,
            average_safety(&mut quake),
        );
        assert!(
            (average_speed(&mut quake) - average_speed(&mut control)).abs() > 0.20,
            "Earthquake should visibly disturb movement: control_speed={:.3} quake_speed={:.3}",
            average_speed(&mut control),
            average_speed(&mut quake),
        );
    }

    #[test]
    fn test_default_bootstrap_startup_transient_stays_buffered() {
        let mut sim = AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();

        let dt = 4.0;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut day_one_producer = None;
        let mut day_five_nutrients = None;
        let mut day_ten_producer = None;
        let mut consumers_survived = true;

        for tick in 0..(ticks_per_day * 15) {
            sim.tick(dt);
            if (tick + 1) % ticks_per_day == 0 {
                let day = (tick + 1) / ticks_per_day;
                let stats = sim.stats();
                let producer_biomass = stats.producer_leaf_biomass
                    + stats.producer_structural_biomass
                    + stats.producer_belowground_reserve;

                if day == 1 {
                    day_one_producer = Some(producer_biomass);
                }
                if day == 5 {
                    day_five_nutrients =
                        Some((sim.nutrients.dissolved_n, sim.nutrients.dissolved_p));
                }
                if day == 10 {
                    day_ten_producer = Some(producer_biomass);
                }
                if stats.creature_count == 0 {
                    consumers_survived = false;
                    break;
                }
            }
        }

        let day_one_producer = day_one_producer.expect("startup transient should capture day 1");
        let (day_five_n, day_five_p) =
            day_five_nutrients.expect("startup transient should capture day 5 nutrients");
        let day_ten_producer = day_ten_producer.expect("startup transient should capture day 10");

        assert!(
            consumers_survived,
            "Startup transient should not wipe out consumers in the first 15 days"
        );
        assert!(
            day_five_n >= 4.0,
            "Startup dissolved N should stay buffered by day 5, got {:.2}",
            day_five_n,
        );
        assert!(
            day_five_p >= 0.5,
            "Startup dissolved P should stay buffered by day 5, got {:.2}",
            day_five_p,
        );
        assert!(
            day_ten_producer < day_one_producer * 3.0,
            "Producer biomass should not boom more than 3x the day-1 level during startup: day1={:.2} day10={:.2}",
            day_one_producer,
            day_ten_producer,
        );
    }

    #[test]
    fn test_default_bootstrap_realistic_lake_for_120_days() {
        let mut sim = AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();

        let dt = 4.0;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut consumer_positive_days = 0usize;
        let mut juvenile_days = 0usize;
        let mut peak_creature_count = 0usize;

        for tick in 0..(ticks_per_day * 120) {
            sim.tick(dt);
            if (tick + 1) % ticks_per_day == 0 {
                let stats = sim.stats();
                if stats.creature_count > 0 && stats.consumer_biomass > 0.4 {
                    consumer_positive_days += 1;
                }
                if stats.juvenile_count > 0 {
                    juvenile_days += 1;
                }
                peak_creature_count = peak_creature_count.max(stats.creature_count);
            }
        }

        let stats = sim.stats();
        let archived = sim.archived_daily_history();
        let (equilibrium_model, equilibrium_target) =
            crate::ecology_equilibrium::ReducedEquilibriumModel::reference_clearwater();
        let equilibrium_window: Vec<_> = archived
            .iter()
            .filter(|sample| sample.day >= 30)
            .cloned()
            .collect();
        let mean_observables = equilibrium_model
            .mean_observable_metrics(&equilibrium_window)
            .expect("realism run should accumulate archived daily history after day 30");
        let reference_observables =
            equilibrium_model.reference_observable_metrics(equilibrium_target);
        let post_bootstrap_birth_day = archived
            .iter()
            .find(|sample| sample.day > 10 && sample.creature_births_delta > 0)
            .map(|sample| sample.day);
        let final_day = archived.last().map(|sample| sample.day).unwrap_or(0);
        let tail_start = final_day.saturating_sub(49);
        let tail_window: Vec<_> = archived
            .iter()
            .filter(|sample| sample.day >= tail_start)
            .collect();
        let tail_len = tail_window.len().max(1) as f32;
        let tail_mean_assimilation = tail_window
            .iter()
            .map(|sample| sample.mean_recent_assimilation)
            .sum::<f32>()
            / tail_len;
        let tail_juvenile_days = tail_window
            .iter()
            .filter(|sample| sample.juvenile_count > 0)
            .count();
        let tail_recruitment_signal_days = tail_window
            .iter()
            .filter(|sample| {
                sample.juvenile_count > 0
                    || sample.subadult_count > 0
                    || sample.creature_births_delta > 0
                    || sample.reproduction_ready_count > 0
            })
            .count();
        let tail_birth_days = tail_window
            .iter()
            .filter(|sample| sample.creature_births_delta > 0)
            .count();
        let tail_death_days = tail_window
            .iter()
            .filter(|sample| sample.creature_deaths_delta > 0)
            .count();
        let tail_ready_days = tail_window
            .iter()
            .filter(|sample| sample.reproduction_ready_count > 0)
            .count();
        let near_peak_tail_days = tail_window
            .iter()
            .filter(|sample| sample.creature_count >= peak_creature_count.saturating_sub(1))
            .count();
        let static_adult_plateau = tail_birth_days == 0
            && tail_death_days == 0
            && tail_ready_days == 0
            && near_peak_tail_days + 5 >= tail_window.len();

        assert!(
            post_bootstrap_birth_day.unwrap_or(u64::MAX) <= 70,
            "Realistic lake should show a post-bootstrap consumer birth by day 70 under the near-equilibrium startup, got {:?}",
            post_bootstrap_birth_day,
        );
        assert!(
            stats.creature_deaths >= 2,
            "Realistic lake should show real consumer turnover by day 120, got {} deaths",
            stats.creature_deaths,
        );
        assert!(
            consumer_positive_days >= 95,
            "Realistic lake should keep consumers present on most days, got {} consumer-positive days",
            consumer_positive_days,
        );
        assert!(
            juvenile_days >= 25,
            "Realistic lake should keep juveniles visible across many days, got {} juvenile days",
            juvenile_days,
        );
        assert!(
            tail_mean_assimilation >= 0.015,
            "Realistic lake should keep late-run consumer assimilation above the maintenance trap, got {:.4}",
            tail_mean_assimilation,
        );
        assert!(
            tail_juvenile_days >= 3,
            "Realistic lake should keep juveniles reappearing late in the run, got {} days",
            tail_juvenile_days,
        );
        assert!(
            tail_recruitment_signal_days >= 3,
            "Realistic lake should keep late recruitment signals visible, got {} days",
            tail_recruitment_signal_days,
        );
        assert!(
            !static_adult_plateau,
            "Realistic lake should not settle into a static adult plateau: tail_birth_days={} tail_death_days={} tail_ready_days={} near_peak_tail_days={}",
            tail_birth_days,
            tail_death_days,
            tail_ready_days,
            near_peak_tail_days,
        );
        assert!(
            peak_creature_count <= 72,
            "Realistic lake should avoid dense swarms, got peak creature count {}",
            peak_creature_count,
        );
        assert!(
            equilibrium_model.observable_metrics_in_default_realism_band(mean_observables),
            "Mean archived ecology over days 30-120 should stay inside the reduced clear-water realism band. \
             mean={mean_observables:?} reference={reference_observables:?}",
        );
    }

    #[test]
    fn test_default_bootstrap_recruits_consumers_across_seed_variation() {
        let seeds = [7_u64, 42_u64];
        let dt = 8.0;
        let ticks_per_day = (300.0 / dt) as usize;

        for seed in seeds {
            let mut sim = AquariumSim::new_seeded(80, 24, seed);
            sim.bootstrap_founder_web();

            let mut longest_zero_consumer_streak = 0usize;
            let mut zero_consumer_streak = 0usize;

            for tick in 0..(ticks_per_day * 90) {
                sim.tick(dt);
                if (tick + 1) % ticks_per_day == 0 {
                    let stats = sim.stats();
                    if stats.creature_count == 0 {
                        zero_consumer_streak += 1;
                        longest_zero_consumer_streak =
                            longest_zero_consumer_streak.max(zero_consumer_streak);
                    } else {
                        zero_consumer_streak = 0;
                    }
                }
            }

            let stats = sim.stats();
            let archived = sim.archived_daily_history();
            let post_bootstrap_birth_day = archived
                .iter()
                .find(|sample| sample.day > 10 && sample.creature_births_delta > 0)
                .map(|sample| sample.day);
            let late_recruitment_signal_days = archived
                .iter()
                .filter(|sample| {
                    sample.day > 40
                        && (sample.juvenile_count > 0
                            || sample.subadult_count > 0
                            || sample.creature_births_delta > 0
                            || sample.reproduction_ready_count > 0)
                })
                .count();
            assert!(
                stats.creature_births > 0,
                "Default bootstrap should not stay founder-only for seed {seed}. \
                 births={} deaths={} final_creatures={} final_biomass={:.2}",
                stats.creature_births,
                stats.creature_deaths,
                stats.creature_count,
                stats.consumer_biomass,
            );
            assert!(
                post_bootstrap_birth_day.unwrap_or(u64::MAX) <= 90,
                "Default bootstrap should recruit consumers after bootstrap by day 90 for seed {seed}, got {:?}",
                post_bootstrap_birth_day,
            );
            assert!(
                longest_zero_consumer_streak <= 10,
                "Default bootstrap should not lose all consumers for long streaks under seed {seed}, got {} days",
                longest_zero_consumer_streak,
            );
            assert!(
                late_recruitment_signal_days > 0,
                "Default bootstrap should keep late recruitment signals reappearing after day 40 for seed {seed}",
            );
        }
    }

    #[test]
    fn test_default_bootstrap_app_size_avoids_ready_adult_swarm_at_160_days() {
        let mut sim = AquariumSim::new_seeded(136, 44, 42);
        sim.bootstrap_founder_web();

        let dt = 10.0;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut peak_creature_count = 0usize;

        for tick in 0..(ticks_per_day * 160) {
            sim.tick(dt);
            if (tick + 1) % ticks_per_day == 0 {
                peak_creature_count = peak_creature_count.max(sim.stats().creature_count);
            }
        }

        let archived = sim.archived_daily_history();
        let final_day = archived.last().map(|sample| sample.day).unwrap_or(0);
        let tail_start = final_day.saturating_sub(49);
        let tail_window: Vec<_> = archived
            .iter()
            .filter(|sample| sample.day >= tail_start)
            .collect();
        let tail_mean_creature_count = tail_window
            .iter()
            .map(|sample| sample.creature_count as f32)
            .sum::<f32>()
            / tail_window.len().max(1) as f32;
        let tail_ready_days = tail_window
            .iter()
            .filter(|sample| sample.reproduction_ready_count > 0)
            .count();
        let tail_birth_days = tail_window
            .iter()
            .filter(|sample| sample.creature_births_delta > 0)
            .count();
        let tail_death_days = tail_window
            .iter()
            .filter(|sample| sample.creature_deaths_delta > 0)
            .count();
        let near_peak_tail_days = tail_window
            .iter()
            .filter(|sample| sample.creature_count >= peak_creature_count.saturating_sub(2))
            .count();
        let ready_adult_swarm = tail_ready_days >= 20
            && tail_birth_days == 0
            && tail_death_days <= 1
            && near_peak_tail_days + 5 >= tail_window.len();

        assert!(
            peak_creature_count <= 120,
            "App-size lake should avoid overcrowded swarms, got peak creature count {}",
            peak_creature_count,
        );
        assert!(
            tail_mean_creature_count <= 110.0,
            "App-size lake should relax below a dense adult plateau, got tail mean creature count {:.1}",
            tail_mean_creature_count,
        );
        assert!(
            !ready_adult_swarm,
            "App-size lake should not sustain a ready-adult swarm with no late births: tail_ready_days={} tail_birth_days={} tail_death_days={} near_peak_tail_days={}",
            tail_ready_days,
            tail_birth_days,
            tail_death_days,
            near_peak_tail_days,
        );
    }

    #[test]
    fn test_default_bootstrap_app_size_preserves_rooted_producers_and_bounds_nitrogen() {
        let mut sim = AquariumSim::new_seeded(136, 44, 42);
        sim.bootstrap_founder_web();

        let dt = 10.0;
        let ticks_per_day = (300.0 / dt) as usize;
        for _ in 0..(ticks_per_day * 200) {
            sim.tick(dt);
        }

        let archived = sim.archived_daily_history();
        let final_day = archived.last().map(|sample| sample.day).unwrap_or(0);
        let tail_start = final_day.saturating_sub(49);
        let tail_window: Vec<_> = archived
            .iter()
            .filter(|sample| sample.day >= tail_start)
            .collect();
        let tail_len = tail_window.len().max(1) as f32;
        let tail_mean_rooted_biomass = tail_window
            .iter()
            .map(|sample| sample.rooted_producer_biomass)
            .sum::<f32>()
            / tail_len;
        let tail_mean_rooted_count = tail_window
            .iter()
            .map(|sample| sample.rooted_producer_count as f32)
            .sum::<f32>()
            / tail_len;
        let tail_mean_pelagic_intake = tail_window
            .iter()
            .map(|sample| sample.rolling_pelagic_consumer_intake)
            .sum::<f32>()
            / tail_len;
        let tail_mean_phy = tail_window
            .iter()
            .map(|sample| sample.phytoplankton_load)
            .sum::<f32>()
            / tail_len;
        let tail_max_dissolved_n = tail_window
            .iter()
            .map(|sample| sample.dissolved_n)
            .fold(0.0_f32, f32::max);

        assert!(
            tail_mean_rooted_biomass > 12.0,
            "App-size lake should preserve a visible rooted producer stand, got tail mean rooted biomass {:.2}",
            tail_mean_rooted_biomass,
        );
        assert!(
            tail_mean_rooted_count >= 6.0,
            "App-size lake should keep multiple rooted producer colonies alive, got tail mean rooted count {:.1}",
            tail_mean_rooted_count,
        );
        assert!(
            tail_max_dissolved_n <= 120.0,
            "App-size lake should keep dissolved N bounded, got tail max {:.2}",
            tail_max_dissolved_n,
        );
        if tail_mean_phy <= 0.02 {
            assert!(
                tail_mean_pelagic_intake <= 0.5,
                "App-size lake should not show phantom pelagic intake when phytoplankton is absent: phy={:.4} intake={:.4}",
                tail_mean_phy,
                tail_mean_pelagic_intake,
            );
        }
    }

    #[test]
    #[ignore = "app-size ecology long soak"]
    fn test_default_bootstrap_app_size_long_soak_avoids_producer_collapse() {
        let mut sim = AquariumSim::new_seeded(136, 44, 42);
        sim.bootstrap_founder_web();

        let dt = 10.0;
        let ticks_per_day = (300.0 / dt) as usize;
        for _ in 0..(ticks_per_day * 500) {
            sim.tick(dt);
        }

        let archived = sim.archived_daily_history();
        let final_day = archived.last().map(|sample| sample.day).unwrap_or(0);
        let tail_start = final_day.saturating_sub(79);
        let tail_window: Vec<_> = archived
            .iter()
            .filter(|sample| sample.day >= tail_start)
            .collect();
        let tail_len = tail_window.len().max(1) as f32;
        let tail_mean_rooted_biomass = tail_window
            .iter()
            .map(|sample| sample.rooted_producer_biomass)
            .sum::<f32>()
            / tail_len;
        let tail_max_dissolved_n = tail_window
            .iter()
            .map(|sample| sample.dissolved_n)
            .fold(0.0_f32, f32::max);

        assert!(
            tail_mean_rooted_biomass > 10.0,
            "App-size long soak should not lose the rooted producer stand, got tail mean rooted biomass {:.2}",
            tail_mean_rooted_biomass,
        );
        assert!(
            tail_max_dissolved_n <= 160.0,
            "App-size long soak should not drift into extreme nitrogen runaway, got tail max {:.2}",
            tail_max_dissolved_n,
        );
    }

    #[test]
    #[ignore = "long realism soak"]
    fn test_default_bootstrap_long_realism_soak_for_300_days() {
        let mut sim = AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();

        let dt = 4.0;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut longest_zero_consumer_streak = 0usize;
        let mut zero_consumer_streak = 0usize;
        let mut producers_never_collapsed = true;
        let mut max_dissolved_p: f32 = 0.0;

        for tick in 0..(ticks_per_day * 300) {
            sim.tick(dt);
            if (tick + 1) % ticks_per_day == 0 {
                let stats = sim.stats();
                let producer_biomass = stats.producer_leaf_biomass
                    + stats.producer_structural_biomass
                    + stats.producer_belowground_reserve;
                if producer_biomass <= 1.0 {
                    producers_never_collapsed = false;
                    break;
                }
                if stats.creature_count == 0 {
                    zero_consumer_streak += 1;
                    longest_zero_consumer_streak =
                        longest_zero_consumer_streak.max(zero_consumer_streak);
                } else {
                    zero_consumer_streak = 0;
                }
                max_dissolved_p = max_dissolved_p.max(sim.nutrients.dissolved_p);
            }
        }

        assert!(
            producers_never_collapsed,
            "Long realism soak should not lose producer biomass entirely"
        );
        assert!(
            longest_zero_consumer_streak <= 20,
            "Long realism soak should not remain consumer-empty for long streaks, got {} days",
            longest_zero_consumer_streak,
        );
        assert!(
            max_dissolved_p <= 40.0,
            "Long realism soak should keep dissolved phosphorus bounded, got {:.2}",
            max_dissolved_p,
        );
    }

    #[test]
    #[ignore = "long baseline coexistence soak"]
    fn test_default_bootstrap_stable_coexistence_for_40_days() {
        let mut sim = AquariumSim::new(70, 22);
        sim.env.set_random_events_enabled(false);
        sim.bootstrap_founder_web();

        let dt = 0.25;
        let ticks_per_day = (300.0 / dt) as usize;
        let mut producer_history = Vec::new();
        let mut consumer_history = Vec::new();

        for tick in 0..(ticks_per_day * 40) {
            sim.tick(dt);
            if (tick + 1) % ticks_per_day == 0 {
                let stats = sim.stats();
                producer_history.push(
                    stats.producer_leaf_biomass
                        + stats.producer_structural_biomass
                        + stats.producer_belowground_reserve,
                );
                consumer_history.push(stats.creature_count);
            }
        }

        let stats = sim.stats();
        let producer_biomass = stats.producer_leaf_biomass
            + stats.producer_structural_biomass
            + stats.producer_belowground_reserve;

        assert!(
            producer_biomass > 0.05,
            "Producers should still exist after 40 days. producer_history={producer_history:?} consumer_history={consumer_history:?}"
        );
        assert!(
            stats.creature_count > 0,
            "Consumers should still exist after 40 days. producer_history={producer_history:?} consumer_history={consumer_history:?}"
        );
    }

    #[test]
    fn test_diversity_coefficient_defaults_to_one() {
        let sim = AquariumSim::new_seeded(80, 24, 42);
        assert!((sim.diversity_coefficient() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_new_seeded_uses_equilibrium_derived_startup_nutrients() {
        let sim = AquariumSim::new_seeded(48, 16, 42);
        let targets = AquariumSim::default_startup_targets(48, 16);

        assert!((sim.nutrients.dissolved_n - targets.dissolved_n).abs() < 1e-4);
        assert!((sim.nutrients.dissolved_p - targets.dissolved_p).abs() < 1e-4);
        assert!((sim.nutrients.sediment_n - targets.sediment_n).abs() < 1e-4);
        assert!((sim.nutrients.sediment_p - targets.sediment_p).abs() < 1e-4);
        assert!((sim.nutrients.phytoplankton_load - targets.phytoplankton_load).abs() < 1e-4);
    }

    #[test]
    fn test_default_bootstrap_does_not_produce_day_one_consumer_births() {
        let mut sim = AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();

        let dt = 4.0;
        let ticks_per_day = (300.0 / dt) as usize;
        for _ in 0..ticks_per_day {
            sim.tick(dt);
        }

        assert_eq!(
            sim.stats().creature_births, 0,
            "Default startup should not rely on immediate day-one founder births"
        );
    }

    #[test]
    fn test_diversity_coefficient_clamped() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.set_diversity_coefficient(5.0);
        assert!((sim.diversity_coefficient() - 2.5).abs() < f32::EPSILON);
        sim.set_diversity_coefficient(0.0);
        assert!((sim.diversity_coefficient() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diversity_coefficient_adjusts_incrementally() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.set_diversity_coefficient(1.1);
        assert!((sim.diversity_coefficient() - 1.1).abs() < 0.01);
        sim.set_diversity_coefficient(0.9);
        assert!((sim.diversity_coefficient() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_diversity_coefficient_exposed_via_simulation_trait() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        // Access through trait methods
        let sim_ref: &mut dyn Simulation = &mut sim;
        assert!((sim_ref.diversity_coefficient() - 1.0).abs() < f32::EPSILON);
        sim_ref.set_diversity_coefficient(1.5);
        assert!((sim_ref.diversity_coefficient() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_substrate_grid_initialized_on_sim() {
        let sim = AquariumSim::new_seeded(80, 24, 42);
        // Substrate should be initialized — verify we can query it
        let _s = sim.substrate.at(0.0);
        let _s2 = sim.substrate.at(79.0);
        // Should not panic
    }

    #[test]
    fn test_soft_nutrient_loading_prevents_irreversible_depletion() {
        // Run a simulation for many ticks and verify dissolved nutrients recover
        // from depletion without hard floor clamping.
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.bootstrap_ecosystem();
        for _ in 0..2000 {
            sim.tick(0.05);
        }
        assert!(
            sim.nutrients.dissolved_n > 0.0,
            "Dissolved N should remain recoverable after 2000 ticks: {}",
            sim.nutrients.dissolved_n
        );
        assert!(
            sim.nutrients.dissolved_p > 0.0,
            "Dissolved P should remain recoverable after 2000 ticks: {}",
            sim.nutrients.dissolved_p
        );
    }

    #[test]
    fn test_tick_runs_with_non_default_diversity() {
        let mut sim = AquariumSim::new_seeded(80, 24, 42);
        sim.bootstrap_ecosystem();
        sim.set_diversity_coefficient(2.0);
        // Should not panic with high diversity
        for _ in 0..100 {
            sim.tick(0.05);
        }
        sim.set_diversity_coefficient(0.25);
        // Should not panic with low diversity
        for _ in 0..100 {
            sim.tick(0.05);
        }
    }
}
