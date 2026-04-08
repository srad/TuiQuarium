//! Ecosystem bootstrap and entity spawning.

use crate::behavior::BehaviorState;
use crate::boids::Boid;
use crate::brain::Brain;
use crate::components::*;
use crate::ecosystem::{self, Age, Detritus, Energy, Producer};
use crate::genome::{self, CreatureGenome, ProducerGenome};
use crate::needs::{NeedWeights, Needs};
use crate::phenotype::{self, derive_feeding, derive_physics, DerivedPhysics, FeedingCapability};
use crate::producer_lifecycle;

use hecs::Entity;
use rand::RngExt;

impl super::AquariumSim {
    pub fn spawn_creature(
        &mut self,
        pos: Position,
        vel: Velocity,
        bbox: BoundingBox,
        appearance: Appearance,
        anim: AnimationState,
    ) -> hecs::Entity {
        self.world.spawn((pos, vel, bbox, appearance, anim))
    }

    /// Spawn a fully simulated creature from a genome.
    /// Derives physics, feeding capability, brain — everything from the genome.
    pub fn spawn_from_genome(&mut self, genome: CreatureGenome, x: f32, y: f32) -> hecs::Entity {
        let physics = derive_physics(&genome);
        let feeding = derive_feeding(&genome, &physics);
        // Research note: longer consumer generation times reduce unrealistic
        // consumer overshoot relative to producer recovery in small ecosystems.
        // Calder (1984), Peters (1983): lifespan ∝ mass^0.25 across taxa.
        let size_longevity = physics.body_mass.max(0.1).powf(0.25);
        let max_ticks = (42_000.0 * genome.behavior.max_lifespan_factor * size_longevity) as u64;

        let vx: f32 = self.rng.random_range(-1.0..1.0);
        let vy: f32 = self.rng.random_range(-0.5..0.5);

        // Placeholder frame — render crate generates the real art
        let frame = AsciiFrame::from_rows(vec!["o"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: if vx >= 0.0 {
                Direction::Right
            } else {
                Direction::Left
            },
            color_index: genome.art.color_index(),
        };

        let bbox_w = 1.0_f32.max(genome.art.body_size * 3.0);
        let bbox_h = 1.0_f32.max(genome.art.body_size * 2.0);
        let need_weights = NeedWeights {
            hunger_rate: 0.01 + 0.02 * genome.behavior.metabolism_factor.min(1.5),
            ..NeedWeights::default()
        };

        let brain = Brain::from_genome_with_learning(&genome.brain, genome.behavior.learning_rate);
        let entity = self.world.spawn((
            Position { x, y },
            Velocity { vx, vy },
            BoundingBox {
                w: bbox_w,
                h: bbox_h,
            },
            appearance,
            AnimationState::new(0.2 / genome.anim.swim_speed),
            genome,
            physics.clone(),
            feeding,
            Energy::new(physics.max_energy),
            Age {
                ticks: 0,
                max_ticks,
            },
            Needs::default(),
            need_weights,
            BehaviorState::default(),
            Boid,
            brain,
        ));
        let _ = self.world.insert(entity, (ConsumerState::default(),));
        entity
    }

    /// Spawn a producer entity from a genome.
    /// The genome determines the producer's physics, appearance, and persistence.
    pub fn spawn_producer(
        &mut self,
        pos: Position,
        genome: genome::ProducerGenome,
    ) -> hecs::Entity {
        self.spawn_producer_with_propagule(pos, genome, None)
    }

    pub(crate) fn spawn_producer_with_propagule(
        &mut self,
        pos: Position,
        genome: genome::ProducerGenome,
        propagule_kind: Option<ProducerOrigin>,
    ) -> hecs::Entity {
        let physics = phenotype::derive_producer_physics(&genome);

        let feeding = FeedingCapability {
            max_prey_mass: 0.0,
            hunt_skill: 0.0,
            graze_skill: 0.0,
            is_producer: true,
        };

        // Research note: simple producer colonies need enough nominal lifespan
        // to turn over through grazing, dispersal, and replacement instead of
        // disappearing on a deterministic short timer.
        let base_ticks = 240_000 + (self.rng.random_range(0..240_000) as u64);
        let lifespan_ticks = (base_ticks as f32 * genome.lifespan_factor) as u64;

        let (struct_frac, leaf_frac, reserve_frac, below_frac, meristem_frac) = match propagule_kind
        {
            None if genome.generation == 0 => (
                0.40 + self.rng.random_range(0.0..0.45),
                0.45 + self.rng.random_range(0.0..0.40),
                0.55 + self.rng.random_range(0.0..0.35),
                0.35 + self.rng.random_range(0.0..0.25),
                0.35 + self.rng.random_range(0.0..0.25),
            ),
            Some(ProducerOrigin::Fragment) => (
                0.50 + genome.clonal_spread * 0.20,
                0.52 + genome.clonal_spread * 0.20,
                0.40 + genome.reserve_allocation * 0.20,
                0.42 + genome.clonal_spread * 0.26,
                0.48 + genome.clonal_spread * 0.22,
            ),
            _ => (
                0.28 + genome.seed_size.min(1.5) * 0.14,
                0.30 + genome.seed_size.min(1.5) * 0.16,
                (0.28 + genome.seed_size.min(1.5) * 0.12).max(0.36),
                0.16 + genome.seed_size.min(1.5) * 0.10,
                0.18 + genome.seed_size.min(1.5) * 0.10,
            ),
        };
        let belowground_target = genome.support_target_biomass()
            * (0.22 + genome.hardiness * 0.22 + genome.clonal_spread * 0.18);
        let state = ProducerState {
            structural_biomass: genome.support_target_biomass() * struct_frac,
            leaf_biomass: genome.active_target_biomass() * leaf_frac,
            belowground_reserve: belowground_target * below_frac,
            meristem_bank: meristem_frac.min(0.95),
            epiphyte_load: 0.0,
            seed_cooldown: 6.0,
            clonal_cooldown: 8.0,
            stress_time: 0.0,
            propagule_kind,
        };
        let initial_energy = physics.max_energy * reserve_frac;

        // Stagger founder ages so producer colonies do not all hit stage
        // thresholds together.
        let initial_age = if genome.generation == 0 {
            self.rng.random_range(0..200) as f32 / 10.0 // 0-20 seconds
        } else {
            0.0
        };
        let age = ProducerAge {
            seconds: initial_age,
        };
        let energy = Energy::new_with(initial_energy, physics.max_energy);
        let stage = producer_lifecycle::compute_stage(&genome, &state, &energy, &age);
        let (appearance, bbox) = producer_lifecycle::build_appearance_from_genome(&genome, stage);

        let entity = self.world.spawn((
            pos,
            Velocity { vx: 0.0, vy: 0.0 },
            bbox,
            appearance,
            AnimationState::new(0.8),
            Producer,
            energy,
            physics,
            feeding,
            genome,
            state,
            stage,
            age,
            Age {
                ticks: 0,
                max_ticks: lifespan_ticks,
            },
        ));
        if propagule_kind.is_some() {
            let _ = self.world.insert(entity, (RootedMacrophyte,));
            self.anchor_rooted_macrophyte(entity);
        }
        entity
    }

    /// Spawn a food pellet (producer) that sinks and can be eaten.
    pub fn spawn_food(
        &mut self,
        pos: Position,
        vel: Velocity,
        bbox: BoundingBox,
        appearance: Appearance,
    ) -> hecs::Entity {
        let physics = phenotype::DerivedPhysics {
            body_mass: 0.1,
            max_energy: 10.0,
            base_metabolism: 0.3, // Food decays
            max_speed: 0.0,
            acceleration: 0.0,
            turn_radius: 0.0,
            drag_coefficient: 0.0,
            visual_profile: 0.2,
            camouflage: 0.0,
            sensory_range: 0.0,
        };

        let feeding = FeedingCapability {
            max_prey_mass: 0.0,
            hunt_skill: 0.0,
            graze_skill: 0.0,
            is_producer: true,
        };

        self.world.spawn((
            pos,
            vel,
            bbox,
            appearance,
            AnimationState::new(1.0),
            Producer,
            Energy::new(10.0),
            physics,
            feeding,
        ))
    }

    /// Spawn detritus at a position (dead creature remains for nutrient cycling).
    pub(crate) fn spawn_detritus(&mut self, x: f32, y: f32, energy: f32) {
        let physics = DerivedPhysics {
            body_mass: 0.1,
            max_energy: energy,
            base_metabolism: 0.5, // Detritus decays faster than food
            max_speed: 0.0,
            acceleration: 0.0,
            turn_radius: 0.0,
            drag_coefficient: 0.0,
            visual_profile: 0.3,
            camouflage: 0.0,
            sensory_range: 0.0,
        };

        let feeding = FeedingCapability {
            max_prey_mass: 0.0,
            hunt_skill: 0.0,
            graze_skill: 0.0,
            is_producer: true,
        };

        let frame = AsciiFrame::from_rows(vec!["~"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 3, // Brownish/dark color
        };

        self.world.spawn((
            Position { x, y },
            Velocity { vx: 0.0, vy: 0.0 },
            BoundingBox { w: 1.0, h: 1.0 },
            appearance,
            AnimationState::new(2.0),
            Producer,
            Detritus,
            Energy {
                current: energy,
                max: energy,
            },
            physics,
            feeding,
        ));
    }

    fn rooted_macrophyte_top_y(&self, height: f32) -> f32 {
        (self.tank_height as f32 - 2.0 - height.max(1.0)).max(0.0)
    }

    pub(crate) fn rooted_settlement_y(&self) -> f32 {
        (self.tank_height as f32 - 3.0).max(1.0)
    }

    fn clamp_rooted_x(&self, x: f32, width: f32) -> f32 {
        let max_x = (self.tank_width as f32 - width.max(1.0)).max(0.0);
        x.clamp(0.0, max_x)
    }

    fn anchor_rooted_macrophyte(&mut self, entity: Entity) {
        if self.world.get::<&RootedMacrophyte>(entity).is_err() {
            return;
        }

        let bbox = match self.world.get::<&BoundingBox>(entity) {
            Ok(bbox) => bbox.clone(),
            Err(_) => return,
        };

        if let Ok(mut pos) = self.world.get::<&mut Position>(entity) {
            pos.x = self.clamp_rooted_x(pos.x, bbox.w);
            pos.y = self.rooted_macrophyte_top_y(bbox.h);
        }
    }

    fn founder_grazer_consumer_genome(&mut self) -> CreatureGenome {
        let mut genome = CreatureGenome::minimal_cell(&mut self.rng);
        // Research note: the unified model starts from simple motile
        // heterotrophs rather than from already complex animal-like founders.
        genome.art.body_size = 0.54 + self.rng.random_range(-0.04..0.04);
        genome.art.body_elongation = 0.40 + self.rng.random_range(-0.08..0.08);
        genome.art.body_height_ratio = 0.50 + self.rng.random_range(-0.08..0.08);
        genome.art.pattern_density = 0.06 + self.rng.random_range(-0.04..0.08);
        genome.behavior.aggression = 0.02 + self.rng.random_range(0.0..0.03);
        genome.behavior.hunting_instinct = 0.0;
        genome.behavior.mouth_size = 0.46 + self.rng.random_range(-0.05..0.05);
        genome.behavior.speed_factor = 0.94 + self.rng.random_range(-0.06..0.08);
        genome.behavior.metabolism_factor = 0.58 + self.rng.random_range(-0.05..0.05);
        genome.behavior.max_lifespan_factor = 2.12 + self.rng.random_range(-0.16..0.24);
        genome.behavior.reproduction_rate = 0.60 + self.rng.random_range(-0.06..0.08);
        genome.behavior.learning_rate = 0.0;
        genome.behavior.pheromone_sensitivity = 0.36 + self.rng.random_range(-0.06..0.08);
        genome.complexity = 0.10 + self.rng.random_range(-0.03..0.05);
        genome
    }

    fn founder_detritivore_consumer_genome(&mut self) -> CreatureGenome {
        let mut genome = CreatureGenome::minimal_cell(&mut self.rng);
        // Simulation assumption: the founder web should include more than one
        // low-complexity heterotroph strategy, otherwise the diversity metric
        // starts from an artificial single-clone cloud.
        genome.art.body_size = 0.46 + self.rng.random_range(-0.04..0.04);
        genome.art.body_elongation = 0.18 + self.rng.random_range(-0.08..0.08);
        genome.art.body_height_ratio = 0.70 + self.rng.random_range(-0.08..0.08);
        genome.art.pattern_density = 0.02 + self.rng.random_range(0.0..0.06);
        genome.behavior.aggression = 0.01 + self.rng.random_range(0.0..0.02);
        genome.behavior.hunting_instinct = 0.0;
        genome.behavior.mouth_size = 0.28 + self.rng.random_range(-0.05..0.05);
        genome.behavior.speed_factor = 0.72 + self.rng.random_range(-0.06..0.06);
        genome.behavior.metabolism_factor = 0.48 + self.rng.random_range(-0.05..0.05);
        genome.behavior.max_lifespan_factor = 2.35 + self.rng.random_range(-0.18..0.22);
        genome.behavior.reproduction_rate = 0.68 + self.rng.random_range(-0.05..0.08);
        genome.behavior.learning_rate = 0.0;
        genome.behavior.pheromone_sensitivity = 0.58 + self.rng.random_range(-0.08..0.10);
        genome.complexity = 0.06 + self.rng.random_range(-0.02..0.04);
        genome
    }

    fn founder_explorer_consumer_genome(&mut self) -> CreatureGenome {
        let mut genome = CreatureGenome::minimal_cell(&mut self.rng);
        // Simulation assumption: founder consumers also need a slightly more
        // mobile strategy so the baseline world explores producer patches
        // instead of clustering into one slow-moving lineage.
        genome.art.body_size = 0.72 + self.rng.random_range(-0.05..0.05);
        genome.art.body_elongation = 0.62 + self.rng.random_range(-0.08..0.08);
        genome.art.body_height_ratio = 0.34 + self.rng.random_range(-0.08..0.08);
        genome.art.pattern_density = 0.18 + self.rng.random_range(-0.04..0.08);
        genome.behavior.aggression = 0.04 + self.rng.random_range(0.0..0.03);
        genome.behavior.hunting_instinct = 0.10 + self.rng.random_range(0.0..0.05);
        genome.behavior.mouth_size = 0.60 + self.rng.random_range(-0.05..0.06);
        genome.behavior.speed_factor = 1.12 + self.rng.random_range(-0.08..0.10);
        genome.behavior.metabolism_factor = 0.72 + self.rng.random_range(-0.05..0.06);
        genome.behavior.max_lifespan_factor = 1.90 + self.rng.random_range(-0.14..0.20);
        genome.behavior.reproduction_rate = 0.60 + self.rng.random_range(-0.05..0.08);
        genome.behavior.learning_rate = 0.0;
        genome.behavior.pheromone_sensitivity = 0.30 + self.rng.random_range(-0.06..0.08);
        genome.complexity = 0.16 + self.rng.random_range(-0.03..0.03);
        genome
    }

    fn founder_consumer_genome_for_cohort(&mut self, cohort_index: usize) -> CreatureGenome {
        match cohort_index % 3 {
            0 => self.founder_grazer_consumer_genome(),
            1 => self.founder_detritivore_consumer_genome(),
            _ => self.founder_explorer_consumer_genome(),
        }
    }

    #[cfg(test)]
    pub(crate) fn founder_consumer_genome(&mut self) -> CreatureGenome {
        let cohort = self.rng.random_range(0..3);
        self.founder_consumer_genome_for_cohort(cohort)
    }

    fn founder_surface_producer_genome(&mut self) -> ProducerGenome {
        let mut genome = ProducerGenome::minimal_producer(&mut self.rng);
        genome.height_factor = 0.18 + self.rng.random_range(-0.05..0.06);
        genome.leaf_area = 0.78 + self.rng.random_range(-0.08..0.08);
        genome.branching = 0.48 + self.rng.random_range(-0.08..0.08);
        genome.curvature = 0.42 + self.rng.random_range(-0.10..0.10);
        genome.photosynthesis_rate = 1.16 + self.rng.random_range(-0.08..0.10);
        genome.hardiness = 0.34 + self.rng.random_range(-0.08..0.08);
        genome.seed_range = 1.18 + self.rng.random_range(-0.10..0.12);
        genome.seed_count = 0.96 + self.rng.random_range(-0.12..0.12);
        genome.seed_size = 0.52 + self.rng.random_range(-0.08..0.08);
        genome.clonal_spread = 0.26 + self.rng.random_range(-0.10..0.10);
        genome.nutrient_affinity = 1.06 + self.rng.random_range(-0.08..0.08);
        genome.epiphyte_resistance = 0.42 + self.rng.random_range(-0.08..0.08);
        genome.reserve_allocation = 0.42 + self.rng.random_range(-0.08..0.08);
        genome.complexity = 0.16 + self.rng.random_range(-0.05..0.06);
        genome
    }

    fn founder_mat_producer_genome(&mut self) -> ProducerGenome {
        let mut genome = ProducerGenome::minimal_producer(&mut self.rng);
        genome.stem_thickness = 0.54 + self.rng.random_range(-0.06..0.06);
        genome.height_factor = 0.14 + self.rng.random_range(-0.04..0.05);
        genome.leaf_area = 0.62 + self.rng.random_range(-0.08..0.08);
        genome.branching = 0.24 + self.rng.random_range(-0.06..0.08);
        genome.curvature = 0.26 + self.rng.random_range(-0.08..0.08);
        genome.photosynthesis_rate = 0.96 + self.rng.random_range(-0.08..0.08);
        genome.hardiness = 0.66 + self.rng.random_range(-0.08..0.08);
        genome.seed_range = 0.78 + self.rng.random_range(-0.08..0.08);
        genome.seed_count = 0.72 + self.rng.random_range(-0.08..0.10);
        genome.seed_size = 0.68 + self.rng.random_range(-0.08..0.08);
        genome.clonal_spread = 0.74 + self.rng.random_range(-0.08..0.08);
        genome.nutrient_affinity = 1.00 + self.rng.random_range(-0.08..0.08);
        genome.epiphyte_resistance = 0.56 + self.rng.random_range(-0.08..0.08);
        genome.reserve_allocation = 0.30 + self.rng.random_range(-0.08..0.08);
        genome.complexity = 0.20 + self.rng.random_range(-0.05..0.06);
        genome
    }

    fn founder_filament_producer_genome(&mut self) -> ProducerGenome {
        let mut genome = ProducerGenome::minimal_producer(&mut self.rng);
        genome.stem_thickness = 0.30 + self.rng.random_range(-0.05..0.06);
        genome.height_factor = 0.44 + self.rng.random_range(-0.08..0.08);
        genome.leaf_area = 0.56 + self.rng.random_range(-0.08..0.08);
        genome.branching = 0.56 + self.rng.random_range(-0.08..0.08);
        genome.curvature = 0.62 + self.rng.random_range(-0.10..0.10);
        genome.photosynthesis_rate = 1.08 + self.rng.random_range(-0.08..0.08);
        genome.hardiness = 0.44 + self.rng.random_range(-0.08..0.08);
        genome.seed_range = 0.92 + self.rng.random_range(-0.10..0.10);
        genome.seed_count = 0.82 + self.rng.random_range(-0.10..0.10);
        genome.seed_size = 0.58 + self.rng.random_range(-0.08..0.08);
        genome.clonal_spread = 0.48 + self.rng.random_range(-0.08..0.10);
        genome.nutrient_affinity = 1.10 + self.rng.random_range(-0.08..0.08);
        genome.epiphyte_resistance = 0.48 + self.rng.random_range(-0.08..0.08);
        genome.reserve_allocation = 0.36 + self.rng.random_range(-0.08..0.08);
        genome.complexity = 0.26 + self.rng.random_range(-0.06..0.08);
        genome
    }

    fn founder_producer_genome(&mut self, founder_index: usize) -> ProducerGenome {
        // Research note: trait-based phytoplankton and microbial-producer
        // ecology organizes primary producers along trade-offs in light capture,
        // nutrient use, and dispersal strategy (Litchman and Klausmeier, 2008),
        // so the unified founder web samples multiple low-complexity producer
        // trait combinations rather than one predefined taxon.
        match founder_index % 3 {
            0 => self.founder_surface_producer_genome(),
            1 => self.founder_mat_producer_genome(),
            _ => self.founder_filament_producer_genome(),
        }
    }

    fn initialize_founder_producer(
        &mut self,
        entity: Entity,
        genome: &ProducerGenome,
        founder_index: usize,
    ) {
        let structural_target = genome.support_target_biomass();
        let leaf_target = genome.active_target_biomass();
        let belowground_target =
            structural_target * (0.22 + genome.hardiness * 0.22 + genome.clonal_spread * 0.18);

        let (
            age_seconds,
            structural_fill,
            leaf_fill,
            below_fill,
            meristem_bank,
            energy_fraction,
            seed_cooldown,
            clonal_cooldown,
        ): (f32, f32, f32, f32, f32, f32, f32, f32) = match founder_index % 4 {
            0 => (
                2.0 + self.rng.random_range(0.0..3.0),
                0.16 + self.rng.random_range(0.0..0.06),
                0.24 + self.rng.random_range(0.0..0.08),
                0.12 + self.rng.random_range(0.0..0.06),
                0.08 + self.rng.random_range(0.0..0.04),
                0.30 + self.rng.random_range(0.0..0.06),
                8.0 + self.rng.random_range(0.0..3.0),
                6.0 + self.rng.random_range(0.0..2.0),
            ),
            1 => (
                4.0 + self.rng.random_range(0.0..4.0),
                0.24 + self.rng.random_range(0.0..0.08),
                0.34 + self.rng.random_range(0.0..0.08),
                0.18 + self.rng.random_range(0.0..0.08),
                0.12 + self.rng.random_range(0.0..0.05),
                0.40 + self.rng.random_range(0.0..0.08),
                6.0 + self.rng.random_range(0.0..3.0),
                4.0 + self.rng.random_range(0.0..2.0),
            ),
            2 => (
                7.0 + self.rng.random_range(0.0..4.0),
                0.34 + self.rng.random_range(0.0..0.08),
                0.46 + self.rng.random_range(0.0..0.08),
                0.28 + self.rng.random_range(0.0..0.08),
                0.16 + self.rng.random_range(0.0..0.05),
                0.48 + self.rng.random_range(0.0..0.08),
                4.0 + self.rng.random_range(0.0..2.0),
                3.0 + self.rng.random_range(0.0..2.0),
            ),
            _ => (
                10.0 + self.rng.random_range(0.0..4.0),
                0.42 + self.rng.random_range(0.0..0.08),
                0.54 + self.rng.random_range(0.0..0.08),
                0.38 + self.rng.random_range(0.0..0.10),
                0.20 + self.rng.random_range(0.0..0.05),
                0.58 + self.rng.random_range(0.0..0.08),
                3.0 + self.rng.random_range(0.0..2.0),
                2.0 + self.rng.random_range(0.0..2.0),
            ),
        };

        let state_snapshot = if let Ok(mut state) = self.world.get::<&mut ProducerState>(entity) {
            // Research note: the unified model starts from low-biomass producer
            // colonies so early visible growth is honest rather than staged.
            state.structural_biomass = structural_target * structural_fill;
            state.leaf_biomass = leaf_target * leaf_fill;
            state.belowground_reserve = belowground_target * below_fill;
            state.meristem_bank = meristem_bank.min(0.90);
            state.epiphyte_load = 0.0;
            state.seed_cooldown = seed_cooldown;
            state.clonal_cooldown = clonal_cooldown;
            state.stress_time = 0.0;
            state.clone()
        } else {
            return;
        };

        let age_snapshot = ProducerAge {
            seconds: age_seconds,
        };
        if let Ok(mut age) = self.world.get::<&mut ProducerAge>(entity) {
            age.seconds = age_seconds;
        }

        let energy_snapshot = if let Ok(mut energy) = self.world.get::<&mut Energy>(entity) {
            energy.current = energy.max * energy_fraction.min(0.88);
            energy.clone()
        } else {
            return;
        };

        let stage = producer_lifecycle::compute_stage(
            genome,
            &state_snapshot,
            &energy_snapshot,
            &age_snapshot,
        );
        if let Ok(mut live_stage) = self.world.get::<&mut ProducerStage>(entity) {
            *live_stage = stage;
        }
        let (appearance, bbox) = producer_lifecycle::build_appearance_from_genome(genome, stage);
        let _ = self
            .world
            .insert(entity, (appearance, bbox, AnimationState::new(0.8)));
        self.anchor_rooted_macrophyte(entity);
    }

    fn initialize_founder_consumer(
        &mut self,
        entity: Entity,
        cohort_index: usize,
        total_founders: usize,
    ) {
        let (physics, genome) = {
            let physics = self.world.get::<&DerivedPhysics>(entity).unwrap().clone();
            let genome = self.world.get::<&CreatureGenome>(entity).unwrap().clone();
            (physics, genome)
        };
        let threshold = ecosystem::consumer_reproductive_threshold(&physics, &genome);
        let adult_slots = if total_founders >= 5 {
            2
        } else if total_founders >= 3 {
            1
        } else {
            1
        };

        let (
            age_fraction,
            founder_lifespan_scale,
            maturity_progress,
            reserve_buffer,
            reproductive_buffer,
            recent_assimilation,
            brood_cooldown,
            hunger,
            energy_fraction,
        ): (f32, f32, f32, f32, f32, f32, f32, f32, f32) = if cohort_index < adult_slots {
            (
                0.26 + self.rng.random_range(0.0..0.08),
                0.36 + self.rng.random_range(0.0..0.04),
                1.0,
                0.72 + self.rng.random_range(0.0..0.08),
                threshold * (0.46 + self.rng.random_range(0.0..0.16)),
                0.10 + self.rng.random_range(0.0..0.05),
                12.0 + self.rng.random_range(0.0..16.0),
                0.34 + self.rng.random_range(0.0..0.08),
                0.82 + self.rng.random_range(0.0..0.06),
            )
        } else if cohort_index < adult_slots + 3 {
            (
                0.12 + self.rng.random_range(0.0..0.08),
                0.50 + self.rng.random_range(0.0..0.06),
                0.54 + self.rng.random_range(0.0..0.18),
                0.48 + self.rng.random_range(0.0..0.10),
                threshold * (0.14 + self.rng.random_range(0.0..0.10)),
                0.05 + self.rng.random_range(0.0..0.04),
                8.0 + self.rng.random_range(0.0..12.0),
                0.42 + self.rng.random_range(0.0..0.10),
                0.74 + self.rng.random_range(0.0..0.08),
            )
        } else {
            (
                0.02 + self.rng.random_range(0.0..0.04),
                0.78 + self.rng.random_range(0.0..0.08),
                0.16 + self.rng.random_range(0.0..0.14),
                0.30 + self.rng.random_range(0.0..0.10),
                threshold * (0.02 + self.rng.random_range(0.0..0.04)),
                0.02 + self.rng.random_range(0.0..0.03),
                6.0 + self.rng.random_range(0.0..10.0),
                0.54 + self.rng.random_range(0.0..0.10),
                0.68 + self.rng.random_range(0.0..0.08),
            )
        };

        if let Ok(mut age) = self.world.get::<&mut Age>(entity) {
            age.max_ticks = ((age.max_ticks as f32) * founder_lifespan_scale).round() as u64;
            age.ticks = ((age.max_ticks as f32) * age_fraction).round() as u64;
        }
        if let Ok(mut state) = self.world.get::<&mut ConsumerState>(entity) {
            // Simulation assumption: default founders are age-structured enough
            // to show ecological turnover without pre-seeding a complex fauna.
            state.maturity_progress = maturity_progress;
            state.reserve_buffer = reserve_buffer;
            state.reproductive_buffer = reproductive_buffer;
            state.recent_assimilation = recent_assimilation;
            state.brood_cooldown = brood_cooldown;
        }
        if let Ok(mut needs) = self.world.get::<&mut Needs>(entity) {
            needs.hunger = hunger;
        }
        if let Ok(mut energy) = self.world.get::<&mut Energy>(entity) {
            energy.current = energy.max * energy_fraction.min(0.96);
        }
        if let Ok(mut vel) = self.world.get::<&mut Velocity>(entity) {
            // Simulation assumption: default consumer founders start with only
            // weak drift so the founder web tests trophic access first instead
            // of immediately losing consumers to random launch directions.
            vel.vx *= 0.15;
            vel.vy *= 0.15;
        }
    }

    pub fn bootstrap_ecosystem(&mut self) -> Vec<hecs::Entity> {
        let tw = self.tank_width as f32;
        let th = self.tank_height as f32;
        let target_producers = self.calibration.ecology.target_producer_count(
            self.tank_width,
            self.tank_height,
            self.max_producers(),
        );
        let spacing = self.calibration.ecology.founder_spacing;

        let mut seeded = 0usize;
        let mut attempts = 0usize;
        let producer_y = self.rooted_settlement_y();
        while seeded < target_producers && attempts < target_producers * 12 {
            attempts += 1;
            let frac = (seeded as f32 + 0.5) / target_producers.max(1) as f32;
            let x =
                (3.0 + frac * (tw - 6.0) + self.rng.random_range(-1.2..1.2)).clamp(3.0, tw - 3.0);
            let crowded = (&mut self.world.query::<(Entity, &ProducerGenome)>())
                .into_iter()
                .filter_map(|(entity, _)| self.world.get::<&Position>(entity).ok())
                .any(|pos| {
                    let dx = pos.x - x;
                    let dy = pos.y - producer_y;
                    dx * dx + dy * dy < spacing * spacing
                });
            if crowded {
                continue;
            }
            let genome = self.founder_producer_genome(seeded);
            let founder_kind = if self.rng.random_bool(0.65) {
                ProducerOrigin::Fragment
            } else {
                ProducerOrigin::Broadcast
            };
            let entity = self.spawn_producer_with_propagule(
                Position { x, y: producer_y },
                genome.clone(),
                Some(founder_kind),
            );
            self.initialize_founder_producer(entity, &genome, seeded);
            seeded += 1;
        }

        self.recompute_cached_stats();
        let producer_biomass = self.cached_producer_leaf_biomass
            + self.cached_producer_structural_biomass
            + self.cached_producer_belowground_reserve;
        let target_consumer_biomass = self
            .calibration
            .ecology
            .target_consumer_biomass(producer_biomass, self.cached_producer_leaf_biomass);
        let producer_positions: Vec<Position> = (&mut self.world.query::<(Entity, &Position)>())
            .into_iter()
            .filter_map(|(entity, pos)| {
                if self.world.get::<&ProducerGenome>(entity).is_ok() {
                    Some(pos.clone())
                } else {
                    None
                }
            })
            .collect();
        let initial_detritus_energy = (self.cached_producer_leaf_biomass * 0.22)
            .clamp(0.0, self.cached_producer_leaf_biomass * 0.35);
        let mut remaining_detritus_energy = initial_detritus_energy;
        let detritus_packet_energy = 1.6;
        let detritus_packets = ((remaining_detritus_energy / detritus_packet_energy).floor()
            as usize)
            .min(producer_positions.len().max(1))
            .min(8);
        // Research note: real aquatic founder webs begin with a detrital /
        // dissolved-organic background rather than perfectly clean water (Azam
        // et al., 1983), so the bootstrap seeds a modest labile detritus pool
        // around producers.
        for packet in 0..detritus_packets {
            if remaining_detritus_energy < detritus_packet_energy {
                break;
            }
            let anchor = producer_positions
                .get(packet % producer_positions.len().max(1))
                .cloned()
                .unwrap_or(Position {
                    x: tw * 0.5,
                    y: self.rooted_settlement_y(),
                });
            let x = (anchor.x + self.rng.random_range(-1.6..1.6)).clamp(2.0, tw - 2.0);
            let y = (self.rooted_settlement_y() + self.rng.random_range(-1.0..0.4))
                .clamp(th - 4.0, th - 2.0);
            self.spawn_detritus(x, y, detritus_packet_energy);
            remaining_detritus_energy -= detritus_packet_energy;
        }
        let min_founders = self.calibration.ecology.min_consumer_founders;
        let max_founders = self
            .calibration
            .ecology
            .max_consumer_founders
            .max(min_founders);
        let mut seeded_consumers = Vec::with_capacity(max_founders);
        let mut total_consumer_biomass = 0.0;
        while seeded_consumers.len() < max_founders
            && (seeded_consumers.len() < min_founders
                || total_consumer_biomass < target_consumer_biomass)
        {
            let i = seeded_consumers.len();
            let genome = self.founder_consumer_genome_for_cohort(i);
            let physics = derive_physics(&genome);
            let anchor = producer_positions
                .get(i % producer_positions.len().max(1))
                .cloned()
                .unwrap_or(Position {
                    x: tw * 0.5,
                    y: self.rooted_settlement_y(),
                });
            let x = (anchor.x + self.rng.random_range(-1.0..1.0)).clamp(3.0, tw - 3.0);
            let (band_min, band_max) = match i % 3 {
                0 => (th * 0.58, th * 0.76),
                1 => (th * 0.72, th * 0.88),
                _ => (th * 0.28, th * 0.56),
            };
            let y = self
                .rng
                .random_range(band_min..band_max)
                .clamp(3.0, th - 3.0);
            let entity = self.spawn_from_genome(genome, x, y);
            self.initialize_founder_consumer(entity, i, max_founders);
            seeded_consumers.push(entity);
            total_consumer_biomass += physics.body_mass;
        }

        self.reset_runtime_counters();
        seeded_consumers
    }

    pub fn bootstrap_founder_web(&mut self) -> Vec<hecs::Entity> {
        self.bootstrap_ecosystem()
    }
}
