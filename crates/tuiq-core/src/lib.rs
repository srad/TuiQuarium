pub mod animation;
pub mod behavior;
pub mod boids;
pub mod brain;
pub mod components;
pub mod ecosystem;
pub mod environment;
pub mod genetics;
pub mod genome;
pub mod needs;
pub mod phenotype;
pub mod physics;
pub mod plant_lifecycle;
pub mod spatial;
pub mod spawner;
pub mod pheromone;

use behavior::BehaviorState;
use boids::{Boid, BoidParams};
use brain::{Brain, InnovationTracker};
use components::*;
use ecosystem::{Age, Detritus, Energy, Producer};
use environment::Environment;
use genetics::genomic_distance;
use rand::RngExt;
use genome::CreatureGenome;
use needs::{NeedWeights, Needs};
use phenotype::{derive_feeding, derive_physics, DerivedPhysics, FeedingCapability};
use spatial::SpatialGrid;

use std::collections::HashMap;
use hecs::Entity;

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

/// Statistics about the current simulation state.
#[derive(Debug, Clone, Default)]
pub struct SimStats {
    pub entity_count: usize,
    pub creature_count: usize,
    pub tick_count: u64,
    pub births: u64,
    pub deaths: u64,
    /// Elapsed in-game days (derived from environment time progression).
    pub elapsed_days: u64,
    /// Maximum generation number among living creatures.
    pub max_generation: u32,
    /// Average complexity of living creatures.
    pub avg_complexity: f32,
    /// Estimated number of species (clusters by genomic distance).
    pub species_count: usize,
}

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

    /// Tank dimensions in cells (width, height) — interior area.
    fn tank_size(&self) -> (u16, u16);
}

/// The main aquarium simulation.
pub struct AquariumSim {
    world: hecs::World,
    env: Environment,
    tick_count: u64,
    total_births: u64,
    total_deaths: u64,
    elapsed_days: u64,
    prev_time_of_day: f32,
    plant_seed_timer: f32,
    tank_width: u16,
    tank_height: u16,
    grid: SpatialGrid,
    boid_params: BoidParams,
    rng: rand::rngs::ThreadRng,
    innovation_tracker: InnovationTracker,
    pheromone_grid: pheromone::PheromoneGrid,
    /// Entities born since last drain — render layer regenerates their art.
    pending_births: Vec<hecs::Entity>,
    // Cached stats (recomputed every 20 ticks, not every frame)
    cached_max_generation: u32,
    cached_avg_complexity: f32,
    cached_species_count: usize,
    cached_creature_count: usize,
    stats_cache_tick: u64,
}

impl AquariumSim {
    pub fn new(tank_width: u16, tank_height: u16) -> Self {
        Self {
            world: hecs::World::new(),
            env: Environment::default(),
            tick_count: 0,
            total_births: 0,
            total_deaths: 0,
            elapsed_days: 0,
            prev_time_of_day: 8.0,
            plant_seed_timer: 0.0,
            tank_width,
            tank_height,
            grid: SpatialGrid::new(6.0),
            boid_params: BoidParams::default(),
            rng: rand::rng(),
            innovation_tracker: InnovationTracker::new(),
            pheromone_grid: pheromone::PheromoneGrid::new(tank_width as f32, tank_height as f32),
            pending_births: Vec::new(),
            cached_max_generation: 0,
            cached_avg_complexity: 0.0,
            cached_species_count: 0,
            cached_creature_count: 0,
            stats_cache_tick: 0,
        }
    }

    /// Mutable access to the ECS world for spawning entities from outside.
    pub fn world_mut(&mut self) -> &mut hecs::World {
        &mut self.world
    }

    /// Drain newly born entities so the render layer can regenerate their art.
    pub fn drain_births(&mut self) -> Vec<hecs::Entity> {
        std::mem::take(&mut self.pending_births)
    }

    /// Spawn a creature with only rendering components (no ecosystem participation).
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
    pub fn spawn_from_genome(
        &mut self,
        genome: CreatureGenome,
        x: f32,
        y: f32,
    ) -> hecs::Entity {
        let physics = derive_physics(&genome);
        let feeding = derive_feeding(&genome, &physics);
        let max_ticks = (8000.0 * genome.behavior.max_lifespan_factor) as u64;

        let vx: f32 = self.rng.random_range(-1.0..1.0);
        let vy: f32 = self.rng.random_range(-0.5..0.5);

        // Placeholder frame — render crate generates the real art
        let frame = AsciiFrame::from_rows(vec!["o"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: if vx >= 0.0 { Direction::Right } else { Direction::Left },
            color_index: genome.art.color_index(),
        };

        let bbox_w = 1.0_f32.max(genome.art.body_size * 3.0);
        let bbox_h = 1.0_f32.max(genome.art.body_size * 2.0);

        let brain = Brain::from_genome_with_learning(&genome.brain, genome.behavior.learning_rate);
        self.world.spawn((
            Position { x, y },
            Velocity { vx, vy },
            BoundingBox { w: bbox_w, h: bbox_h },
            appearance,
            AnimationState::new(0.2 / genome.anim.swim_speed),
            genome,
            physics.clone(),
            feeding,
            Energy::new(physics.max_energy),
            Age { ticks: 0, max_ticks },
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ))
    }

    /// Spawn a plant entity from a PlantGenome.
    /// The genome determines the plant's physics, appearance, and lifespan.
    pub fn spawn_plant(
        &mut self,
        pos: Position,
        genome: genome::PlantGenome,
    ) -> hecs::Entity {
        let stage = PlantStage::Seedling;
        let (appearance, bbox) = plant_lifecycle::build_appearance_from_genome(&genome, stage);

        let physics = phenotype::derive_plant_physics(&genome);

        let feeding = FeedingCapability {
            max_prey_mass: 0.0,
            hunt_skill: 0.0,
            graze_skill: 0.0,
            is_producer: true,
        };

        // Plant lifespan scales with genome's lifespan_factor
        // Base: ~2-4 in-game days; factor stretches or compresses
        let base_ticks = 12000 + (self.rng.random_range(0..12000) as u64);
        let lifespan_ticks = (base_ticks as f32 * genome.lifespan_factor) as u64;

        let initial_energy = if genome.generation == 0 {
            // Stagger Gen-0 starting energy (50-100%) so plants don't all sync
            let frac = 0.5 + self.rng.random_range(0..50) as f32 / 100.0;
            physics.max_energy * frac
        } else {
            // Seeds start with energy proportional to parent's seed_size gene
            // Minimum 35% to avoid immediate wilting (threshold is 20%)
            let seed_frac = (0.3 * genome.seed_size.min(1.5)).max(0.35);
            physics.max_energy * seed_frac
        };

        // Stagger Gen-0 starting age so plants don't all hit stage thresholds together
        let initial_age = if genome.generation == 0 {
            self.rng.random_range(0..200) as f32 / 10.0 // 0-20 seconds
        } else {
            0.0
        };

        self.world.spawn((
            pos,
            bbox,
            appearance,
            AnimationState::new(0.8),
            Producer,
            Energy::new_with(initial_energy, physics.max_energy),
            physics,
            feeding,
            genome,
            stage,
            PlantAge { seconds: initial_age },
            Age { ticks: 0, max_ticks: lifespan_ticks },
        ))
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
    fn spawn_detritus(&mut self, x: f32, y: f32, energy: f32) {
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
            BoundingBox { w: 1.0, h: 1.0 },
            appearance,
            AnimationState::new(2.0),
            Producer,
            Detritus,
            Energy { current: energy, max: energy },
            physics,
            feeding,
        ));
    }

    /// Recompute cached stats (creature count, generation, complexity, species).
    /// Called periodically instead of every frame.
    fn recompute_cached_stats(&mut self) {
        let mut creature_count = 0;
        let mut max_generation: u32 = 0;
        let mut total_complexity: f32 = 0.0;

        for genome in &mut self.world.query::<&CreatureGenome>() {
            creature_count += 1;
            max_generation = max_generation.max(genome.generation);
            total_complexity += genome.complexity;
        }

        self.cached_creature_count = creature_count;
        self.cached_max_generation = max_generation;
        self.cached_avg_complexity = if creature_count > 0 {
            total_complexity / creature_count as f32
        } else {
            0.0
        };

        // Estimate species without cloning — sample up to 100 genomes as centroids
        let threshold = 3.0;
        let mut centroid_indices: Vec<usize> = Vec::new();
        let mut sampled: Vec<CreatureGenome> = Vec::new();

        // Collect a sample (all if <200, else first 200)
        let mut count = 0;
        for genome in &mut self.world.query::<&CreatureGenome>() {
            if count >= 200 { break; }
            sampled.push((*genome).clone());
            count += 1;
        }

        for (i, g) in sampled.iter().enumerate() {
            let fits = centroid_indices.iter().any(|&ci| {
                genomic_distance(&sampled[ci], g) < threshold
            });
            if !fits {
                centroid_indices.push(i);
            }
        }

        self.cached_species_count = centroid_indices.len();
        self.stats_cache_tick = self.tick_count;
    }
}

impl Simulation for AquariumSim {
    fn tick(&mut self, dt: f32) {
        let tw = self.tank_width as f32;
        let th = self.tank_height as f32;

        // 0. Advance environment
        self.env.tick(dt, &mut self.rng);
        if self.env.time_of_day < self.prev_time_of_day {
            self.elapsed_days += 1;
        }
        self.prev_time_of_day = self.env.time_of_day;

        // 1. Rebuild spatial grid
        self.grid.rebuild(&self.world);

        // 2. Build shared entity info map (used by brain, boids, hunting)
        let entity_map: EntityInfoMap = {
            let mut m = HashMap::with_capacity(self.world.len() as usize);
            for (entity, pos, vel, physics) in
                &mut self.world.query::<(Entity, &Position, &Velocity, &DerivedPhysics)>()
            {
                let is_producer = self.world.get::<&Producer>(entity).is_ok();
                let is_boid = self.world.get::<&Boid>(entity).is_ok();
                let (max_prey_mass, hunt_skill, graze_skill) = self.world
                    .get::<&FeedingCapability>(entity)
                    .map(|f| (f.max_prey_mass, f.hunt_skill, f.graze_skill))
                    .unwrap_or((0.0, 0.0, 0.0));
                m.insert(entity, EntityInfo {
                    x: pos.x, y: pos.y,
                    vx: vel.vx, vy: vel.vy,
                    body_mass: physics.body_mass,
                    max_speed: physics.max_speed,
                    is_producer,
                    is_boid,
                    max_prey_mass,
                    hunt_skill,
                    graze_skill,
                });
            }
            m
        };

        // 3. Update needs
        for (needs, weights) in self.world.query_mut::<(&mut needs::Needs, &needs::NeedWeights)>() {
            needs::needs_tick(needs, weights, dt);
        }

        // 4. Brain system (returns pheromone deposits)
        let pheromone_deposits = brain::brain_system(&mut self.world, &self.grid, &entity_map, &self.env, &self.pheromone_grid, dt, tw, th);
        for (x, y, amount) in pheromone_deposits {
            self.pheromone_grid.deposit(x, y, amount);
        }
        self.pheromone_grid.tick();

        // 5. Boids flocking
        boids::boids_system(&mut self.world, &self.grid, &entity_map, &self.boid_params, tw, th, dt);

        // 6. Physics integration + boundary bounce
        physics::physics_system(&mut self.world, dt, tw, th);

        // 7. Ecosystem: metabolism, aging
        ecosystem::metabolism_system(&mut self.world, dt, &self.env, th);
        ecosystem::age_system(&mut self.world);

        // 7b. Plant lifecycle: age plants, update growth stages and appearance
        plant_lifecycle::plant_lifecycle_system(&mut self.world, dt);

        // 8. Hunting + feeding
        let kills = ecosystem::hunting_check(&self.world, &self.grid, &entity_map);
        let _hunted_dead = ecosystem::apply_kills(&mut self.world, &kills);

        // 8. Reproduction (with population cap)
        let creature_count = self.cached_creature_count;
        self.innovation_tracker.new_generation();
        let births = spawner::reproduction_system(&mut self.world, &self.grid, &mut self.rng, tw, th, creature_count, &mut self.innovation_tracker);
        self.total_births += births.len() as u64;
        self.pending_births.extend(&births);

        // 9. Death cleanup + nutrient cycling (dead creatures → detritus)
        let death_result = ecosystem::death_system(&mut self.world);
        self.total_deaths += death_result.creature_deaths;

        // Spawn detritus at dead creature locations (50% of max energy recycled)
        for (x, y, max_e) in &death_result.dead_creature_info {
            let detritus_energy = max_e * 0.5;
            if detritus_energy > 1.0 {
                self.spawn_detritus(*x, *y, detritus_energy);
            }
        }

        // 10. Plant reproduction — only flowering plants can seed
        self.plant_seed_timer += dt;
        let seed_interval = 5.0; // seconds between seeding attempts
        if self.plant_seed_timer >= seed_interval {
            self.plant_seed_timer -= seed_interval;

            // Count existing plants and collect flowering seed candidates
            let mut plant_count = 0;
            let mut seed_candidates: Vec<(f32, f32, genome::PlantGenome)> = Vec::new();
            for (pos, plant_genome, stage) in
                &mut self.world.query::<(&Position, &genome::PlantGenome, &PlantStage)>()
            {
                plant_count += 1;
                if *stage == PlantStage::Flowering {
                    seed_candidates.push((pos.x, pos.y, plant_genome.clone()));
                }
            }

            // Scale plant cap with tank area (~1 per 100 cells, min 15)
            let max_plants = ((tw * th / 100.0) as usize).max(15);
            if plant_count < max_plants && !seed_candidates.is_empty() {
                // 30% chance to seed per interval — prevents all plants seeding simultaneously
                if self.rng.random_range(0..100) < 30 {
                    let idx = self.rng.random_range(0..seed_candidates.len());
                    let (px, py, ref parent_genome) = seed_candidates[idx];

                    // Number of seeds this parent produces (genome-driven)
                    let num_seeds = (parent_genome.seed_count).round().max(1.0) as usize;
                    let seed_range = 4.0 + parent_genome.seed_range * 8.0;

                    for _ in 0..num_seeds.min(3) {
                        if plant_count >= max_plants { break; }

                        // Mutate parent genome to create offspring
                        let mut child_genome = parent_genome.clone();
                        let mutation_rate = parent_genome.mutation_rate_factor * 0.15;
                        genetics::mutate_plant(&mut child_genome, mutation_rate, &mut self.rng);

                        let offset_x = self.rng.random_range(-seed_range..seed_range);
                        let offset_y = self.rng.random_range((-seed_range * 0.5)..(seed_range * 0.5));
                        let new_x = (px + offset_x).clamp(2.0, tw - 2.0);
                        let new_y = (py + offset_y).clamp(th * 0.3, th - 3.0);

                        self.spawn_plant(
                            Position { x: new_x, y: new_y },
                            child_genome,
                        );
                        plant_count += 1;
                    }
                }
            }
        }

        // 11. Animation
        animation::animation_system(&mut self.world, dt);

        self.tick_count += 1;

        // 12. Periodically recompute cached stats (every 20 ticks, not every frame)
        if self.tick_count % 20 == 0 || self.tick_count == 1 {
            self.recompute_cached_stats();
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
            births: self.total_births,
            deaths: self.total_deaths,
            elapsed_days: self.elapsed_days,
            max_generation: self.cached_max_generation,
            avg_complexity: self.cached_avg_complexity,
            species_count: self.cached_species_count,
        }
    }

    fn tank_size(&self) -> (u16, u16) {
        (self.tank_width, self.tank_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut rng = rand::rng();
        let genome = CreatureGenome::minimal_cell(&mut rng);
        sim.spawn_from_genome(genome, x, y);
    }

    /// Helper: spawn a seaweed plant for integration tests.
    fn spawn_test_plant(sim: &mut AquariumSim, x: f32, y: f32) {
        let mut rng = rand::rng();
        let genome = genome::PlantGenome::minimal_plant(&mut rng);
        sim.spawn_plant(
            Position { x, y },
            genome,
        );
    }

    #[test]
    fn test_creatures_can_eat_food() {
        let mut sim = AquariumSim::new(40, 20);

        spawn_test_creature(&mut sim, 10.0, 10.0);
        spawn_test_plant(&mut sim, 12.0, 10.0);

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
        assert!(creature_alive, "Creature should still be alive after {} ticks (deaths={}, entity_count={})",
            stats.tick_count, stats.deaths, stats.entity_count);
    }

    #[test]
    fn test_ecosystem_survives_2000_ticks() {
        let mut sim = AquariumSim::new(60, 20);

        for i in 0..5 {
            spawn_test_creature(&mut sim, 10.0 + i as f32 * 8.0, 8.0);
        }
        for i in 0..6 {
            spawn_test_plant(&mut sim, 5.0 + i as f32 * 8.0, 15.0);
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
             Births={}, Deaths={}, Entities={}",
            stats.births, stats.deaths, stats.entity_count,
        );
    }

    #[test]
    fn test_plants_seed_new_plants() {
        let mut sim = AquariumSim::new(40, 20);

        // Start with one high-energy plant near surface
        spawn_test_plant(&mut sim, 20.0, 5.0);

        // Override to high energy and mature age so it reaches Flowering stage
        for energy in sim.world.query_mut::<&mut Energy>() {
            energy.current = energy.max * 0.95;
        }
        for age in sim.world.query_mut::<&mut PlantAge>() {
            age.seconds = 50.0; // past Young threshold (40s)
        }

        assert_eq!(sim.stats().entity_count, 1);

        // Run enough ticks for seeding (5s interval × 30% probability)
        // ~50 intervals over 250s = very high chance of at least one seed
        for _ in 0..5000 {
            sim.tick(0.05);
        }

        assert!(
            sim.stats().entity_count > 1,
            "Flowering plant should seed new plants. Entities={}",
            sim.stats().entity_count,
        );
    }

    #[test]
    fn test_plants_regenerate_energy() {
        let mut sim = AquariumSim::new(40, 20);
        spawn_test_plant(&mut sim, 10.0, 3.0); // near surface for better light

        // Set energy to mid-range (above unstable equilibrium ~19%)
        for energy in sim.world.query_mut::<&mut Energy>() {
            energy.current = energy.max * 0.4;
        }

        let initial: f32 = sim.world.query_mut::<&Energy>()
            .into_iter().next().unwrap().current;

        for _ in 0..400 {
            sim.tick(0.05);
        }

        for energy in &mut sim.world.query::<&Energy>() {
            assert!(
                energy.current > initial,
                "Plant should regenerate energy via photosynthesis, got {} (started at {})",
                energy.current, initial
            );
            break;
        }
    }

    #[test]
    fn test_spawn_from_genome() {
        let mut sim = AquariumSim::new(80, 24);
        let mut rng = rand::rng();
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
    fn test_cell_survives_long_enough_to_reproduce() {
        // A minimal cell surrounded by food should live long enough for
        // its reproduction need to reach threshold and produce offspring.
        let mut sim = AquariumSim::new(30, 15);

        // One cell with plenty of nearby food
        spawn_test_creature(&mut sim, 15.0, 7.0);
        for i in 0..6 {
            spawn_test_plant(&mut sim, 10.0 + i as f32 * 3.0, 7.0);
        }

        let dt = 0.05;
        // Run for 5000 ticks = 250 sim-seconds
        for _ in 0..5000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.births > 0,
            "Cell with ample food should reproduce at least once! \
             Creatures={}, Deaths={}, Entities={}",
            stats.creature_count, stats.deaths, stats.entity_count,
        );
    }

    #[test]
    fn test_cell_energy_math_is_viable() {
        // Unit test: verify a minimal cell's max_energy is large enough
        // relative to its metabolism that it can survive a reasonable time.
        let mut rng = rand::rng();
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
                survive_secs, p.max_energy, p.base_metabolism, drain_per_sec,
            );
        }
    }

    #[test]
    fn test_population_grows_with_food() {
        // Start with 10 cells and abundant plants.
        // Population should grow (births > deaths) within 5000 ticks.
        let mut sim = AquariumSim::new(60, 20);
        let tw = 60.0_f32;
        let th = 20.0_f32;

        for i in 0..10 {
            let x = 5.0 + (i as f32 / 10.0) * (tw - 10.0);
            let y = th * 0.4 + (i as f32 % 3.0) * 2.0;
            spawn_test_creature(&mut sim, x, y);
        }

        // Dense plant coverage
        for i in 0..12 {
            let x = 3.0 + (i as f32 / 12.0) * (tw - 6.0);
            let y = th * 0.3 + (i as f32 % 4.0) * 3.0;
            spawn_test_plant(&mut sim, x, y);
        }

        let dt = 0.05;
        for _ in 0..5000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.births >= 5,
            "Population should have at least 5 births with abundant food! \
             Births={}, Deaths={}, Creatures={}, Entities={}",
            stats.births, stats.deaths, stats.creature_count, stats.entity_count,
        );
    }

    #[test]
    fn test_ecosystem_survives_10000_ticks() {
        // Long-running stability test: 20 cells + plants should sustain
        // a population over 10000 ticks (500 sim-seconds).
        let mut sim = AquariumSim::new(80, 24);
        let tw = 80.0_f32;
        let th = 24.0_f32;

        for i in 0..20 {
            let x = 3.0 + (i as f32 / 20.0) * (tw - 6.0);
            let y = 3.0 + (i as f32 % 5.0) * 3.0;
            spawn_test_creature(&mut sim, x, y);
        }
        for i in 0..10 {
            let x = 3.0 + (i as f32 / 10.0) * (tw - 6.0);
            let y = th * 0.5 + (i as f32 % 3.0) * 2.0;
            spawn_test_plant(&mut sim, x, y);
        }

        let dt = 0.05;
        for _ in 0..10000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.creature_count > 0,
            "Population should survive 10000 ticks! \
             Births={}, Deaths={}, Creatures={}, Entities={}",
            stats.births, stats.deaths, stats.creature_count, stats.entity_count,
        );
        assert!(
            stats.births > 0,
            "There should be at least some reproduction in 10000 ticks! \
             Births={}, Deaths={}, Creatures={}",
            stats.births, stats.deaths, stats.creature_count,
        );
    }

    #[test]
    fn test_stats_include_generation_and_complexity() {
        let mut sim = AquariumSim::new(80, 24);
        let mut rng = rand::rng();

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

    // ── Hypothesis tests ──────────────────────────────────────────
    // These tests verify the evolutionary dynamics fixes actually work.

    /// Hypothesis test: grazing drains plant energy without killing the plant.
    /// Plants should survive being eaten and regenerate via photosynthesis.
    #[test]
    fn test_grazing_drains_but_preserves_plants() {
        let mut sim = AquariumSim::new(30, 15);

        // Place a creature right next to a plant so it can graze
        spawn_test_creature(&mut sim, 11.0, 10.0);
        spawn_test_plant(&mut sim, 12.0, 10.0);

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

        // Plant should still exist (not destroyed)
        let mut plant_count = 0;
        for _p in &mut sim.world.query::<&Producer>() {
            plant_count += 1;
        }
        assert!(
            plant_count >= 1,
            "Plant should survive grazing (partial energy drain, not destruction)"
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
            spawn_test_plant(&mut sim, x, y);
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
            plant_count, stats.creature_count,
        );
        assert!(
            stats.creature_count > 0,
            "Creatures should survive on plant-based ecosystem! \
             Births={}, Deaths={}, Plants={}",
            stats.births, stats.deaths, plant_count,
        );
    }

    /// Hypothesis test: sexual reproduction works across multiple generations.
    /// Starting from identical clones, creatures should be able to find
    /// compatible mates and reproduce sexually, reaching generation > 5.
    #[test]
    fn test_sexual_reproduction_across_generations() {
        let mut sim = AquariumSim::new(40, 15);
        let mut rng = rand::rng();

        // Spawn 15 creatures with complexity 0.4 (sexual reproduction range)
        // All clones of the same genome so they start compatible
        let mut base = CreatureGenome::minimal_cell(&mut rng);
        base.complexity = 0.4;

        for i in 0..15 {
            let mut g = base.clone();
            // Small variation so they're not exact clones
            g.art.body_elongation += (i as f32) * 0.01;
            let x = 3.0 + (i as f32 / 15.0) * 34.0;
            let y = 3.0 + (i as f32 % 3.0) * 3.0;
            sim.spawn_from_genome(g, x, y);
        }

        // Add food
        for i in 0..8 {
            spawn_test_plant(&mut sim, 3.0 + i as f32 * 4.0, 10.0);
        }

        let dt = 0.05;
        for _ in 0..8000 {
            sim.tick(dt);
        }

        let stats = sim.stats();
        assert!(
            stats.max_generation >= 3,
            "Sexual reproduction should produce multiple generations. \
             Max gen={}, Births={}, Deaths={}, Creatures={}",
            stats.max_generation, stats.births, stats.deaths, stats.creature_count,
        );
    }

    /// Integration hypothesis test: average complexity rises in a long simulation.
    /// Starting from minimal cells (complexity ~0.05), the average complexity
    /// should increase over 10000 ticks due to the mutation step fix and the
    /// sensory/metabolism advantages of higher complexity.
    #[test]
    fn test_complexity_rises_in_simulation() {
        let mut sim = AquariumSim::new(60, 20);
        let tw = 60.0_f32;
        let th = 20.0_f32;

        // Spawn 20 minimal cells
        for i in 0..20 {
            let x = 3.0 + (i as f32 / 20.0) * (tw - 6.0);
            let y = 3.0 + (i as f32 % 5.0) * 3.0;
            spawn_test_creature(&mut sim, x, y);
        }

        // Dense plant coverage
        for i in 0..10 {
            let x = 3.0 + (i as f32 / 10.0) * (tw - 6.0);
            let y = th * 0.5 + (i as f32 % 3.0) * 2.0;
            spawn_test_plant(&mut sim, x, y);
        }

        // Record initial complexity
        let initial_stats = sim.stats();
        // Minimal cells start at 0.0–0.1
        assert!(
            initial_stats.avg_complexity <= 0.15,
            "Initial complexity should be low, got {:.3}",
            initial_stats.avg_complexity,
        );

        let dt = 0.05;
        for _ in 0..10000 {
            sim.tick(dt);
        }

        let final_stats = sim.stats();

        // Population should still exist
        assert!(
            final_stats.creature_count > 0,
            "Population must survive! Births={}, Deaths={}",
            final_stats.births, final_stats.deaths,
        );

        // Key assertion: avg complexity should have increased
        assert!(
            final_stats.avg_complexity > initial_stats.avg_complexity,
            "Average complexity should increase over 10000 ticks. \
             Initial={:.3}, Final={:.3}, Gen={}, Births={}, Creatures={}",
            initial_stats.avg_complexity, final_stats.avg_complexity,
            final_stats.max_generation, final_stats.births, final_stats.creature_count,
        );
    }

    #[test]
    fn test_detritus_spawns_on_creature_death() {
        let mut sim = AquariumSim::new(100, 50);
        let mut rng = rand::rng();
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
        assert!(detritus_count > 0, "Detritus should spawn when creature dies");
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
        assert!(sim.stats().tick_count >= 100, "Simulation with pheromone system runs without panicking");
    }

    #[test]
    fn test_nutrient_cycle_detritus_is_edible() {
        let mut sim = AquariumSim::new(40, 20);
        let mut rng = rand::rng();
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
        assert!(detritus_is_producer, "Detritus should be a Producer (grazeable)");

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
            initial_detritus_energy, final_detritus_energy,
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
        let mut rng = rand::rng();

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

        // Provide food: dense plant coverage so survival depends on foraging ability
        for i in 0..15 {
            let x = 3.0 + (i as f32 / 15.0) * 74.0;
            let y = 10.0 + (i as f32 % 3.0) * 5.0;
            spawn_test_plant(&mut sim, x, y);
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

    /// Diagnostic test: verify that complexity naturally rises through mutation
    /// when starting from minimal_cell creatures. This is the core evolution test.
    #[test]
    fn test_complexity_rises_naturally() {
        let mut sim = AquariumSim::new(80, 30);
        let mut rng = rand::rng();

        // Spawn initial population (matching main.rs setup)
        for i in 0..15 {
            let genome = CreatureGenome::minimal_cell(&mut rng);
            let x = 5.0 + (i as f32) * 5.0;
            let y = 5.0 + (i as f32 % 5.0) * 4.0;
            sim.spawn_from_genome(genome, x, y);
        }

        // Spawn plants for food
        for i in 0..20 {
            let x = 3.0 + (i as f32 / 20.0) * 74.0;
            let y = 10.0 + (i as f32 % 4.0) * 5.0;
            spawn_test_plant(&mut sim, x, y);
        }

        // Run for 5 simulated days (30000 ticks)
        let mut max_complexity_seen = 0.0_f32;
        let mut total_births = 0u64;
        let checkpoint_ticks = [6000, 12000, 18000, 24000, 30000];

        for tick in 1..=30000 {
            sim.tick(0.05);

            if checkpoint_ticks.contains(&tick) {
                let stats = sim.stats();
                let day = tick as f32 / 6000.0;

                // Scan all creatures for complexity
                let mut complexities: Vec<f32> = Vec::new();
                for genome in &mut sim.world.query::<&CreatureGenome>() {
                    complexities.push(genome.complexity);
                }

                let max_c = complexities.iter().copied().fold(0.0_f32, f32::max);
                let avg_c = if complexities.is_empty() { 0.0 }
                    else { complexities.iter().sum::<f32>() / complexities.len() as f32 };
                max_complexity_seen = max_complexity_seen.max(max_c);
                total_births = stats.births;

                eprintln!(
                    "Day {:.0}: pop={}, births={}, gen={}, avg_c={:.3}, max_c={:.3}",
                    day, complexities.len(), stats.births, stats.max_generation, avg_c, max_c,
                );
            }
        }

        // After 5 sim days, complexity should have risen above initial 0.0-0.1
        assert!(
            total_births > 0,
            "No births occurred in 30000 ticks — creatures cannot reproduce!"
        );
        assert!(
            max_complexity_seen > 0.15,
            "Max complexity should exceed 0.15 after 5 sim days (saw {:.3}). \
             Births: {}. Complexity mutations aren't driving evolution.",
            max_complexity_seen, total_births,
        );
    }

    #[test]
    fn test_plant_full_lifecycle_to_death() {
        // Spawn a plant with a very short lifespan and verify it gets removed by death_system
        let mut sim = AquariumSim::new(40, 20);
        let mut rng = rand::rng();
        let mut genome = genome::PlantGenome::minimal_plant(&mut rng);
        genome.lifespan_factor = 0.01; // very short lifespan

        sim.spawn_plant(Position { x: 20.0, y: 15.0 }, genome);
        assert_eq!(sim.stats().entity_count, 1);

        // Tick long enough for age to exceed max_ticks
        // With lifespan_factor=0.01, max_ticks ≈ 12000*0.01 = 120 ticks
        for _ in 0..500 {
            sim.tick(0.05);
        }

        // Plant should have died by age
        let plant_count: usize = {
            let mut count = 0;
            for (_, g) in &mut sim.world.query::<(hecs::Entity, &genome::PlantGenome)>() {
                if g.generation == 0 {
                    count += 1;
                }
            }
            count
        };

        // The original gen-0 plant should be dead (it may have seeded before dying)
        assert_eq!(
            plant_count, 0,
            "Gen-0 plant with very short lifespan should die by age"
        );
    }

    #[test]
    fn test_gen0_plants_have_staggered_stages() {
        let mut sim = AquariumSim::new(80, 24);

        // Spawn 10 plants
        for i in 0..10 {
            let mut rng = rand::rng();
            let genome = genome::PlantGenome::minimal_plant(&mut rng);
            sim.spawn_plant(Position { x: 5.0 + i as f32 * 7.0, y: 18.0 }, genome);
        }

        // Tick enough so starting age spread (0-20s) straddles Young→Mature boundary (40s)
        // After 30s sim time: plants at ages 30-50s → some Young (<40s), some Mature (≥40s)
        for _ in 0..600 {
            sim.tick(0.05); // 30 sim seconds
        }

        // Collect stages
        let stages: Vec<PlantStage> = {
            let mut v = Vec::new();
            for s in sim.world.query_mut::<&PlantStage>() {
                v.push(*s);
            }
            v
        };

        // With staggered starting age (0-20s) and energy (50-100%),
        // not all plants should be in the same stage
        let all_same = stages.windows(2).all(|w| w[0] == w[1]);
        assert!(
            !all_same || stages.len() <= 1,
            "Gen-0 plants should have staggered stages, but all {} are {:?}",
            stages.len(), stages.first()
        );
    }

    #[test]
    fn test_flowering_plant_produces_offspring() {
        let mut sim = AquariumSim::new(60, 20);

        // Spawn one plant, manually set it to flowering with high energy
        let mut rng = rand::rng();
        let mut genome = genome::PlantGenome::minimal_plant(&mut rng);
        genome.seed_count = 3.0;
        genome.seed_range = 1.0;

        let entity = sim.spawn_plant(Position { x: 30.0, y: 15.0 }, genome);

        // Set to flowering stage + high energy + past young age
        if let Ok(mut stage) = sim.world.get::<&mut PlantStage>(entity) {
            *stage = PlantStage::Flowering;
        }
        if let Ok(mut energy) = sim.world.get::<&mut ecosystem::Energy>(entity) {
            energy.current = energy.max * 0.95;
        }
        if let Ok(mut age) = sim.world.get::<&mut PlantAge>(entity) {
            age.seconds = 50.0;
        }

        let initial_count = sim.stats().entity_count;

        // Run enough ticks for seeding (5s interval × 30% probability → ~17s on average)
        // Run 50 seeding intervals to almost guarantee at least one seeds
        for _ in 0..5000 {
            sim.tick(0.05);
        }

        let final_count = sim.stats().entity_count;
        assert!(
            final_count > initial_count,
            "Flowering plant should produce offspring. Initial={}, Final={}",
            initial_count, final_count
        );
    }
}
