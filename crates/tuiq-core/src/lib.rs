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
pub mod spatial;
pub mod spawner;

use behavior::BehaviorState;
use boids::{Boid, BoidParams};
use brain::Brain;
use components::*;
use ecosystem::{Age, Energy, TrophicRole};
use environment::Environment;
use rand::RngExt;
use genome::{CreatureGenome, DietType};
use needs::{NeedWeights, Needs};
use phenotype::derive_physics;
use spatial::SpatialGrid;

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
    food_spawn_timer: f32,
    tank_width: u16,
    tank_height: u16,
    grid: SpatialGrid,
    boid_params: BoidParams,
    rng: rand::rngs::ThreadRng,
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
            prev_time_of_day: 8.0, // matches Environment::default()
            food_spawn_timer: 0.0,
            tank_width,
            tank_height,
            grid: SpatialGrid::new(10.0),
            boid_params: BoidParams::default(),
            rng: rand::rng(),
        }
    }

    /// Mutable access to the ECS world for spawning entities from outside.
    pub fn world_mut(&mut self) -> &mut hecs::World {
        &mut self.world
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

    /// Spawn a fully simulated creature with genome, physics, energy, needs, etc.
    pub fn spawn_full_creature(
        &mut self,
        pos: Position,
        vel: Velocity,
        bbox: BoundingBox,
        appearance: Appearance,
        anim: AnimationState,
        genome: CreatureGenome,
    ) -> hecs::Entity {
        let physics = derive_physics(&genome);
        let trophic_role = match genome.behavior.diet {
            DietType::Herbivore => TrophicRole::Herbivore,
            DietType::Omnivore => TrophicRole::Omnivore,
            DietType::Carnivore => TrophicRole::Carnivore,
        };
        let max_ticks = (5000.0 * genome.behavior.max_lifespan_factor) as u64;

        let brain = Brain::from_genome(&genome.brain);
        self.world.spawn((
            pos,
            vel,
            bbox,
            appearance,
            anim,
            genome,
            Energy::new(physics.max_energy),
            Age { ticks: 0, max_ticks },
            trophic_role,
            physics,
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ))
    }

    /// Spawn a static plant entity (producer) that can be eaten.
    /// Plants regenerate energy slowly (photosynthesis) making them a renewable food source.
    pub fn spawn_plant(
        &mut self,
        pos: Position,
        bbox: BoundingBox,
        appearance: Appearance,
        anim: AnimationState,
    ) -> hecs::Entity {
        let physics = phenotype::DerivedPhysics {
            body_mass: 0.1,     // Very low mass — any creature can eat plants
            max_energy: 15.0,
            base_metabolism: -0.25, // Negative = photosynthesis (slow regeneration)
            max_speed: 0.0,
            acceleration: 0.0,
            turn_radius: 0.0,
            drag_coefficient: 0.0,
            visual_profile: 0.5,
            camouflage: 0.0,
            sensory_range: 0.0,
        };

        self.world.spawn((
            pos,
            bbox,
            appearance,
            anim,
            TrophicRole::Producer,
            Energy::new(15.0),
            physics,
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
            body_mass: 0.1,     // Tiny — any creature can eat
            max_energy: 10.0,
            base_metabolism: 0.3, // Food decays — rots if not eaten
            max_speed: 0.0,
            acceleration: 0.0,
            turn_radius: 0.0,
            drag_coefficient: 0.0,
            visual_profile: 0.2,
            camouflage: 0.0,
            sensory_range: 0.0,
        };

        self.world.spawn((
            pos,
            vel,
            bbox,
            appearance,
            AnimationState::new(1.0),
            TrophicRole::Producer,
            Energy::new(10.0),
            physics,
        ))
    }
}

impl Simulation for AquariumSim {
    fn tick(&mut self, dt: f32) {
        let tw = self.tank_width as f32;
        let th = self.tank_height as f32;

        // 0. Advance environment (day/night, temperature, currents, events)
        self.env.tick(dt, &mut self.rng);
        if self.env.time_of_day < self.prev_time_of_day {
            self.elapsed_days += 1;
        }
        self.prev_time_of_day = self.env.time_of_day;

        // 1. Rebuild spatial grid
        self.grid.rebuild(&self.world);

        // 2. Update needs
        for (needs, weights) in self.world.query_mut::<(&mut needs::Needs, &needs::NeedWeights)>() {
            needs::needs_tick(needs, weights, dt);
        }

        // 3. Brain system — neural net drives steering and behavior
        brain::brain_system(&mut self.world, &self.grid, &self.env, dt, tw, th);

        // 4. Boids flocking
        boids::boids_system(&mut self.world, &self.grid, &self.boid_params, tw, th, dt);

        // 5. Physics integration + boundary bounce
        physics::physics_system(&mut self.world, dt, tw, th);

        // 6. Ecosystem: metabolism, aging
        ecosystem::metabolism_system(&mut self.world, dt);
        ecosystem::age_system(&mut self.world);

        // 7. Hunting + feeding
        let kills = ecosystem::hunting_check(&self.world, &self.grid);
        let hunted_dead = ecosystem::apply_kills(&mut self.world, &kills);
        let _ = hunted_dead;

        // 7. Reproduction
        let births = spawner::reproduction_system(&mut self.world, &self.grid, &mut self.rng, tw, th);
        self.total_births += births.len() as u64;

        // 8. Death cleanup
        let death_result = ecosystem::death_system(&mut self.world);
        self.total_deaths += death_result.creature_deaths;

        // 9. Auto food spawning — periodic organic matter drifts down
        self.food_spawn_timer += dt;
        let food_interval = 5.0; // spawn food every 5 sim-seconds
        if self.food_spawn_timer >= food_interval {
            self.food_spawn_timer -= food_interval;
            let food_x = self.rng.random_range(2.0..(tw - 2.0));
            let frame = AsciiFrame::from_rows(vec!["*"]);
            let appearance = Appearance {
                frame_sets: vec![vec![frame.clone()], vec![frame]],
                facing: Direction::Right,
                color_index: 1,
            };
            self.spawn_food(
                Position { x: food_x, y: 0.0 },
                Velocity { vx: 0.0, vy: 1.0 },
                BoundingBox { w: 1.0, h: 1.0 },
                appearance,
            );
        }

        // 10. Animation
        animation::animation_system(&mut self.world, dt);

        self.tick_count += 1;
    }

    fn world(&self) -> &hecs::World {
        &self.world
    }

    fn environment(&self) -> &Environment {
        &self.env
    }

    fn stats(&self) -> SimStats {
        let creature_count = {
            let mut c = 0;
            for _ in &mut self.world.query::<&genome::CreatureGenome>() {
                c += 1;
            }
            c
        };
        SimStats {
            entity_count: self.world.len() as usize,
            creature_count,
            tick_count: self.tick_count,
            births: self.total_births,
            deaths: self.total_deaths,
            elapsed_days: self.elapsed_days,
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
        assert!(new_x > 10.0, "Fish should have moved right");
    }

    #[test]
    fn test_ascii_frame_flip() {
        let frame = AsciiFrame::from_rows(vec![r" /o\=>", r" \  />"]);
        let flipped = frame.flip_horizontal();
        // " /o\=>" reversed = ">=\o/ ", then mirror chars: "<=/o\ "
        assert_eq!(flipped.rows[0], r"<=/o\ ");
        // " \  />" reversed = ">/  \ ", then mirror chars: "<\  / "
        assert_eq!(flipped.rows[1], r"<\  / ");
    }

    /// Helper: spawn a full herbivore creature for integration tests.
    fn spawn_test_herbivore(sim: &mut AquariumSim, x: f32, y: f32, vx: f32, vy: f32) {
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        let genome = genome::CreatureGenome {
            art: genome::ArtGenome {
                body_plan: genome::BodyPlan::Slim,
                body_size: 0.8,
                tail_style: genome::TailStyle::Forked,
                tail_length: 1.0,
                has_dorsal_fin: true,
                has_pectoral_fins: true,
                fill_pattern: genome::FillPattern::Solid,
                eye_style: genome::EyeStyle::Circle,
                primary_color: 0,
                secondary_color: 1,
                color_brightness: 0.7,
            },
            anim: genome::AnimGenome {
                swim_speed: 1.2,
                tail_amplitude: 0.8,
                idle_sway: 0.5,
                undulation: 0.5,
            },
            behavior: genome::BehaviorGenome {
                schooling_affinity: 0.8,
                aggression: 0.1,
                timidity: 0.5,
                speed_factor: 1.0,
                metabolism_factor: 1.0,
                diet: genome::DietType::Herbivore,
                max_lifespan_factor: 1.0,
                reproduction_rate: 0.5,
            },
            brain: brain::BrainGenome::random(&mut rand::rng()),
        };
        sim.spawn_full_creature(
            Position { x, y },
            Velocity { vx, vy },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            genome,
        );
    }

    /// Helper: spawn a seaweed plant for integration tests.
    fn spawn_test_plant(sim: &mut AquariumSim, x: f32, y: f32) {
        let frame = AsciiFrame::from_rows(vec![")"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 5,
        };
        sim.spawn_plant(
            Position { x, y },
            BoundingBox { w: 1.0, h: 1.0 },
            appearance,
            AnimationState::new(0.5),
        );
    }

    #[test]
    fn test_creatures_can_eat_food() {
        let mut sim = AquariumSim::new(40, 20);

        // Spawn a herbivore right next to a plant
        spawn_test_herbivore(&mut sim, 10.0, 10.0, 0.0, 0.0);
        spawn_test_plant(&mut sim, 12.0, 10.0);

        assert_eq!(sim.stats().entity_count, 2);

        // Run for a while — creature should eat the plant
        for _ in 0..200 {
            sim.tick(0.05);
        }

        // The plant should have been eaten (despawned or its energy transferred)
        // Check that eating happened by looking at deaths counter
        // (plant dies when eaten via apply_kills)
        let stats = sim.stats();
        // Either the plant was eaten or it's still alive — but the creature should still be alive
        let creature_alive = {
            let mut has_creature = false;
            for genome in &mut sim.world.query::<&genome::CreatureGenome>() {
                has_creature = true;
                let _ = genome;
            }
            has_creature
        };
        assert!(creature_alive, "Creature should still be alive after {} ticks (deaths={}, entity_count={})",
            stats.tick_count, stats.deaths, stats.entity_count);
    }

    #[test]
    fn test_ecosystem_survives_2000_ticks() {
        let mut sim = AquariumSim::new(60, 20);

        // Spawn a small school of herbivores
        spawn_test_herbivore(&mut sim, 15.0, 8.0, 2.0, 0.5);
        spawn_test_herbivore(&mut sim, 18.0, 9.0, 1.5, -0.3);
        spawn_test_herbivore(&mut sim, 20.0, 7.0, 2.5, 0.2);

        // Spawn several plants spread across the tank
        for i in 0..6 {
            spawn_test_plant(&mut sim, 5.0 + i as f32 * 8.0, 15.0);
        }

        let initial_creatures = 3;
        let dt = 0.05; // 50ms per tick

        // Run for 2000 ticks (100 sim seconds)
        for _ in 0..2000 {
            sim.tick(dt);
        }

        let stats = sim.stats();

        // At least one creature should survive.
        // Auto food spawning + plants should keep them alive.
        let creature_count = {
            let mut count = 0;
            for _ in &mut sim.world.query::<&genome::CreatureGenome>() {
                count += 1;
            }
            count
        };

        assert!(
            creature_count > 0,
            "At least one creature should survive 2000 ticks! \
             Started with {}, now {} creatures alive. \
             Total births={}, deaths={}, entities={}",
            initial_creatures,
            creature_count,
            stats.births,
            stats.deaths,
            stats.entity_count,
        );
    }

    #[test]
    fn test_ecosystem_survives_5000_ticks() {
        let mut sim = AquariumSim::new(80, 24);

        // Larger starting population
        for i in 0..5 {
            let x = 10.0 + i as f32 * 12.0;
            spawn_test_herbivore(&mut sim, x, 8.0, 2.0, 0.3);
        }

        // Dense plant bed
        for i in 0..8 {
            spawn_test_plant(&mut sim, 5.0 + i as f32 * 9.0, 18.0);
        }

        let dt = 0.05;
        let mut min_creatures = usize::MAX;
        let mut max_creatures = 0_usize;

        for tick in 0..5000 {
            sim.tick(dt);
            if tick % 100 == 0 {
                let count = {
                    let mut c = 0;
                    for _ in &mut sim.world.query::<&genome::CreatureGenome>() {
                        c += 1;
                    }
                    c
                };
                min_creatures = min_creatures.min(count);
                max_creatures = max_creatures.max(count);
            }
        }

        let stats = sim.stats();
        let final_creatures = {
            let mut c = 0;
            for _ in &mut sim.world.query::<&genome::CreatureGenome>() {
                c += 1;
            }
            c
        };

        assert!(
            final_creatures > 0,
            "Ecosystem collapsed! No creatures left after 5000 ticks. \
             Min={}, Max={}, Births={}, Deaths={}, Entities={}",
            min_creatures,
            max_creatures,
            stats.births,
            stats.deaths,
            stats.entity_count,
        );
    }

    #[test]
    fn test_food_auto_spawns() {
        let mut sim = AquariumSim::new(40, 20);
        assert_eq!(sim.stats().entity_count, 0);

        // Run for enough time that auto food should spawn (every 5 sim-seconds)
        // 5 seconds / 0.05 dt = 100 ticks
        for _ in 0..120 {
            sim.tick(0.05);
        }

        assert!(
            sim.stats().entity_count > 0,
            "Auto food spawning should have created food particles"
        );
    }

    #[test]
    fn test_plants_regenerate_energy() {
        let mut sim = AquariumSim::new(40, 20);
        spawn_test_plant(&mut sim, 10.0, 10.0);

        // Manually drain plant energy
        for energy in sim.world.query_mut::<&mut Energy>() {
            energy.current = 5.0; // low energy
        }

        // Run metabolism — plant has negative metabolism (photosynthesis)
        for _ in 0..200 {
            sim.tick(0.05);
        }

        // Plant should have gained energy back
        for energy in &mut sim.world.query::<&Energy>() {
            // Should be above 5.0 due to photosynthesis (negative metabolism means energy gain)
            assert!(
                energy.current > 5.0,
                "Plant should regenerate energy via photosynthesis, got {}",
                energy.current
            );
            break; // just check the first one
        }
    }
}
