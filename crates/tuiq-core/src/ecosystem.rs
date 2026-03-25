//! Energy-based ecosystem: metabolism, feeding, hunting, death.
//! Feeding roles are emergent from morphology — no fixed trophic categories.

use std::collections::HashSet;

use hecs::{Entity, World};
use rayon::prelude::*;

use crate::components::Position;
use crate::environment::Environment;
use crate::genome::CreatureGenome;
use crate::needs::Needs;
use crate::phenotype::{DerivedPhysics, FeedingCapability};
use crate::spatial::SpatialGrid;
use crate::EntityInfoMap;

/// Energy state for a living creature.
#[derive(Debug, Clone)]
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

/// Age of a creature in simulation ticks.
#[derive(Debug, Clone)]
pub struct Age {
    pub ticks: u64,
    pub max_ticks: u64,
}

/// Marker component for producer entities (plants, food pellets).
/// Creatures do NOT have this — their feeding behavior is determined by FeedingCapability.
#[derive(Debug, Clone, Copy)]
pub struct Producer;

/// Marker component for detritus entities (dead creature remains).
/// Detritus is a producer that decays faster and can be grazed.
#[derive(Debug, Clone, Copy)]
pub struct Detritus;

/// Drain energy from metabolism each tick.
/// base_metabolism is already computed in DerivedPhysics.
/// For producers (plants): LAI-based photosynthesis (Beer–Lambert law) scaled by
///   depth-dependent light; maintenance cost follows Kleiber's law (mass^0.75).
/// For detritus: decay only (positive metabolism), no photosynthesis.
/// For creatures: metabolism scaled by depth (cooler deep water = slower).
pub fn metabolism_system(world: &mut World, dt: f32, env: &Environment, tank_height: f32) {
    let temp_mod = env.temperature_modifier();
    // Night stress: creatures burn more energy in darkness (thermoregulation, vigilance)
    let night_stress = 1.0 + 0.3 * (1.0 - env.light_level).max(0.0);

    for (energy, physics, pos, is_detritus, plant_genome) in world.query_mut::<(
        &mut Energy,
        &DerivedPhysics,
        Option<&Position>,
        Option<&Detritus>,
        Option<&crate::genome::PlantGenome>,
    )>() {
        let y = pos.map(|p| p.y).unwrap_or(0.0);

        if is_detritus.is_some() {
            // Detritus decays at a fixed rate
            energy.current -= 0.5 * dt * 0.25;
        } else if physics.base_metabolism < 0.0 {
            // Photosynthetic producer: LAI-based photosynthesis
            let fill = energy.fraction();
            let logistic = fill * (1.0 - fill) * 4.0; // peaks at 0.5 fill
            let light = env.light_at_depth(y, tank_height);

            // Photosynthesis gain
            energy.current -= physics.base_metabolism * dt * 0.25 * logistic * light * temp_mod;

            // Maintenance cost: allometric scaling from genome mass
            // Larger plants have higher maintenance (Kleiber's law)
            // Tuned so energy equilibrium is ~80-84% (Mature) under standard light (0.8).
            // At noon peak (light=1.0), strong plants can briefly reach Flowering (≥85%)
            // for seed production. At night (light≈0.05), all plants drain.
            let maintenance = if let Some(pg) = plant_genome {
                let mass = pg.plant_mass();
                0.08 * mass.powf(0.75) * dt
            } else {
                0.025 * dt
            };
            energy.current -= maintenance;
        } else {
            // Creature: metabolism scaled by depth, temperature, and day/night cycle
            let depth_mod = env.metabolism_at_depth(y, tank_height);
            energy.current -= physics.base_metabolism * dt * 0.25 * depth_mod * temp_mod * night_stress;
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
/// - For mobile prey: hunt_skill > 0.3 AND speed/size advantage
pub fn hunting_check(
    world: &World,
    grid: &SpatialGrid,
    entity_map: &EntityInfoMap,
) -> Vec<(Entity, Entity)> {
    // Collect all creatures with feeding capability (potential eaters)
    let eaters: Vec<(Entity, f32, f32, f32, f32, f32, f32, f32)> = {
        let mut v = Vec::new();
        for (entity, pos, physics, feeding) in
            &mut world.query::<(Entity, &Position, &DerivedPhysics, &FeedingCapability)>()
        {
            if !feeding.is_producer {
                v.push((
                    entity, pos.x, pos.y,
                    physics.max_speed, physics.sensory_range,
                    feeding.max_prey_mass, feeding.hunt_skill, feeding.graze_skill,
                ));
            }
        }
        v
    };

    // Compute kills in parallel — each eater finds at most one prey
    eaters.par_iter().filter_map(
        |&(pred_entity, px, py, pred_speed, sense_range, max_prey_mass, hunt_skill, graze_skill)| {
        let neighbors = grid.neighbors(px, py, sense_range);
        for &prey_entity in &neighbors {
            if prey_entity == pred_entity {
                continue;
            }

            let prey = match entity_map.get(&prey_entity) {
                Some(p) => p,
                None => continue,
            };

            if prey.body_mass > max_prey_mass {
                continue;
            }

            if prey.is_producer {
                if graze_skill < 0.2 {
                    continue;
                }
            } else {
                if hunt_skill < 0.3 {
                    continue;
                }
                if pred_speed < prey.max_speed * 0.8 {
                    continue;
                }
            }

            let strike_dist = if prey.is_producer { 5.0 } else { 3.0 };
            let dx = px - prey.x;
            let dy = py - prey.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < strike_dist {
                return Some((pred_entity, prey_entity));
            }
        }
        None
    }).collect()
}

/// Apply feeding interactions.
/// - **Producers** (plants/food): partial grazing — drain energy, plant survives and
///   regenerates via photosynthesis. Only dies naturally if overgrazed to zero.
/// - **Mobile prey**: killed on capture — energy transferred, prey marked for removal.
pub fn apply_kills(world: &mut World, kills: &[(Entity, Entity)]) -> Vec<Entity> {
    let mut dead = Vec::new();
    for &(pred, prey) in kills {
        let is_producer = world.get::<&Producer>(prey).is_ok();

        if is_producer {
            // Partial grazing: drain energy from producer, don't destroy it
            let graze_amount = world
                .get::<&Energy>(prey)
                .map(|e| e.current.min(e.max * 0.5))
                .unwrap_or(0.0);

            if graze_amount <= 0.0 {
                continue;
            }

            if let Ok(mut prey_energy) = world.get::<&mut Energy>(prey) {
                prey_energy.current -= graze_amount;
            }

            // Grazing efficiency scales with complexity — complex digestive systems
            // extract more nutrition from food
            let complexity = world.get::<&CreatureGenome>(pred)
                .map(|g| g.complexity)
                .unwrap_or(0.0);
            let efficiency = 0.5 + 0.5 * complexity.max(0.1);

            if let Ok(mut pred_energy) = world.get::<&mut Energy>(pred) {
                pred_energy.current = (pred_energy.current + graze_amount * efficiency).min(pred_energy.max);
            }
        } else {
            // Predation: kill mobile prey, transfer energy
            let prey_caloric_value = world
                .get::<&Energy>(prey)
                .map(|e| e.max)
                .unwrap_or(10.0);

            if let Ok(mut pred_energy) = world.get::<&mut Energy>(pred) {
                pred_energy.current = (pred_energy.current + prey_caloric_value * 0.8).min(pred_energy.max);
            }

            // Set prey energy to 0 so death_system removes it
            if let Ok(mut prey_energy) = world.get::<&mut Energy>(prey) {
                prey_energy.current = 0.0;
            }

            dead.push(prey);
        }

        if let Ok(mut needs) = world.get::<&mut Needs>(pred) {
            needs.hunger = (needs.hunger - 0.5).max(0.0);
        }
    }
    dead
}

/// Result of the death system: how many creatures vs non-creatures died.
pub struct DeathResult {
    pub creature_deaths: u64,
    pub total_removed: u64,
    /// Positions and energy of dead creatures for nutrient cycling (detritus spawning).
    pub dead_creature_info: Vec<(f32, f32, f32)>,
}

/// Remove dead entities (energy <= 0 or age exceeded).
pub fn death_system(world: &mut World) -> DeathResult {
    let mut dead = HashSet::new();

    for (entity, energy) in &mut world.query::<(Entity, &Energy)>() {
        if energy.current <= 0.0 {
            dead.insert(entity);
        }
    }
    for (entity, age) in &mut world.query::<(Entity, &Age)>() {
        if age.ticks >= age.max_ticks {
            dead.insert(entity); // HashSet deduplicates automatically
        }
    }

    let mut creature_deaths = 0u64;
    let mut dead_creature_info = Vec::new();

    for &entity in &dead {
        if world.get::<&crate::genome::CreatureGenome>(entity).is_ok() {
            creature_deaths += 1;
            // Collect position and energy for detritus spawning
            let pos = world.get::<&Position>(entity).ok().map(|p| (p.x, p.y));
            let max_e = world.get::<&Energy>(entity).ok().map(|e| e.max).unwrap_or(0.0);
            if let Some((x, y)) = pos {
                dead_creature_info.push((x, y, max_e));
            }
        }
    }

    let total_removed = dead.len() as u64;

    for entity in &dead {
        let _ = world.despawn(*entity);
    }

    DeathResult {
        creature_deaths,
        total_removed,
        dead_creature_info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        metabolism_system(&mut world, 1.0, &default_env(), 24.0);
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
        let e = world.spawn((Energy { current: 0.0, max: 50.0 },));
        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 1);
        assert!(world.get::<&Energy>(e).is_err(), "Entity should be despawned");
    }

    #[test]
    fn test_death_removes_old_age() {
        let mut world = World::new();
        let _e = world.spawn((
            Energy::new(50.0),
            Age { ticks: 1000, max_ticks: 500 },
        ));
        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 1);
    }

    #[test]
    fn test_energy_fraction() {
        let e = Energy { current: 25.0, max: 50.0 };
        assert!((e.fraction() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_logistic_plant_growth() {
        let mut world = World::new();
        let physics = DerivedPhysics {
            base_metabolism: -0.25,
            body_mass: 0.1,
            max_energy: 15.0,
            ..Default::default()
        };

        // Both plants at surface (y=2.0) for max light
        let depleted = world.spawn((
            Energy { current: 3.0, max: 15.0 },
            physics.clone(),
            Position { x: 5.0, y: 2.0 },
        ));
        let full = world.spawn((
            Energy { current: 14.0, max: 15.0 },
            physics,
            Position { x: 10.0, y: 2.0 },
        ));

        let before_depleted = world.get::<&Energy>(depleted).unwrap().current;
        let before_full = world.get::<&Energy>(full).unwrap().current;

        metabolism_system(&mut world, 1.0, &default_env(), 24.0);

        let gained_depleted = world.get::<&Energy>(depleted).unwrap().current - before_depleted;
        let gained_full = world.get::<&Energy>(full).unwrap().current - before_full;

        // Both should net-gain at low fill; at high fill, maintenance may exceed photosynthesis
        assert!(gained_depleted > 0.0, "Depleted plant should gain energy (net of maintenance)");
        // The critical test: depleted plant gains MORE than full plant (logistic growth)
        assert!(
            gained_depleted > gained_full,
            "Depleted plant should grow faster than full: {:.4} vs {:.4}",
            gained_depleted, gained_full,
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
            Energy { current: 10.0, max: 10.0 },
            physics,
            Detritus,
            Position { x: 5.0, y: 5.0 },
        ));

        let initial = world.get::<&Energy>(e).unwrap().current;
        metabolism_system(&mut world, 1.0, &default_env(), 24.0);
        let after = world.get::<&Energy>(e).unwrap().current;
        assert!(
            after < initial,
            "Detritus should decay: {} -> {}",
            initial, after,
        );
    }

    #[test]
    fn test_death_returns_creature_info_for_detritus() {
        let mut world = World::new();
        let genome = crate::genome::CreatureGenome::minimal_cell(&mut rand::rng());
        let physics = crate::phenotype::derive_physics(&genome);
        world.spawn((
            Position { x: 10.0, y: 5.0 },
            Energy { current: 0.0, max: 50.0 },
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

        // Surface plant (y=2, top 8% of 24-high tank)
        let surface = world.spawn((
            Energy { current: 7.5, max: 15.0 },
            physics.clone(),
            Position { x: 5.0, y: 2.0 },
        ));
        // Deep plant (y=20, bottom 83% of 24-high tank)
        let deep = world.spawn((
            Energy { current: 7.5, max: 15.0 },
            physics,
            Position { x: 5.0, y: 20.0 },
        ));

        metabolism_system(&mut world, 1.0, &default_env(), 24.0);

        let surface_e = world.get::<&Energy>(surface).unwrap().current;
        let deep_e = world.get::<&Energy>(deep).unwrap().current;

        assert!(
            surface_e > deep_e,
            "Surface plant should gain more energy than deep: {:.4} vs {:.4}",
            surface_e, deep_e,
        );
    }

    // -- Helper to build an EntityInfoMap entry for prey --
    fn prey_info(x: f32, y: f32, body_mass: f32, max_speed: f32, is_producer: bool) -> crate::EntityInfo {
        crate::EntityInfo {
            x, y,
            vx: 0.0, vy: 0.0,
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
    fn test_grazing_drains_plant_partially() {
        use std::collections::HashMap;

        let mut world = World::new();

        let plant = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Energy { current: 100.0, max: 100.0 },
            Producer,
        ));

        let creature = world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy { current: 20.0, max: 100.0 },
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
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(plant, prey_info(10.0, 10.0, 0.1, 0.0, true));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert_eq!(kills.len(), 1, "Should find one grazing interaction");

        let dead = apply_kills(&mut world, &kills);
        assert!(dead.is_empty(), "Plant should not be killed by grazing");

        let plant_energy = world.get::<&Energy>(plant).unwrap().current;
        // graze_amount = min(100.0, 100.0 * 0.5) = 50.0
        // plant energy = 100.0 - 50.0 = 50.0
        assert!(plant_energy > 0.0, "Plant should survive grazing");
        assert!(
            (plant_energy - 50.0).abs() < 0.01,
            "Plant should have ~50 energy after 50% graze, got {}",
            plant_energy
        );

        let creature_energy = world.get::<&Energy>(creature).unwrap().current;
        // creature gains 50.0 * 0.55 (complexity 0.0, efficiency = 0.5 + 0.5*0.1) = 27.5
        // 20.0 + 27.5 = 47.5
        assert!(
            (creature_energy - 47.5).abs() < 0.01,
            "Creature should gain efficiency-scaled grazed amount: expected ~47.5, got {}",
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
            Energy { current: 40.0, max: 50.0 },
        ));

        // Predator — fast, high hunt_skill
        let predator = world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy { current: 30.0, max: 200.0 },
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
        ));

        let mut grid = SpatialGrid::new(5.0);
        grid.rebuild(&world);

        let mut entity_map: EntityInfoMap = HashMap::new();
        entity_map.insert(prey, prey_info(10.0, 10.0, 0.5, 3.0, false));

        let kills = hunting_check(&world, &grid, &entity_map);
        assert_eq!(kills.len(), 1, "Should find one predation interaction");

        let dead = apply_kills(&mut world, &kills);
        assert_eq!(dead.len(), 1, "Prey should be marked dead");
        assert_eq!(dead[0], prey);

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
            Energy { current: 40.0, max: 50.0 },
        ));

        // hunt_skill = 0.1, below the 0.3 threshold
        world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy { current: 30.0, max: 100.0 },
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
            Energy { current: 40.0, max: 50.0 },
        ));

        // pred_speed 5.0 < prey.max_speed 10.0 * 0.8 = 8.0
        world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy { current: 30.0, max: 100.0 },
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
        let deep = world.spawn((
            Energy::new(100.0),
            physics,
            Position { x: 10.0, y: 45.0 },
        ));

        let initial = world.get::<&Energy>(surface).unwrap().current;

        metabolism_system(&mut world, 1.0, &default_env(), 50.0);

        let loss_surface = initial - world.get::<&Energy>(surface).unwrap().current;
        let loss_deep = initial - world.get::<&Energy>(deep).unwrap().current;

        assert!(
            loss_surface > loss_deep,
            "Surface creature should lose MORE energy: surface lost {:.4}, deep lost {:.4}",
            loss_surface, loss_deep
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
            Energy { current: 100.0, max: 100.0 },
            Producer,
        ));

        // graze_skill = 0.1, below the 0.2 threshold
        world.spawn((
            Position { x: 11.0, y: 10.0 },
            Energy { current: 20.0, max: 100.0 },
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
        let e_day = world_day.spawn((Energy::new(100.0), physics.clone(), Position { x: 50.0, y: 5.0 }));
        let initial_day = world_day.get::<&Energy>(e_day).unwrap().current;

        let mut env_day = Environment::default();
        env_day.light_level = 1.0;
        metabolism_system(&mut world_day, 1.0, &env_day, 50.0);
        let after_day = world_day.get::<&Energy>(e_day).unwrap().current;
        let loss_day = initial_day - after_day;

        // Night world: light_level = 0.0 → night_stress = 1.3
        let mut world_night = World::new();
        let e_night = world_night.spawn((Energy::new(100.0), physics, Position { x: 50.0, y: 5.0 }));
        let initial_night = world_night.get::<&Energy>(e_night).unwrap().current;

        let mut env_night = Environment::default();
        env_night.light_level = 0.0;
        metabolism_system(&mut world_night, 1.0, &env_night, 50.0);
        let after_night = world_night.get::<&Energy>(e_night).unwrap().current;
        let loss_night = initial_night - after_night;

        assert!(loss_day > 0.0, "Day creature should lose energy");
        assert!(loss_night > loss_day, "Night creature should lose more energy than day");

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
        let e_normal = world_normal.spawn((Energy::new(100.0), physics.clone(), Position { x: 50.0, y: 5.0 }));
        let initial_normal = world_normal.get::<&Energy>(e_normal).unwrap().current;

        let mut env_normal = Environment::default();
        env_normal.temperature = 25.0;
        env_normal.light_level = 1.0; // eliminate night stress
        metabolism_system(&mut world_normal, 1.0, &env_normal, 50.0);
        let loss_normal = initial_normal - world_normal.get::<&Energy>(e_normal).unwrap().current;

        // Cold snap (20°C → temp_modifier = 1.0 + (20-25)*0.02 = 0.9)
        let mut world_cold = World::new();
        let e_cold = world_cold.spawn((Energy::new(100.0), physics, Position { x: 50.0, y: 5.0 }));
        let initial_cold = world_cold.get::<&Energy>(e_cold).unwrap().current;

        let mut env_cold = Environment::default();
        env_cold.temperature = 20.0;
        env_cold.light_level = 1.0;
        metabolism_system(&mut world_cold, 1.0, &env_cold, 50.0);
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
            Energy { current: 20.0, max: 50.0 },
            plant_physics.clone(),
            Position { x: 5.0, y: 2.0 },
        ));
        let initial_normal = world_normal.get::<&Energy>(p_normal).unwrap().current;

        let mut env_normal = Environment::default();
        env_normal.temperature = 25.0;
        env_normal.light_level = 1.0;
        metabolism_system(&mut world_normal, 1.0, &env_normal, 50.0);
        let gain_normal = world_normal.get::<&Energy>(p_normal).unwrap().current - initial_normal;

        // Cold snap (20°C → temp_modifier = 0.9)
        let mut world_cold = World::new();
        let p_cold = world_cold.spawn((
            Energy { current: 20.0, max: 50.0 },
            plant_physics,
            Position { x: 5.0, y: 2.0 },
        ));
        let initial_cold = world_cold.get::<&Energy>(p_cold).unwrap().current;

        let mut env_cold = Environment::default();
        env_cold.temperature = 20.0;
        env_cold.light_level = 1.0;
        metabolism_system(&mut world_cold, 1.0, &env_cold, 50.0);
        let gain_cold = world_cold.get::<&Energy>(p_cold).unwrap().current - initial_cold;

        assert!(gain_normal > 0.0, "Plant should gain energy via photosynthesis");
        assert!(gain_cold > 0.0, "Cold plant should still gain energy");
        assert!(
            gain_cold < gain_normal,
            "Cold plant should gain LESS energy: cold={:.4} vs normal={:.4}",
            gain_cold, gain_normal,
        );
    }

    #[test]
    fn test_complex_creature_survives_night_better() {
        use crate::genome::CreatureGenome;
        use crate::phenotype::derive_physics;

        let mut rng = rand::rng();
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
            physics_complex.base_metabolism, physics_simple.base_metabolism,
        );

        // Run both through night metabolism
        let mut world_simple = World::new();
        let max_e = 100.0;
        let e_simple = world_simple.spawn((
            Energy { current: max_e, max: max_e },
            physics_simple,
            Position { x: 50.0, y: 5.0 },
        ));

        let mut world_complex = World::new();
        let e_complex = world_complex.spawn((
            Energy { current: max_e, max: max_e },
            physics_complex,
            Position { x: 50.0, y: 5.0 },
        ));

        let mut env_night = Environment::default();
        env_night.light_level = 0.0; // full night stress

        metabolism_system(&mut world_simple, 1.0, &env_night, 50.0);
        metabolism_system(&mut world_complex, 1.0, &env_night, 50.0);

        let loss_simple = max_e - world_simple.get::<&Energy>(e_simple).unwrap().current;
        let loss_complex = max_e - world_complex.get::<&Energy>(e_complex).unwrap().current;

        assert!(
            loss_complex < loss_simple,
            "Complex creature should lose less energy at night: complex={:.4} vs simple={:.4}",
            loss_complex, loss_simple,
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
                Energy { current: 100.0, max: 100.0 },
                Producer,
            ));

            let mut genome = CreatureGenome::minimal_cell(&mut rand::rng());
            genome.complexity = c;

            let creature = world.spawn((
                Position { x: 11.0, y: 10.0 },
                Energy { current: 20.0, max: 100.0 },
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
            ));

            let mut grid = SpatialGrid::new(5.0);
            grid.rebuild(&world);

            let mut entity_map: EntityInfoMap = HashMap::new();
            entity_map.insert(plant, prey_info(10.0, 10.0, 0.1, 0.0, true));

            let kills = hunting_check(&world, &grid, &entity_map);
            apply_kills(&mut world, &kills);

            let gain = world.get::<&Energy>(creature).unwrap().current - 20.0;
            gains.push((c, gain));
        }

        // Each higher complexity should yield more energy
        for i in 1..gains.len() {
            assert!(
                gains[i].1 > gains[i - 1].1,
                "Higher complexity should yield more grazing energy: \
                 complexity {:.1} got {:.2}, complexity {:.1} got {:.2}",
                gains[i].0, gains[i].1, gains[i - 1].0, gains[i - 1].1,
            );
        }
    }
}
