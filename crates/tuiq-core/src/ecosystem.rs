//! Energy-based ecosystem: metabolism, feeding, hunting, death.

use hecs::{Entity, World};

use crate::components::{Position, Velocity};
use crate::needs::Needs;
use crate::phenotype::DerivedPhysics;
use crate::spatial::SpatialGrid;

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

/// Trophic role tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrophicRole {
    Producer,   // plants
    Herbivore,
    Omnivore,
    Carnivore,
}

impl TrophicRole {
    pub fn can_eat(&self, prey_role: TrophicRole) -> bool {
        match self {
            TrophicRole::Producer => false,
            TrophicRole::Herbivore => matches!(prey_role, TrophicRole::Producer),
            TrophicRole::Omnivore => matches!(prey_role, TrophicRole::Producer | TrophicRole::Herbivore),
            TrophicRole::Carnivore => matches!(prey_role, TrophicRole::Herbivore | TrophicRole::Omnivore),
        }
    }
}

/// Drain energy from metabolism each tick.
/// base_metabolism is already computed in DerivedPhysics.
pub fn metabolism_system(world: &mut World, dt: f32) {
    for (energy, physics) in world.query_mut::<(&mut Energy, &DerivedPhysics)>() {
        // Metabolism multiplier balances energy economy: creatures must eat regularly
        // but don't starve instantly. Full rate (1.0) is too harsh for small populations.
        energy.current -= physics.base_metabolism * dt * 0.5;
    }
}

/// Advance age of all creatures.
pub fn age_system(world: &mut World) {
    for age in world.query_mut::<&mut Age>() {
        age.ticks += 1;
    }
}

/// Steer hungry creatures toward nearby food sources.
/// This gives creatures agency to seek food rather than relying on random encounters.
pub fn feeding_steering_system(world: &mut World, _grid: &SpatialGrid, dt: f32) {
    // Collect hungry creatures and their positions
    let hungry: Vec<(Entity, f32, f32, f32, TrophicRole)> = {
        let mut v = Vec::new();
        for (entity, pos, needs, physics, role) in
            &mut world.query::<(Entity, &Position, &Needs, &DerivedPhysics, &TrophicRole)>()
        {
            if needs.hunger > 0.3 && *role != TrophicRole::Producer {
                v.push((entity, pos.x, pos.y, physics.sensory_range, *role));
            }
        }
        v
    };

    // Collect food source positions
    let food_sources: Vec<(Entity, f32, f32, TrophicRole)> = {
        let mut v = Vec::new();
        for (entity, pos, role) in &mut world.query::<(Entity, &Position, &TrophicRole)>() {
            v.push((entity, pos.x, pos.y, *role));
        }
        v
    };

    // Steer each hungry creature toward nearest edible target
    let mut steers: Vec<(Entity, f32, f32)> = Vec::new();

    for &(entity, px, py, sense_range, role) in &hungry {
        let mut best_dist = f32::MAX;
        let mut best_dx = 0.0_f32;
        let mut best_dy = 0.0_f32;

        for &(food_entity, fx, fy, food_role) in &food_sources {
            if food_entity == entity {
                continue;
            }
            if !role.can_eat(food_role) {
                continue;
            }
            let dx = fx - px;
            let dy = fy - py;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < sense_range && dist < best_dist {
                best_dist = dist;
                best_dx = dx;
                best_dy = dy;
            }
        }

        if best_dist < f32::MAX {
            // Normalize and scale by hunger urgency
            let mag = (best_dx * best_dx + best_dy * best_dy).sqrt().max(0.01);
            let strength = 2.0; // steering strength toward food
            steers.push((entity, best_dx / mag * strength, best_dy / mag * strength));
        }
    }

    // Apply steering forces to velocities
    for (entity, sx, sy) in steers {
        if let Ok(mut vel) = world.get::<&mut Velocity>(entity) {
            vel.vx += sx * dt;
            vel.vy += sy * dt;
        }
    }
}

/// Check predator-prey interactions. Returns list of (predator, prey) entity pairs
/// where the predator successfully catches the prey.
pub fn hunting_check(
    world: &World,
    grid: &SpatialGrid,
) -> Vec<(Entity, Entity)> {
    let mut kills = Vec::new();

    // Collect all potential eaters (Herbivores, Omnivores, Carnivores)
    let eaters: Vec<(Entity, f32, f32, f32, f32, TrophicRole)> = {
        let mut v = Vec::new();
        for (entity, pos, physics, role) in
            &mut world.query::<(Entity, &Position, &DerivedPhysics, &TrophicRole)>()
        {
            if *role != TrophicRole::Producer {
                v.push((entity, pos.x, pos.y, physics.max_speed, physics.sensory_range, *role));
            }
        }
        v
    };

    for &(pred_entity, px, py, pred_speed, sense_range, pred_role) in &eaters {
        let neighbors = grid.neighbors(px, py, sense_range);
        for &prey_entity in &neighbors {
            if prey_entity == pred_entity {
                continue;
            }
            // Check if prey is edible
            let prey_role = match world.get::<&TrophicRole>(prey_entity) {
                Ok(role) => *role,
                Err(_) => continue,
            };
            if !pred_role.can_eat(prey_role) {
                continue;
            }

            // Body size and speed checks only apply to hunting animals.
            // Grazing on producers (plants/food) skips these — any creature can graze.
            let is_grazing = prey_role == TrophicRole::Producer;

            if !is_grazing {
                if let (Ok(pred_phys), Ok(prey_phys)) = (
                    world.get::<&DerivedPhysics>(pred_entity),
                    world.get::<&DerivedPhysics>(prey_entity),
                ) {
                    if pred_phys.body_mass < prey_phys.body_mass * 1.5 {
                        continue; // Too big to eat
                    }
                    if pred_speed < prey_phys.max_speed * 0.8 {
                        continue; // Too fast to catch
                    }
                }
            }

            // Check distance — grazing has a larger reach than hunting strikes
            let strike_dist = if is_grazing { 5.0 } else { 3.0 };

            if let Ok(prey_pos) = world.get::<&Position>(prey_entity) {
                let dx = px - prey_pos.x;
                let dy = py - prey_pos.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < strike_dist {
                    kills.push((pred_entity, prey_entity));
                    break; // One meal per creature per tick
                }
            }
        }
    }
    kills
}

/// Apply kills: transfer energy from prey to predator, mark prey for removal.
pub fn apply_kills(world: &mut World, kills: &[(Entity, Entity)]) -> Vec<Entity> {
    let mut dead = Vec::new();
    for &(pred, prey) in kills {
        // Get prey's caloric value (max_energy represents the total energy content of the body)
        let prey_caloric_value = world
            .get::<&Energy>(prey)
            .map(|e| e.max)
            .unwrap_or(10.0);

        // Transfer 80% of caloric value to predator (high efficiency for gameplay)
        if let Ok(mut pred_energy) = world.get::<&mut Energy>(pred) {
            pred_energy.current = (pred_energy.current + prey_caloric_value * 0.8).min(pred_energy.max);
            
            // Eating also reduces hunger drive immediately
            if let Ok(mut needs) = world.get::<&mut crate::needs::Needs>(pred) {
                needs.hunger = (needs.hunger - 0.5).max(0.0);
            }
        }

        dead.push(prey);
    }
    dead
}

/// Result of the death system: how many creatures vs non-creatures died.
pub struct DeathResult {
    pub creature_deaths: u64,
    pub total_removed: u64,
}

/// Remove dead entities (energy <= 0 or age exceeded).
/// Returns separate counts for creature deaths vs total removals (food/plants).
pub fn death_system(world: &mut World) -> DeathResult {
    let mut dead = Vec::new();

    // Collect dead entities
    for (entity, energy) in &mut world.query::<(Entity, &Energy)>() {
        if energy.current <= 0.0 {
            dead.push(entity);
        }
    }
    for (entity, age) in &mut world.query::<(Entity, &Age)>() {
        if age.ticks >= age.max_ticks && !dead.contains(&entity) {
            dead.push(entity);
        }
    }

    // Count creature deaths (entities that have a CreatureGenome)
    let creature_deaths = dead
        .iter()
        .filter(|e| world.get::<&crate::genome::CreatureGenome>(**e).is_ok())
        .count() as u64;

    let total_removed = dead.len() as u64;

    // Remove them
    for entity in &dead {
        let _ = world.despawn(*entity);
    }

    DeathResult {
        creature_deaths,
        total_removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let e = world.spawn((Energy::new(50.0), physics));

        // 50 * 0.8 = 40 initial energy
        let initial = world.get::<&Energy>(e).unwrap().current;
        metabolism_system(&mut world, 1.0);
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
    fn test_trophic_roles() {
        assert!(!TrophicRole::Producer.can_eat(TrophicRole::Herbivore));
        assert!(TrophicRole::Herbivore.can_eat(TrophicRole::Producer));
        assert!(TrophicRole::Carnivore.can_eat(TrophicRole::Herbivore));
        assert!(!TrophicRole::Carnivore.can_eat(TrophicRole::Carnivore));
        assert!(TrophicRole::Omnivore.can_eat(TrophicRole::Producer));
        assert!(TrophicRole::Omnivore.can_eat(TrophicRole::Herbivore));
    }

    #[test]
    fn test_energy_fraction() {
        let e = Energy { current: 25.0, max: 50.0 };
        assert!((e.fraction() - 0.5).abs() < 0.01);
    }
}
