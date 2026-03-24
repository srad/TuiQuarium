//! Reproduction system: mate pairing, genome crossover, offspring spawning.

use hecs::{Entity, World};
use rand::Rng;
use rand::RngExt;

use crate::behavior::BehaviorState;
use crate::boids::Boid;
use crate::brain::Brain;
use crate::components::*;
use crate::ecosystem::{Age, Energy, TrophicRole};
use crate::genetics::{crossover, genomic_distance, mutate};
use crate::genome::{CreatureGenome, DietType};
use crate::needs::{NeedWeights, Needs};
use crate::phenotype::derive_physics;
use crate::spatial::SpatialGrid;

/// Minimum energy fraction to be eligible to reproduce.
const REPRODUCTION_ENERGY_THRESHOLD: f32 = 0.7;
/// Energy cost to the parent upon reproduction (fraction of max).
const REPRODUCTION_ENERGY_COST: f32 = 0.4;
/// Maximum genomic distance for two creatures to be considered compatible mates.
const MATE_COMPATIBILITY_DISTANCE: f32 = 5.0;
/// Mutation rate for offspring.
const MUTATION_RATE: f32 = 0.15;
/// Maximum number of offspring per tick (prevents burst spawning).
const MAX_BIRTHS_PER_TICK: usize = 3;

/// Find ready-to-mate pairs and spawn offspring.
/// Returns the list of newly spawned entities.
pub fn reproduction_system(
    world: &mut World,
    grid: &SpatialGrid,
    rng: &mut impl Rng,
    tank_w: f32,
    tank_h: f32,
) -> Vec<Entity> {
    // Collect potential parents: entities with genome, needs, energy
    let candidates: Vec<(Entity, f32, f32, f32, f32)> = {
        let mut v = Vec::new();
        for (entity, pos, needs, energy) in
            &mut world.query::<(Entity, &Position, &Needs, &Energy)>()
        {
            if needs.reproduction > 0.9 && energy.fraction() > REPRODUCTION_ENERGY_THRESHOLD {
                v.push((entity, pos.x, pos.y, needs.reproduction, energy.fraction()));
            }
        }
        v
    };

    let mut births = Vec::new();
    let mut already_mated: Vec<Entity> = Vec::new();

    for &(parent_a, ax, ay, _, _) in &candidates {
        if births.len() >= MAX_BIRTHS_PER_TICK {
            break;
        }
        if already_mated.contains(&parent_a) {
            continue;
        }

        // Find nearby compatible mate
        let neighbors = grid.neighbors(ax, ay, 8.0);
        let mut best_mate: Option<Entity> = None;

        for &neighbor in &neighbors {
            if neighbor == parent_a || already_mated.contains(&neighbor) {
                continue;
            }
            // Check if neighbor is also a ready candidate
            if !candidates.iter().any(|(e, ..)| *e == neighbor) {
                continue;
            }
            // Check genomic compatibility
            let compatible = {
                let ga = world.get::<&CreatureGenome>(parent_a);
                let gb = world.get::<&CreatureGenome>(neighbor);
                if let (Ok(ga), Ok(gb)) = (ga, gb) {
                    genomic_distance(&ga, &gb) < MATE_COMPATIBILITY_DISTANCE
                } else {
                    false
                }
            };
            if compatible {
                best_mate = Some(neighbor);
                break;
            }
        }

        let parent_b = match best_mate {
            Some(b) => b,
            None => continue,
        };

        // Cross genomes and mutate
        let child_genome = {
            let ga = world.get::<&CreatureGenome>(parent_a).unwrap().clone();
            let gb = world.get::<&CreatureGenome>(parent_b).unwrap().clone();
            let mut child = crossover(&ga, &gb, rng);
            mutate(&mut child, MUTATION_RATE, rng);
            child
        };

        // Derive physics from child genome
        let physics = derive_physics(&child_genome);

        // Spawn offspring near parents
        let spawn_x = (ax + rng.random_range(-3.0..3.0)).clamp(1.0, tank_w - 1.0);
        let spawn_y = (ay + rng.random_range(-2.0..2.0)).clamp(1.0, tank_h - 3.0);

        let vx = rng.random_range(-1.0..1.0);
        let vy = rng.random_range(-0.5..0.5);

        // Build appearance from genome — simple placeholder frames
        // (The render crate generates the real art; core just needs the frame structure)
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: if vx >= 0.0 { Direction::Right } else { Direction::Left },
            color_index: child_genome.art.primary_color,
        };

        let trophic_role = match child_genome.behavior.diet {
            DietType::Herbivore => TrophicRole::Herbivore,
            DietType::Omnivore => TrophicRole::Omnivore,
            DietType::Carnivore => TrophicRole::Carnivore,
        };

        let max_ticks = (5000.0 * child_genome.behavior.max_lifespan_factor) as u64;

        let brain = Brain::from_genome(&child_genome.brain);
        let child_entity = world.spawn((
            Position { x: spawn_x, y: spawn_y },
            Velocity { vx, vy },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2 / child_genome.anim.swim_speed),
            child_genome,
            physics.clone(),
            Energy::new(physics.max_energy * 0.3), // offspring start hungry, must find food
            Age { ticks: 0, max_ticks },
            trophic_role,
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ));

        // Deduct energy from parents
        if let Ok(mut e) = world.get::<&mut Energy>(parent_a) {
            e.current -= e.max * REPRODUCTION_ENERGY_COST;
        }
        if let Ok(mut e) = world.get::<&mut Energy>(parent_b) {
            e.current -= e.max * REPRODUCTION_ENERGY_COST;
        }
        // Reset reproduction need for parents
        if let Ok(mut n) = world.get::<&mut Needs>(parent_a) {
            n.reproduction = 0.0;
        }
        if let Ok(mut n) = world.get::<&mut Needs>(parent_b) {
            n.reproduction = 0.0;
        }

        already_mated.push(parent_a);
        already_mated.push(parent_b);
        births.push(child_entity);
    }

    births
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_ready_parent(
        world: &mut World,
        x: f32,
        y: f32,
        genome: CreatureGenome,
    ) -> Entity {
        let physics = derive_physics(&genome);
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        let mut needs = Needs::default();
        needs.reproduction = 0.95; // ready to mate

        let brain = Brain::from_genome(&genome.brain);
        world.spawn((
            Position { x, y },
            Velocity { vx: 0.0, vy: 0.0 },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            genome,
            physics.clone(),
            Energy::new(physics.max_energy), // full energy
            Age { ticks: 100, max_ticks: 5000 },
            TrophicRole::Herbivore,
            needs,
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ))
    }

    #[test]
    fn test_reproduction_spawns_offspring() {
        let mut world = World::new();
        let mut rng = rand::rng();

        // Use identical genomes to ensure compatibility
        let g1 = CreatureGenome::random(&mut rng);
        let g2 = g1.clone();

        let _p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let initial_count = world.len();
        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0);

        assert!(!births.is_empty(), "Should have produced offspring");
        assert_eq!(world.len() as usize, initial_count as usize + births.len());

        // Offspring should have a genome
        for &child in &births {
            assert!(
                world.get::<&CreatureGenome>(child).is_ok(),
                "Offspring should carry a genome"
            );
            assert!(
                world.get::<&Energy>(child).is_ok(),
                "Offspring should have energy"
            );
        }
    }

    #[test]
    fn test_reproduction_costs_parent_energy() {
        let mut world = World::new();
        let mut rng = rand::rng();

        let g1 = CreatureGenome::random(&mut rng);
        let g2 = g1.clone();

        let p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let before = world.get::<&Energy>(p1).unwrap().current;

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0);

        if !births.is_empty() {
            let after = world.get::<&Energy>(p1).unwrap().current;
            assert!(after < before, "Parent should lose energy: {} -> {}", before, after);
        }
    }

    #[test]
    fn test_low_energy_prevents_reproduction() {
        let mut world = World::new();
        let mut rng = rand::rng();

        let g1 = CreatureGenome::random(&mut rng);
        let physics = derive_physics(&g1);
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        let mut needs = Needs::default();
        needs.reproduction = 0.95;

        // Parent with very low energy
        let brain1 = Brain::from_genome(&g1.brain);
        let _p1 = world.spawn((
            Position { x: 10.0, y: 10.0 },
            Velocity { vx: 0.0, vy: 0.0 },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance.clone(),
            AnimationState::new(0.2),
            g1.clone(),
            physics.clone(),
            Energy { current: 1.0, max: physics.max_energy },
            Age { ticks: 100, max_ticks: 5000 },
            TrophicRole::Herbivore,
            needs.clone(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain1,
        ));

        let g2 = g1.clone();
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0);
        assert!(births.is_empty(), "Low-energy creature should not reproduce");
    }

    #[test]
    fn test_offspring_genome_is_blend() {
        let mut world = World::new();
        let mut rng = rand::rng();

        // Create parents with very different body plans but identical behavior for compatibility
        let mut g1 = CreatureGenome::random(&mut rng);
        g1.art.body_size = 0.7;
        
        let mut g2 = g1.clone();
        g2.art.body_size = 1.5;
        // Make sure behavior genome is close enough (cloning ensures identical behavior)

        let _p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0);

        if !births.is_empty() {
            let child_genome = world.get::<&CreatureGenome>(births[0]).unwrap();
            // Child body size should be either 0.7 or 1.5 (uniform crossover picks one parent's gene)
            // After mutation it may shift slightly
            assert!(
                child_genome.art.body_size >= 0.6 && child_genome.art.body_size <= 1.6,
                "Child body_size {} should be near a parent value",
                child_genome.art.body_size
            );
        }
    }
}
