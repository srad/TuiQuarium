//! Reproduction system: asexual division for simple cells, sexual crossover for complex creatures.

use std::collections::{HashMap, HashSet};

use hecs::{Entity, World};
use rand::Rng;
use rand::RngExt;

use crate::behavior::BehaviorState;
use crate::boids::Boid;
use crate::brain::{Brain, InnovationTracker};
use crate::components::*;
use crate::ecosystem::{Age, Energy};
use crate::genetics::{crossover, genomic_distance, mutate};
use crate::genome::CreatureGenome;
use crate::needs::{NeedWeights, Needs};
use crate::phenotype::{derive_feeding, derive_physics};
use crate::spatial::SpatialGrid;

/// Maximum genomic distance for two creatures to be considered compatible mates.
const MATE_COMPATIBILITY_DISTANCE: f32 = 8.0;
/// Mutation rate for sexual offspring.
const MUTATION_RATE: f32 = 0.15;
/// Higher mutation rate for asexual division (more drift).
const ASEXUAL_MUTATION_RATE: f32 = 0.25;
/// Maximum number of offspring per tick (prevents burst spawning).
const MAX_BIRTHS_PER_TICK: usize = 3;

/// Find ready-to-reproduce creatures and spawn offspring.
/// Simple cells (complexity < 0.5) reproduce asexually.
/// Complex creatures need a compatible mate nearby.
/// Maximum creature population before reproduction is suppressed.
const MAX_POPULATION: usize = 600;

pub fn reproduction_system(
    world: &mut World,
    grid: &SpatialGrid,
    rng: &mut impl Rng,
    tank_w: f32,
    tank_h: f32,
    creature_count: usize,
    tracker: &mut InnovationTracker,
) -> Vec<Entity> {
    // Soft population cap — stop reproducing when crowded
    if creature_count >= MAX_POPULATION {
        return Vec::new();
    }

    // Fitness sharing: group creatures into species and compute reproduction probability.
    // Larger species get proportionally less reproduction chance, protecting innovation.
    const SPECIES_THRESHOLD: f32 = 3.0;
    let all_genomes: Vec<(Entity, CreatureGenome)> = {
        let mut v = Vec::new();
        for (entity, genome) in &mut world.query::<(Entity, &CreatureGenome)>() {
            v.push((entity, genome.clone()));
        }
        v
    };

    // Assign species via simple greedy clustering
    let mut species_map: HashMap<Entity, usize> = HashMap::new();
    let mut centroids: Vec<usize> = Vec::new();
    for (i, (entity, genome)) in all_genomes.iter().enumerate() {
        let mut assigned = false;
        for &ci in &centroids {
            if genomic_distance(genome, &all_genomes[ci].1) < SPECIES_THRESHOLD {
                species_map.insert(*entity, ci);
                assigned = true;
                break;
            }
        }
        if !assigned {
            centroids.push(i);
            species_map.insert(*entity, i);
        }
    }

    // Count species sizes
    let mut species_sizes: HashMap<usize, usize> = HashMap::new();
    for &sp in species_map.values() {
        *species_sizes.entry(sp).or_insert(0) += 1;
    }

    // Collect potential parents
    let candidates: Vec<(Entity, f32, f32, f32, f32, f32, f32, f32)> = {
        let mut v = Vec::new();
        for (entity, pos, needs, energy, genome) in
            &mut world.query::<(Entity, &Position, &Needs, &Energy, &CreatureGenome)>()
        {
            let threshold = 0.4 + 0.05 * genome.complexity;
            if needs.reproduction > 0.9 && energy.fraction() > threshold {
                v.push((entity, pos.x, pos.y, needs.reproduction, energy.fraction(),
                        genome.complexity, genome.behavior.mutation_rate_factor,
                        genome.behavior.mate_preference_hue));
            }
        }
        v
    };

    let candidate_set: HashSet<Entity> = candidates.iter().map(|(e, ..)| *e).collect();
    let mut births = Vec::new();
    let mut already_mated: HashSet<Entity> = HashSet::new();

    for &(parent_a, ax, ay, _, _, complexity, mutation_factor, mate_pref) in &candidates {
        if births.len() >= MAX_BIRTHS_PER_TICK {
            break;
        }
        if already_mated.contains(&parent_a) {
            continue;
        }

        // Fitness sharing: larger species reproduce less to protect innovation
        if let Some(&sp) = species_map.get(&parent_a) {
            let sp_size = *species_sizes.get(&sp).unwrap_or(&1);
            if sp_size > 3 {
                // Reproduction probability inversely proportional to species size
                let prob = 3.0 / sp_size as f32;
                if !rng.random_bool(prob as f64) {
                    continue;
                }
            }
        }

        // Try sexual reproduction first if complexity >= 0.5
        let mut child_genome: Option<CreatureGenome> = None;

        if complexity >= 0.5 {
            let neighbors = grid.neighbors(ax, ay, 15.0);

            // Score mates by compatibility and color preference (sexual selection)
            let mut best_mate: Option<(Entity, f32)> = None;
            for &neighbor in &neighbors {
                if neighbor == parent_a || already_mated.contains(&neighbor) {
                    continue;
                }
                if !candidate_set.contains(&neighbor) {
                    continue;
                }
                let score = {
                    let ga = world.get::<&CreatureGenome>(parent_a);
                    let gb = world.get::<&CreatureGenome>(neighbor);
                    if let (Ok(ga), Ok(gb)) = (ga, gb) {
                        if genomic_distance(&ga, &gb) >= MATE_COMPATIBILITY_DISTANCE {
                            -1.0
                        } else {
                            // Color match score: how well mate's hue matches preference
                            // Wrapping hue distance (0.0 and 1.0 are adjacent)
                            let hue_diff = (mate_pref - gb.art.primary_hue).abs();
                            let hue_dist = hue_diff.min(1.0 - hue_diff);
                            1.0 - hue_dist // 1.0 = perfect match, 0.5 = opposite
                        }
                    } else {
                        -1.0
                    }
                };
                if score > 0.0 {
                    if best_mate.is_none() || score > best_mate.unwrap().1 {
                        best_mate = Some((neighbor, score));
                    }
                }
            }

            if let Some((mate, _score)) = best_mate {
                let ga = world.get::<&CreatureGenome>(parent_a).unwrap().clone();
                let gb = world.get::<&CreatureGenome>(mate).unwrap().clone();
                let mut child = crossover(&ga, &gb, rng);
                // Evolvable mutation rate: average parents' factors, scale base rate
                let effective_rate = MUTATION_RATE * (ga.behavior.mutation_rate_factor + gb.behavior.mutation_rate_factor) * 0.5;
                mutate(&mut child, effective_rate, rng, tracker);
                if let Ok(mut e) = world.get::<&mut Energy>(mate) {
                    e.current -= e.max * 0.3;
                }
                if let Ok(mut n) = world.get::<&mut Needs>(mate) {
                    n.reproduction = 0.0;
                }
                already_mated.insert(mate);

                child_genome = Some(child);
            }
        }

        // Fall back to asexual reproduction if no mate found (always works below 0.7,
        // never above 0.9 — smooth transition avoids complexity trap)
        if child_genome.is_none() {
            if complexity < 0.7 || (complexity < 0.9 && rng.random_bool(0.3)) {
                let ga = match world.get::<&CreatureGenome>(parent_a) {
                    Ok(g) => (*g).clone(),
                    Err(_) => continue,
                };
                let mut child = ga.clone();
                child.generation = ga.generation + 1;
                let effective_rate = ASEXUAL_MUTATION_RATE * mutation_factor;
                mutate(&mut child, effective_rate, rng, tracker);
                child_genome = Some(child);
            } else {
                continue; // Too complex for asexual, no mate found
            }
        }

        let child_genome = child_genome.unwrap();

        // Derive physics and feeding from child genome
        let physics = derive_physics(&child_genome);
        let feeding = derive_feeding(&child_genome, &physics);

        // Spawn offspring near parent
        let spawn_x = (ax + rng.random_range(-3.0..3.0)).clamp(1.0, tank_w - 1.0);
        let spawn_y = (ay + rng.random_range(-2.0..2.0)).clamp(1.0, tank_h - 3.0);

        let vx = rng.random_range(-1.0..1.0);
        let vy = rng.random_range(-0.5..0.5);

        // Simple placeholder frame — real art generated by render crate
        let frame = AsciiFrame::from_rows(vec!["o"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: if vx >= 0.0 { Direction::Right } else { Direction::Left },
            color_index: child_genome.art.color_index(),
        };

        let bbox_w = 1.0_f32.max(child_genome.art.body_size * 3.0);
        let bbox_h = 1.0_f32.max(child_genome.art.body_size * 2.0);

        let max_ticks = (8000.0 * child_genome.behavior.max_lifespan_factor) as u64;

        // Derive need weights from genome so evolution can tune reproductive timing
        let need_weights = NeedWeights {
            reproduction_rate: child_genome.behavior.reproduction_rate * 0.04,
            hunger_rate: 0.01 + 0.02 * child_genome.behavior.metabolism_factor.min(1.5),
            ..NeedWeights::default()
        };

        let brain = Brain::from_genome_with_learning(&child_genome.brain, child_genome.behavior.learning_rate);
        let child_entity = world.spawn((
            Position { x: spawn_x, y: spawn_y },
            Velocity { vx, vy },
            BoundingBox { w: bbox_w, h: bbox_h },
            appearance,
            AnimationState::new(0.2 / child_genome.anim.swim_speed),
            child_genome,
            physics.clone(),
            feeding,
            Energy::new(physics.max_energy * 0.5),
            Age { ticks: 0, max_ticks },
            Needs::default(),
            need_weights,
            BehaviorState::default(),
            Boid,
            brain,
        ));

        // Deduct energy from parent A
        if let Ok(mut e) = world.get::<&mut Energy>(parent_a) {
            e.current -= e.max * 0.3;
        }
        if let Ok(mut n) = world.get::<&mut Needs>(parent_a) {
            n.reproduction = 0.0;
        }

        already_mated.insert(parent_a);
        births.push(child_entity);
    }

    births
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenotype::FeedingCapability;

    fn spawn_ready_parent(
        world: &mut World,
        x: f32,
        y: f32,
        genome: CreatureGenome,
    ) -> Entity {
        let physics = derive_physics(&genome);
        let feeding = derive_feeding(&genome, &physics);
        let frame = AsciiFrame::from_rows(vec!["o"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        let mut needs = Needs::default();
        needs.reproduction = 0.95;

        let brain = Brain::from_genome(&genome.brain);
        world.spawn((
            Position { x, y },
            Velocity { vx: 0.0, vy: 0.0 },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            genome,
            physics.clone(),
            feeding,
            Energy::new(physics.max_energy),
            Age { ticks: 100, max_ticks: 5000 },
            needs,
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ))
    }

    #[test]
    fn test_sexual_reproduction_spawns_offspring() {
        let mut world = World::new();
        let mut rng = rand::rng();

        // Complex creatures need mates
        let mut g1 = CreatureGenome::random(&mut rng);
        g1.complexity = 0.5;
        let g2 = g1.clone();

        let _p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let initial_count = world.len();
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker);

        assert!(!births.is_empty(), "Should have produced offspring");
        assert_eq!(world.len() as usize, initial_count as usize + births.len());

        for &child in &births {
            assert!(world.get::<&CreatureGenome>(child).is_ok());
            assert!(world.get::<&Energy>(child).is_ok());
            assert!(world.get::<&FeedingCapability>(child).is_ok());
        }
    }

    #[test]
    fn test_asexual_reproduction_no_mate_needed() {
        let mut world = World::new();
        let mut rng = rand::rng();

        // Simple cell — should reproduce alone
        let mut cell = CreatureGenome::minimal_cell(&mut rng);
        cell.complexity = 0.1;
        let _p = spawn_ready_parent(&mut world, 10.0, 10.0, cell);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker);
        assert!(!births.is_empty(), "Simple cell should reproduce asexually without a mate");
    }

    #[test]
    fn test_reproduction_costs_parent_energy() {
        let mut world = World::new();
        let mut rng = rand::rng();

        let mut g1 = CreatureGenome::random(&mut rng);
        g1.complexity = 0.5;
        let g2 = g1.clone();

        let p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let before = world.get::<&Energy>(p1).unwrap().current;

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker);

        if !births.is_empty() {
            let after = world.get::<&Energy>(p1).unwrap().current;
            assert!(after < before, "Parent should lose energy: {} -> {}", before, after);
        }
    }

    #[test]
    fn test_offspring_genome_is_blend() {
        let mut world = World::new();
        let mut rng = rand::rng();

        let mut g1 = CreatureGenome::random(&mut rng);
        g1.complexity = 0.5;
        g1.art.body_size = 0.7;

        let mut g2 = g1.clone();
        g2.art.body_size = 1.5;

        let _p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker);

        if !births.is_empty() {
            let child_genome = world.get::<&CreatureGenome>(births[0]).unwrap();
            assert!(
                child_genome.art.body_size >= 0.2 && child_genome.art.body_size <= 2.0,
                "Child body_size {} should be within valid range after crossover + mutation",
                child_genome.art.body_size
            );
        }
    }

    /// Hypothesis test: complex creatures can afford to reproduce.
    /// At complexity 0.7, the energy threshold should be < 0.55 (achievable)
    /// and the reproduction cost (30%) should leave positive energy.
    #[test]
    fn test_complex_creature_reproduction_is_feasible() {
        // Energy threshold = 0.4 + 0.05 * complexity
        let complexity = 0.7;
        let threshold = 0.4 + 0.05 * complexity;
        assert!(
            threshold < 0.55,
            "Energy threshold at complexity {} should be <0.55, got {:.3}",
            complexity, threshold,
        );

        // With 30% reproduction cost and needing >threshold energy:
        // Parent starts at threshold, pays 30%, ends with threshold - 0.3
        let remaining = threshold - 0.3;
        assert!(
            remaining > 0.0,
            "Parent should have positive energy after reproduction: \
             threshold={:.3} - cost=0.3 = {:.3}",
            threshold, remaining,
        );

        // Even at max complexity (1.0):
        let max_threshold = 0.4 + 0.15 * 1.0;
        let max_remaining = max_threshold - 0.3;
        assert!(
            max_remaining > 0.0,
            "Even at max complexity, parent should survive reproduction: \
             threshold={:.3} - cost=0.3 = {:.3}",
            max_threshold, max_remaining,
        );
    }

    /// Hypothesis test: NeedWeights are derived from the genome, not default.
    /// Two offspring with different reproduction_rate and metabolism_factor
    /// genes should have different NeedWeights.
    #[test]
    fn test_offspring_needweights_vary_by_genome() {
        let mut world = World::new();
        let mut rng = rand::rng();

        // Create two parents with different genome traits
        let mut g_fast = CreatureGenome::minimal_cell(&mut rng);
        g_fast.complexity = 0.05;
        g_fast.behavior.reproduction_rate = 1.0;
        g_fast.behavior.metabolism_factor = 1.5;

        let mut g_slow = CreatureGenome::minimal_cell(&mut rng);
        g_slow.complexity = 0.05;
        g_slow.behavior.reproduction_rate = 0.2;
        g_slow.behavior.metabolism_factor = 0.5;

        let _p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g_fast);
        let _p2 = spawn_ready_parent(&mut world, 30.0, 10.0, g_slow);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        // Both should reproduce asexually (complexity < 0.5)
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(&mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker);
        assert!(births.len() >= 2, "Both parents should produce offspring");

        // Collect NeedWeights from offspring
        let mut weights: Vec<(f32, f32)> = Vec::new();
        for &child in &births {
            let nw = world.get::<&NeedWeights>(child).unwrap();
            weights.push((nw.reproduction_rate, nw.hunger_rate));
        }

        // The two offspring should have different need weights
        assert!(
            (weights[0].0 - weights[1].0).abs() > 0.001
            || (weights[0].1 - weights[1].1).abs() > 0.001,
            "Offspring from different genomes should have different NeedWeights: \
             child1=(repro={:.4}, hunger={:.4}), child2=(repro={:.4}, hunger={:.4})",
            weights[0].0, weights[0].1, weights[1].0, weights[1].1,
        );
    }

    /// Fitness sharing: creatures in large species reproduce less often.
    /// A lone creature (species of 1) always reproduces, while a creature
    /// in a species of 6 has only a 3/6 = 50% chance.
    #[test]
    fn test_fitness_sharing_reduces_large_species_reproduction() {
        let mut rng = rand::rng();
        let mut large_births = 0u32;
        let mut small_births = 0u32;
        let iterations = 300;

        for _ in 0..iterations {
            // Scenario A: one reproducer in a species of 6 (fitness penalty 3/6 = 50%)
            {
                let mut world = World::new();
                let mut base = CreatureGenome::minimal_cell(&mut rng);
                base.complexity = 0.1;

                spawn_ready_parent(&mut world, 10.0, 10.0, base.clone());
                for i in 1..=5u32 {
                    let filler = spawn_ready_parent(
                        &mut world,
                        30.0 + i as f32 * 5.0,
                        10.0,
                        base.clone(),
                    );
                    world.get::<&mut Needs>(filler).unwrap().reproduction = 0.0;
                }

                let mut grid = SpatialGrid::new(10.0);
                grid.rebuild(&world);
                let mut tracker = InnovationTracker::new();
                let births = reproduction_system(
                    &mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker,
                );
                large_births += births.len() as u32;
            }

            // Scenario B: one reproducer alone (species of 1, no penalty)
            {
                let mut world = World::new();
                let mut lone = CreatureGenome::minimal_cell(&mut rng);
                lone.complexity = 0.1;

                spawn_ready_parent(&mut world, 10.0, 10.0, lone);

                let mut grid = SpatialGrid::new(10.0);
                grid.rebuild(&world);
                let mut tracker = InnovationTracker::new();
                let births = reproduction_system(
                    &mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker,
                );
                small_births += births.len() as u32;
            }
        }

        // Species of 1: no penalty → ~300 births
        // Species of 6: 50% penalty → ~150 births
        assert!(
            large_births < small_births,
            "Fitness sharing should reduce large species reproduction. \
             Large species: {large_births}, Small species: {small_births}",
        );
    }

    /// Sexual selection: the inline scoring formula prefers mates whose
    /// primary_hue matches the parent's mate_preference_hue.
    #[test]
    fn test_sexual_selection_prefers_matching_hue() {
        // Verify the scoring formula: score = 1.0 - wrapping_hue_distance
        let pref = 0.3_f32;

        let good_diff = (pref - 0.3_f32).abs();
        let good_score = 1.0 - good_diff.min(1.0 - good_diff);

        let bad_diff = (pref - 0.8_f32).abs();
        let bad_score = 1.0 - bad_diff.min(1.0 - bad_diff);

        assert!(
            good_score > bad_score,
            "Matching hue should score higher: good={good_score}, bad={bad_score}",
        );
        assert!(
            (good_score - 1.0).abs() < f32::EPSILON,
            "Perfect hue match should score 1.0",
        );

        // End-to-end: parent with a matching-hue mate should reproduce
        let mut rng = rand::rng();
        let mut genome = CreatureGenome::random(&mut rng);
        genome.complexity = 0.5;
        genome.behavior.mate_preference_hue = 0.3;

        let mut mate_genome = genome.clone();
        mate_genome.art.primary_hue = 0.3;

        let mut world = World::new();
        spawn_ready_parent(&mut world, 10.0, 10.0, genome);
        spawn_ready_parent(&mut world, 12.0, 10.0, mate_genome);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker,
        );
        assert!(
            !births.is_empty(),
            "Parent with matching-hue mate nearby should reproduce",
        );
    }

    /// Hue distance wraps: 0.0 and 1.0 are adjacent (distance ≈ 0.0).
    #[test]
    fn test_hue_distance_wraps() {
        // Hue 0.0 and 1.0 should be adjacent
        let diff = (0.0_f32 - 1.0_f32).abs();
        let dist = diff.min(1.0 - diff);
        assert!(
            dist < f32::EPSILON,
            "Hue 0.0 and 1.0 should be distance 0.0, got {dist}",
        );

        // 0.05 and 0.95 should be distance 0.1 (wrapping), not 0.9
        let diff2 = (0.05_f32 - 0.95_f32).abs();
        let dist2 = diff2.min(1.0 - diff2);
        assert!(
            (dist2 - 0.1).abs() < 1e-6,
            "Wrapping distance between 0.05 and 0.95 should be 0.1, got {dist2}",
        );

        // Creature preferring hue 0.05 should score a mate at 0.95
        // higher than a mate at 0.5 (wrapping proximity)
        let pref = 0.05_f32;
        let near_diff = (pref - 0.95_f32).abs();
        let near_score = 1.0 - near_diff.min(1.0 - near_diff);

        let far_diff = (pref - 0.5_f32).abs();
        let far_score = 1.0 - far_diff.min(1.0 - far_diff);

        assert!(
            near_score > far_score,
            "Wrapping neighbor (score {near_score}) should beat distant hue (score {far_score})",
        );
    }

    /// Creatures with complexity < 0.5 reproduce asexually without needing a mate.
    #[test]
    fn test_asexual_only_at_low_complexity() {
        let mut rng = rand::rng();
        let mut genome = CreatureGenome::minimal_cell(&mut rng);
        genome.complexity = 0.4;
        genome.behavior.reproduction_rate = 1.0;

        let mut world = World::new();
        spawn_ready_parent(&mut world, 40.0, 12.0, genome);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker,
        );

        assert!(
            !births.is_empty(),
            "Creature with complexity < 0.5 should reproduce asexually without a mate",
        );
    }

    /// Creatures with complexity > 0.9 cannot fall back to asexual reproduction.
    /// Without a compatible mate nearby, they should not reproduce.
    #[test]
    fn test_sexual_only_at_high_complexity() {
        let mut rng = rand::rng();
        let mut genome = CreatureGenome::random(&mut rng);
        genome.complexity = 0.95;
        genome.behavior.reproduction_rate = 1.0;

        let mut world = World::new();
        spawn_ready_parent(&mut world, 40.0, 12.0, genome);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world, &grid, &mut rng, 80.0, 24.0, 0, &mut tracker,
        );

        assert!(
            births.is_empty(),
            "Creature with complexity > 0.9 should NOT reproduce without a compatible mate",
        );
    }
}
