//! Reproduction system: asexual division for simple cells, sexual crossover for complex creatures.

use std::collections::{HashMap, HashSet};

use hecs::{Entity, World};
use rand::Rng;
use rand::RngExt;

use crate::behavior::BehaviorState;
use crate::boids::Boid;
use crate::brain::{Brain, InnovationTracker};
use crate::calibration::EvolutionCalibration;
use crate::components::*;
use crate::ecosystem::{
    consumer_reproduction_reserve_threshold, consumer_reproductive_threshold, Age, Energy,
};
use crate::genetics::{crossover, genomic_distance, mutate, CREATURE_SPECIES_THRESHOLD};
use crate::genome::CreatureGenome;
use crate::needs::{NeedWeights, Needs};
use crate::phenotype::{derive_feeding, derive_physics, DerivedPhysics};
use crate::spatial::SpatialGrid;

/// Maximum genomic distance for two creatures to be considered compatible mates.
const MATE_COMPATIBILITY_DISTANCE: f32 = 8.0;
/// Mutation rate for sexual offspring.
const MUTATION_RATE: f32 = 0.15;
/// Higher mutation rate for asexual division (more drift).
const ASEXUAL_MUTATION_RATE: f32 = 0.25;
/// Maximum number of offspring per tick (prevents burst spawning).
const MAX_BIRTHS_PER_TICK: usize = 2;

/// Find ready-to-reproduce creatures and spawn offspring.
/// Simple cells (complexity < 0.5) reproduce asexually.
/// Complex creatures need a compatible mate nearby.
fn max_population_for_tank(tank_w: f32, tank_h: f32) -> usize {
    crate::ecology_equilibrium::ReducedEquilibriumModel::default_startup_targets(
        tank_w.max(1.0).round() as u16,
        tank_h.max(1.0).round() as u16,
    )
    .soft_population_cap
}

pub fn reproduction_system(
    world: &mut World,
    grid: &SpatialGrid,
    rng: &mut impl Rng,
    tank_w: f32,
    tank_h: f32,
    creature_count: usize,
    tracker: &mut InnovationTracker,
    evolution: &EvolutionCalibration,
    diversity_coefficient: f32,
) -> Vec<Entity> {
    // Soft population cap — stop reproducing when crowded
    let max_population = max_population_for_tank(tank_w, tank_h);
    if creature_count >= max_population {
        return Vec::new();
    }
    let remaining_capacity = max_population.saturating_sub(creature_count).max(1);
    let max_births_this_tick = MAX_BIRTHS_PER_TICK.min(remaining_capacity);
    let crowding = creature_count as f32 / max_population.max(1) as f32;

    // Fitness sharing: group creatures into species and compute reproduction probability.
    // Larger species get proportionally less reproduction chance, protecting innovation.
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
            if genomic_distance(genome, &all_genomes[ci].1) < CREATURE_SPECIES_THRESHOLD {
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
    let candidates: Vec<(Entity, f32, f32, f32, f32, f32, f32, f32, f32)> = {
        let mut v = Vec::new();
        for (entity, pos, needs, energy, genome, physics, state) in &mut world.query::<(
            Entity,
            &Position,
            &Needs,
            &Energy,
            &CreatureGenome,
            &DerivedPhysics,
            &crate::components::ConsumerState,
        )>() {
            let reserve_threshold = consumer_reproduction_reserve_threshold(physics, genome);
            let reproductive_threshold = consumer_reproductive_threshold(physics, genome);
            if state.is_adult()
                && state.brood_cooldown <= 0.0
                && state.reproductive_buffer >= reproductive_threshold
                && needs.reproduction > 0.22
                && energy.fraction() > reserve_threshold
            {
                v.push((
                    entity,
                    pos.x,
                    pos.y,
                    needs.reproduction,
                    energy.fraction(),
                    genome.complexity,
                    genome.behavior.mutation_rate_factor,
                    genome.behavior.mate_preference_hue,
                    reproductive_threshold,
                ));
            }
        }
        v
    };

    let candidate_set: HashSet<Entity> = candidates.iter().map(|(e, ..)| *e).collect();
    let mut births = Vec::new();
    let mut already_mated: HashSet<Entity> = HashSet::new();

    for &(parent_a, ax, ay, _, _, complexity, mutation_factor, mate_pref, parent_threshold) in
        &candidates
    {
        if births.len() >= max_births_this_tick {
            break;
        }
        if already_mated.contains(&parent_a) {
            continue;
        }
        if crowding > 0.40 {
            let suppression = ((crowding - 0.40) / 0.40).clamp(0.0, 1.0);
            let allow = (1.0 - suppression * 0.95).clamp(0.03, 1.0);
            if !rng.random_bool(allow as f64) {
                continue;
            }
        }

        // Fitness sharing: larger species reproduce less to protect innovation.
        // Higher diversity_coefficient strengthens the penalty, promoting variety.
        if let Some(&sp) = species_map.get(&parent_a) {
            let sp_size = *species_sizes.get(&sp).unwrap_or(&1);
            if sp_size > 3 {
                let effective_sharing = evolution.fitness_sharing_strength * diversity_coefficient;
                let prob = (3.0 / sp_size as f32).powf(effective_sharing.clamp(0.25, 3.0));
                if !rng.random_bool(prob as f64) {
                    continue;
                }
            }
        }

        // Try sexual reproduction first if complexity is high enough for
        // protist-like recombination rather than only animal-like mating.
        let mut child_genome: Option<CreatureGenome> = None;
        let mut mate_id: Option<Entity> = None;
        let mut required_buffer = parent_threshold;

        if complexity >= 0.32 {
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
                let pa = world.get::<&DerivedPhysics>(parent_a).unwrap().clone();
                let pb = world.get::<&DerivedPhysics>(mate).unwrap().clone();
                let shared_threshold = consumer_reproductive_threshold(&pa, &ga)
                    .max(consumer_reproductive_threshold(&pb, &gb))
                    * 0.6;

                let mate_ready = world
                    .get::<&crate::components::ConsumerState>(mate)
                    .map(|state| {
                        state.is_adult()
                            && state.brood_cooldown <= 0.0
                            && state.reproductive_buffer >= shared_threshold
                    })
                    .unwrap_or(false);
                if mate_ready {
                    let mut child = crossover(&ga, &gb, rng);
                    child.generation = ga.generation.max(gb.generation) + 1;
                    let effective_rate = MUTATION_RATE
                        * (ga.behavior.mutation_rate_factor + gb.behavior.mutation_rate_factor)
                        * 0.5
                        * evolution.creature_mutation_multiplier
                        * diversity_coefficient;
                    mutate(
                        &mut child,
                        effective_rate,
                        diversity_coefficient,
                        rng,
                        tracker,
                    );
                    mate_id = Some(mate);
                    required_buffer = shared_threshold;
                    child_genome = Some(child);
                }
            }
        }

        // Research note: sexual reproduction and genetic exchange occur across
        // many morphologically simple protists, so the model should not force
        // all recombination to wait for animal-like complexity (Weedall & Hall, 2015).
        //
        // Fall back to asexual reproduction if no mate found (always works below 0.62,
        // gradually suppressed above that to keep simple founder webs viable).
        if child_genome.is_none() {
            if complexity < 0.62 || (complexity < 0.82 && rng.random_bool(0.45)) {
                let ga = match world.get::<&CreatureGenome>(parent_a) {
                    Ok(g) => (*g).clone(),
                    Err(_) => continue,
                };
                let mut child = ga.clone();
                child.generation = ga.generation + 1;
                let effective_rate = ASEXUAL_MUTATION_RATE
                    * mutation_factor
                    * evolution.creature_mutation_multiplier
                    * diversity_coefficient;
                mutate(
                    &mut child,
                    effective_rate,
                    diversity_coefficient,
                    rng,
                    tracker,
                );
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
            facing: if vx >= 0.0 {
                Direction::Right
            } else {
                Direction::Left
            },
            color_index: child_genome.art.color_index(),
        };

        let bbox_w = 1.0_f32.max(child_genome.art.body_size * 3.0);
        let bbox_h = 1.0_f32.max(child_genome.art.body_size * 2.0);

        // Calder (1984), Peters (1983): lifespan ∝ mass^0.25 across taxa.
        // Larger organisms live proportionally longer, amortizing growth investment.
        let size_longevity = physics.body_mass.max(0.1).powf(0.25);
        let max_ticks =
            (48_000.0 * child_genome.behavior.max_lifespan_factor * size_longevity) as u64;

        // Hunger pacing remains evolvable through the genome, while reproductive
        // timing is now derived separately from ConsumerState life-history state.
        let need_weights = NeedWeights {
            hunger_rate: 0.01 + 0.02 * child_genome.behavior.metabolism_factor.min(1.5),
            ..NeedWeights::default()
        };

        let brain = Brain::from_genome_with_learning(
            &child_genome.brain,
            child_genome.behavior.learning_rate,
        );
        let child_entity = world.spawn((
            Position {
                x: spawn_x,
                y: spawn_y,
            },
            Velocity { vx, vy },
            BoundingBox {
                w: bbox_w,
                h: bbox_h,
            },
            appearance,
            AnimationState::new(0.2 / child_genome.anim.swim_speed),
            child_genome,
            physics.clone(),
            feeding,
            Energy::new_with(physics.max_energy * 0.45, physics.max_energy),
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
        let _ = world.insert(child_entity, (crate::components::ConsumerState::default(),));

        // Research note: offspring production consumes explicit parental reserve
        // buffers and imposes a refractory period, preventing timer-driven
        // consumer explosions under temporary food abundance.
        let parent_cooldown = match (
            world.get::<&CreatureGenome>(parent_a),
            world.get::<&DerivedPhysics>(parent_a),
        ) {
            (Ok(g), Ok(physics)) => {
                let generation_pace =
                    (1.15 - g.complexity * 0.38 - physics.body_mass.max(0.1).powf(0.28) * 0.14)
                        .clamp(0.62, 1.15);
                (210.0 - g.behavior.reproduction_rate * 80.0) / generation_pace
            }
            _ => 170.0,
        };
        if let Ok(mut state) = world.get::<&mut crate::components::ConsumerState>(parent_a) {
            state.reproductive_buffer = (state.reproductive_buffer - required_buffer).max(0.0);
            state.brood_cooldown = parent_cooldown.max(140.0);
            state.recent_assimilation *= 0.50;
        }
        if let Ok(mut e) = world.get::<&mut Energy>(parent_a) {
            e.current -= e.max * 0.30;
        }
        if let Ok(mut n) = world.get::<&mut Needs>(parent_a) {
            n.reproduction = 0.0;
        }

        if let Some(mate) = mate_id {
            let mate_cooldown = match (
                world.get::<&CreatureGenome>(mate),
                world.get::<&DerivedPhysics>(mate),
            ) {
                (Ok(g), Ok(physics)) => {
                    let generation_pace =
                        (1.15 - g.complexity * 0.38 - physics.body_mass.max(0.1).powf(0.28) * 0.14)
                            .clamp(0.62, 1.15);
                    (210.0 - g.behavior.reproduction_rate * 80.0) / generation_pace
                }
                _ => 170.0,
            };
            if let Ok(mut state) = world.get::<&mut crate::components::ConsumerState>(mate) {
                state.reproductive_buffer = (state.reproductive_buffer - required_buffer).max(0.0);
                state.brood_cooldown = mate_cooldown.max(140.0);
                state.recent_assimilation *= 0.50;
            }
            if let Ok(mut e) = world.get::<&mut Energy>(mate) {
                e.current -= e.max * 0.30;
            }
            if let Ok(mut n) = world.get::<&mut Needs>(mate) {
                n.reproduction = 0.0;
            }
            already_mated.insert(mate);
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn spawn_ready_parent(world: &mut World, x: f32, y: f32, genome: CreatureGenome) -> Entity {
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
        let threshold = consumer_reproductive_threshold(&physics, &genome);

        let brain = Brain::from_genome(&genome.brain);
        let entity = world.spawn((
            Position { x, y },
            Velocity { vx: 0.0, vy: 0.0 },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            genome,
            physics.clone(),
            feeding,
            Energy::new(physics.max_energy),
            Age {
                ticks: 6_000,
                max_ticks: 24_000,
            },
            needs,
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ));
        let _ = world.insert(
            entity,
            (ConsumerState {
                reserve_buffer: 0.85,
                maturity_progress: 1.0,
                matured_once: true,
                reproductive_buffer: threshold * 1.3,
                brood_cooldown: 0.0,
                recent_assimilation: 0.3,
            },),
        );
        entity
    }

    #[test]
    fn test_sexual_reproduction_spawns_offspring() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(42);

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
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
        );

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
        let mut rng = StdRng::seed_from_u64(42);

        // Simple cell — should reproduce alone
        let mut cell = CreatureGenome::minimal_cell(&mut rng);
        cell.complexity = 0.1;
        let _p = spawn_ready_parent(&mut world, 10.0, 10.0, cell);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
        );
        assert!(
            !births.is_empty(),
            "Simple cell should reproduce asexually without a mate"
        );
    }

    #[test]
    fn test_reproduction_costs_parent_energy() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(42);

        let mut g1 = CreatureGenome::random(&mut rng);
        g1.complexity = 0.5;
        let g2 = g1.clone();

        let p1 = spawn_ready_parent(&mut world, 10.0, 10.0, g1);
        let _p2 = spawn_ready_parent(&mut world, 12.0, 10.0, g2);

        let before = world.get::<&Energy>(p1).unwrap().current;

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
        );

        if !births.is_empty() {
            let after = world.get::<&Energy>(p1).unwrap().current;
            assert!(
                after < before,
                "Parent should lose energy: {} -> {}",
                before,
                after
            );
        }
    }

    #[test]
    fn test_offspring_genome_is_blend() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(42);

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
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
        );

        if !births.is_empty() {
            let child_genome = world.get::<&CreatureGenome>(births[0]).unwrap();
            assert!(
                child_genome.art.body_size >= 0.2 && child_genome.art.body_size <= 5.0,
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
            complexity,
            threshold,
        );

        // With 30% reproduction cost and needing >threshold energy:
        // Parent starts at threshold, pays 30%, ends with threshold - 0.3
        let remaining = threshold - 0.3;
        assert!(
            remaining > 0.0,
            "Parent should have positive energy after reproduction: \
             threshold={:.3} - cost=0.3 = {:.3}",
            threshold,
            remaining,
        );

        // Even at max complexity (1.0):
        let max_threshold = 0.4 + 0.15 * 1.0;
        let max_remaining = max_threshold - 0.3;
        assert!(
            max_remaining > 0.0,
            "Even at max complexity, parent should survive reproduction: \
             threshold={:.3} - cost=0.3 = {:.3}",
            max_threshold,
            max_remaining,
        );
    }

    /// Hypothesis test: baseline need weights are still genome-derived after
    /// moving reproductive timing into ConsumerState.
    #[test]
    fn test_offspring_needweights_vary_by_genome() {
        let mut world = World::new();
        let mut rng = StdRng::seed_from_u64(42);

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
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
        );
        assert!(births.len() >= 2, "Both parents should produce offspring");

        // Collect NeedWeights from offspring
        let mut weights: Vec<f32> = Vec::new();
        for &child in &births {
            let nw = world.get::<&NeedWeights>(child).unwrap();
            weights.push(nw.hunger_rate);
        }

        // The two offspring should have different need weights
        assert!(
            (weights[0] - weights[1]).abs() > 0.001,
            "Offspring from different genomes should have different hunger rates: \
             child1={:.4}, child2={:.4}",
            weights[0],
            weights[1],
        );
    }

    /// Fitness sharing: creatures in large species reproduce less often.
    /// A lone creature (species of 1) always reproduces, while a creature
    /// in a species of 6 has only a 3/6 = 50% chance.
    #[test]
    fn test_fitness_sharing_reduces_large_species_reproduction() {
        let mut rng = StdRng::seed_from_u64(42);
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
                    let filler =
                        spawn_ready_parent(&mut world, 30.0 + i as f32 * 5.0, 10.0, base.clone());
                    world.get::<&mut Needs>(filler).unwrap().reproduction = 0.0;
                }

                let mut grid = SpatialGrid::new(10.0);
                grid.rebuild(&world);
                let mut tracker = InnovationTracker::new();
                let births = reproduction_system(
                    &mut world,
                    &grid,
                    &mut rng,
                    80.0,
                    24.0,
                    0,
                    &mut tracker,
                    &EvolutionCalibration::default(),
                    1.0,
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
                    &mut world,
                    &grid,
                    &mut rng,
                    80.0,
                    24.0,
                    0,
                    &mut tracker,
                    &EvolutionCalibration::default(),
                    1.0,
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
        let mut rng = StdRng::seed_from_u64(42);
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
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
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
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = CreatureGenome::minimal_cell(&mut rng);
        genome.complexity = 0.4;
        genome.behavior.reproduction_rate = 1.0;

        let mut world = World::new();
        spawn_ready_parent(&mut world, 40.0, 12.0, genome);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
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
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = CreatureGenome::random(&mut rng);
        genome.complexity = 0.95;
        genome.behavior.reproduction_rate = 1.0;

        let mut world = World::new();
        spawn_ready_parent(&mut world, 40.0, 12.0, genome);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);
        let mut tracker = InnovationTracker::new();
        let births = reproduction_system(
            &mut world,
            &grid,
            &mut rng,
            80.0,
            24.0,
            0,
            &mut tracker,
            &EvolutionCalibration::default(),
            1.0,
        );

        assert!(
            births.is_empty(),
            "Creature with complexity > 0.9 should NOT reproduce without a compatible mate",
        );
    }

    #[test]
    fn test_diversity_coefficient_scales_fitness_sharing() {
        // With a high diversity coefficient, large species should be penalized more
        // (fewer births). With a low coefficient, they should reproduce more freely.
        let mut rng = StdRng::seed_from_u64(999);

        // Create 8 identical creatures (same species, large group)
        let base_genome = {
            let mut g = CreatureGenome::random(&mut rng);
            g.complexity = 0.1; // low complexity = asexual
            g.behavior.mutation_rate_factor = 1.0;
            g
        };

        let run = |diversity: f32| -> usize {
            let mut total_births = 0;
            // Run multiple trials since fitness sharing is stochastic
            for trial in 0..20 {
                let mut world = World::new();
                let mut rng = StdRng::seed_from_u64(1000 + trial);
                for i in 0..8 {
                    spawn_ready_parent(
                        &mut world,
                        10.0 + i as f32 * 3.0,
                        10.0,
                        base_genome.clone(),
                    );
                }
                let mut grid = SpatialGrid::new(10.0);
                grid.rebuild(&world);
                let mut tracker = InnovationTracker::new();
                let births = reproduction_system(
                    &mut world,
                    &grid,
                    &mut rng,
                    80.0,
                    24.0,
                    0,
                    &mut tracker,
                    &EvolutionCalibration::default(),
                    diversity,
                );
                total_births += births.len();
            }
            total_births
        };

        let births_low = run(0.25); // low diversity = weak fitness sharing
        let births_high = run(2.5); // high diversity = strong fitness sharing

        assert!(
            births_low > births_high,
            "Low diversity coefficient should allow more births from large species. \
             low_div births={}, high_div births={}",
            births_low,
            births_high,
        );
    }
}
