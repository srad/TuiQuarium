//! Genetic operations: crossover, mutation, speciation.
//! All genes are continuous floats — mutations are always gradual perturbations.

use rand::Rng;
use rand::RngExt;

use crate::brain::{self, InnovationTracker};
use crate::genome::*;

/// Genomic distance threshold used for creature species clustering.
///
/// Simulation assumption: the founder web needs a threshold tight enough that
/// ecologically distinct low-complexity consumer strategies are not merged into
/// one species cloud, while still requiring multiple trait shifts before a new
/// lineage is counted.
pub const CREATURE_SPECIES_THRESHOLD: f32 = 1.5;

/// Uniform crossover: each gene randomly from parent A or B.
pub fn crossover(a: &CreatureGenome, b: &CreatureGenome, rng: &mut impl Rng) -> CreatureGenome {
    CreatureGenome {
        art: crossover_art(&a.art, &b.art, rng),
        anim: crossover_anim(&a.anim, &b.anim, rng),
        behavior: crossover_behavior(&a.behavior, &b.behavior, rng),
        brain: brain::crossover_brain(&a.brain, &b.brain, rng),
        complexity: pick_f32(a.complexity, b.complexity, rng),
        generation: a.generation.max(b.generation) + 1,
    }
}

fn pick_f32(a: f32, b: f32, rng: &mut impl Rng) -> f32 {
    if rng.random_bool(0.5) {
        a
    } else {
        b
    }
}

fn crossover_art(a: &ArtGenome, b: &ArtGenome, rng: &mut impl Rng) -> ArtGenome {
    ArtGenome {
        body_elongation: pick_f32(a.body_elongation, b.body_elongation, rng),
        body_height_ratio: pick_f32(a.body_height_ratio, b.body_height_ratio, rng),
        body_size: pick_f32(a.body_size, b.body_size, rng),
        tail_fork: pick_f32(a.tail_fork, b.tail_fork, rng),
        tail_length: pick_f32(a.tail_length, b.tail_length, rng),
        top_appendage: pick_f32(a.top_appendage, b.top_appendage, rng),
        side_appendages: pick_f32(a.side_appendages, b.side_appendages, rng),
        pattern_density: pick_f32(a.pattern_density, b.pattern_density, rng),
        eye_size: pick_f32(a.eye_size, b.eye_size, rng),
        primary_hue: pick_f32(a.primary_hue, b.primary_hue, rng),
        secondary_hue: pick_f32(a.secondary_hue, b.secondary_hue, rng),
        color_brightness: pick_f32(a.color_brightness, b.color_brightness, rng),
    }
}

fn crossover_anim(a: &AnimGenome, b: &AnimGenome, rng: &mut impl Rng) -> AnimGenome {
    AnimGenome {
        swim_speed: pick_f32(a.swim_speed, b.swim_speed, rng),
        tail_amplitude: pick_f32(a.tail_amplitude, b.tail_amplitude, rng),
        idle_sway: pick_f32(a.idle_sway, b.idle_sway, rng),
        undulation: pick_f32(a.undulation, b.undulation, rng),
    }
}

fn crossover_behavior(
    a: &BehaviorGenome,
    b: &BehaviorGenome,
    rng: &mut impl Rng,
) -> BehaviorGenome {
    BehaviorGenome {
        schooling_affinity: pick_f32(a.schooling_affinity, b.schooling_affinity, rng),
        aggression: pick_f32(a.aggression, b.aggression, rng),
        timidity: pick_f32(a.timidity, b.timidity, rng),
        speed_factor: pick_f32(a.speed_factor, b.speed_factor, rng),
        metabolism_factor: pick_f32(a.metabolism_factor, b.metabolism_factor, rng),
        max_lifespan_factor: pick_f32(a.max_lifespan_factor, b.max_lifespan_factor, rng),
        reproduction_rate: pick_f32(a.reproduction_rate, b.reproduction_rate, rng),
        mouth_size: pick_f32(a.mouth_size, b.mouth_size, rng),
        hunting_instinct: pick_f32(a.hunting_instinct, b.hunting_instinct, rng),
        mutation_rate_factor: pick_f32(a.mutation_rate_factor, b.mutation_rate_factor, rng),
        mate_preference_hue: pick_f32(a.mate_preference_hue, b.mate_preference_hue, rng),
        learning_rate: pick_f32(a.learning_rate, b.learning_rate, rng),
        pheromone_sensitivity: pick_f32(a.pheromone_sensitivity, b.pheromone_sensitivity, rng),
    }
}

/// Mutate a genome in-place. Each float gene has `rate` probability of being perturbed.
pub fn mutate(
    genome: &mut CreatureGenome,
    rate: f32,
    diversity: f32,
    rng: &mut impl Rng,
    tracker: &mut InnovationTracker,
) {
    mutate_art(&mut genome.art, rate, rng);
    mutate_anim(&mut genome.anim, rate, rng);
    mutate_behavior(&mut genome.behavior, rate, rng);
    // Compute body mass to determine brain capacity (Jerison 1973)
    let plan_mass_factor = 0.6 + 0.6 * (1.0 - genome.art.body_elongation);
    let body_mass = genome.art.body_size * genome.art.body_size * plan_mass_factor;
    let effective_cap = brain::effective_max_nodes(body_mass);
    brain::mutate_brain(&mut genome.brain, rate, diversity, effective_cap, rng, tracker);
    // Complexity ALWAYS mutates — it's the master gene driving morphological evolution.
    // Gating it behind per-gene rate (~15%) made exploration too slow for complexity
    // to increase meaningfully within simulation timescales.
    let delta: f32 = rng.random_range(-0.15..0.15);
    genome.complexity = (genome.complexity + delta).clamp(0.0, 1.0);
}

fn perturb(val: &mut f32, min: f32, max: f32, rate: f32, rng: &mut impl Rng) {
    if rng.random_bool(rate as f64) {
        let delta: f32 = rng.random_range(-0.1..0.1);
        *val = (*val + delta * (max - min)).clamp(min, max);
    }
}

fn mutate_art(art: &mut ArtGenome, rate: f32, rng: &mut impl Rng) {
    perturb(&mut art.body_elongation, 0.0, 1.0, rate, rng);
    perturb(&mut art.body_height_ratio, 0.0, 1.0, rate, rng);
    perturb(&mut art.body_size, 0.2, 5.0, rate, rng);
    perturb(&mut art.tail_fork, 0.0, 1.0, rate, rng);
    perturb(&mut art.tail_length, 0.0, 1.5, rate, rng);
    perturb(&mut art.top_appendage, 0.0, 1.0, rate, rng);
    perturb(&mut art.side_appendages, 0.0, 1.0, rate, rng);
    perturb(&mut art.pattern_density, 0.0, 1.0, rate, rng);
    perturb(&mut art.eye_size, 0.0, 1.0, rate, rng);
    perturb(&mut art.primary_hue, 0.0, 1.0, rate, rng);
    perturb(&mut art.secondary_hue, 0.0, 1.0, rate, rng);
    perturb(&mut art.color_brightness, 0.0, 1.0, rate, rng);
}

fn mutate_anim(anim: &mut AnimGenome, rate: f32, rng: &mut impl Rng) {
    perturb(&mut anim.swim_speed, 0.3, 2.0, rate, rng);
    perturb(&mut anim.tail_amplitude, 0.0, 1.0, rate, rng);
    perturb(&mut anim.idle_sway, 0.0, 1.0, rate, rng);
    perturb(&mut anim.undulation, 0.0, 1.0, rate, rng);
}

fn mutate_behavior(beh: &mut BehaviorGenome, rate: f32, rng: &mut impl Rng) {
    perturb(&mut beh.schooling_affinity, 0.0, 1.0, rate, rng);
    perturb(&mut beh.aggression, 0.0, 1.0, rate, rng);
    perturb(&mut beh.timidity, 0.0, 1.0, rate, rng);
    perturb(&mut beh.speed_factor, 0.5, 2.0, rate, rng);
    perturb(&mut beh.metabolism_factor, 0.5, 2.0, rate, rng);
    perturb(&mut beh.max_lifespan_factor, 0.5, 2.0, rate, rng);
    perturb(&mut beh.reproduction_rate, 0.2, 1.0, rate, rng);
    perturb(&mut beh.mouth_size, 0.0, 1.0, rate, rng);
    perturb(&mut beh.hunting_instinct, 0.0, 1.0, rate, rng);
    // Mutation rate factor mutates at a fixed low rate (independent of itself)
    perturb(&mut beh.mutation_rate_factor, 0.5, 2.0, 0.05, rng);
    perturb(&mut beh.mate_preference_hue, 0.0, 1.0, rate, rng);
    perturb(&mut beh.learning_rate, 0.0, 0.1, rate, rng);
    perturb(&mut beh.pheromone_sensitivity, 0.0, 1.0, rate, rng);
}

/// Genomic distance for speciation — how different two genomes are.
/// Pure float distance — no discrete jumps.
pub fn genomic_distance(a: &CreatureGenome, b: &CreatureGenome) -> f32 {
    let mut dist = 0.0_f32;

    // Art genes
    dist += (a.art.body_elongation - b.art.body_elongation).abs();
    dist += (a.art.body_height_ratio - b.art.body_height_ratio).abs();
    dist += (a.art.body_size - b.art.body_size).abs();
    dist += (a.art.tail_length - b.art.tail_length).abs();
    dist += (a.art.color_brightness - b.art.color_brightness).abs();
    dist += (a.art.top_appendage - b.art.top_appendage).abs();
    dist += (a.art.side_appendages - b.art.side_appendages).abs();

    // Behavior genes
    dist += (a.behavior.schooling_affinity - b.behavior.schooling_affinity).abs();
    dist += (a.behavior.aggression - b.behavior.aggression).abs();
    dist += (a.behavior.speed_factor - b.behavior.speed_factor).abs();
    dist += (a.behavior.mouth_size - b.behavior.mouth_size).abs();
    dist += (a.behavior.hunting_instinct - b.behavior.hunting_instinct).abs();

    // Complexity
    dist += (a.complexity - b.complexity).abs();

    // Brain weight distance (scaled down to prevent instant speciation)
    dist += brain::brain_distance(&a.brain, &b.brain) * 0.5;

    dist
}

/// Mutate a producer genome in-place. Each float gene has `rate` probability of being perturbed.
/// Complexity always mutates (like creature genomes). Producers reproduce asexually only —
/// no crossover. Mutation is the sole source of genetic variation.
///
/// Gene ranges enforce ecological constraints: physiology genes (0.3–2.0) allow trade-offs
/// between Grime's C-S-R strategies; seed_count vs seed_size encodes r/K selection.
pub fn mutate_producer(genome: &mut ProducerGenome, rate: f32, rng: &mut impl Rng) {
    // Morphology
    perturb(&mut genome.stem_thickness, 0.0, 1.0, rate, rng);
    perturb(&mut genome.height_factor, 0.0, 1.0, rate, rng);
    perturb(&mut genome.leaf_area, 0.0, 1.0, rate, rng);
    perturb(&mut genome.branching, 0.0, 1.0, rate, rng);
    perturb(&mut genome.curvature, 0.0, 1.0, rate, rng);
    perturb(&mut genome.primary_hue, 0.0, 1.0, rate, rng);

    // Physiology
    perturb(&mut genome.photosynthesis_rate, 0.5, 2.0, rate, rng);
    perturb(&mut genome.max_energy_factor, 0.5, 2.0, rate, rng);
    perturb(&mut genome.hardiness, 0.0, 1.0, rate, rng);
    perturb(&mut genome.seed_range, 0.5, 2.0, rate, rng);
    perturb(&mut genome.seed_count, 0.3, 2.0, rate, rng);
    perturb(&mut genome.seed_size, 0.3, 2.0, rate, rng);
    perturb(&mut genome.lifespan_factor, 0.5, 2.0, rate, rng);
    perturb(&mut genome.nutritional_value, 0.3, 1.5, rate, rng);
    perturb(&mut genome.clonal_spread, 0.0, 1.0, rate, rng);
    perturb(&mut genome.nutrient_affinity, 0.3, 1.5, rate, rng);
    perturb(&mut genome.epiphyte_resistance, 0.0, 1.0, rate, rng);
    perturb(&mut genome.reserve_allocation, 0.0, 1.0, rate, rng);

    // Mutation rate factor mutates slowly (independent of itself)
    perturb(&mut genome.mutation_rate_factor, 0.5, 2.0, 0.05, rng);

    // Complexity always mutates
    let delta: f32 = rng.random_range(-0.15..0.15);
    genome.complexity = (genome.complexity + delta).clamp(0.0, 1.0);
}

/// Genomic distance between two producer colonies for speciation analysis.
pub fn producer_genomic_distance(a: &ProducerGenome, b: &ProducerGenome) -> f32 {
    let mut dist = 0.0_f32;
    dist += (a.stem_thickness - b.stem_thickness).abs();
    dist += (a.height_factor - b.height_factor).abs();
    dist += (a.leaf_area - b.leaf_area).abs();
    dist += (a.branching - b.branching).abs();
    dist += (a.curvature - b.curvature).abs();
    dist += (a.photosynthesis_rate - b.photosynthesis_rate).abs();
    dist += (a.max_energy_factor - b.max_energy_factor).abs();
    dist += (a.hardiness - b.hardiness).abs();
    dist += (a.nutritional_value - b.nutritional_value).abs();
    dist += (a.clonal_spread - b.clonal_spread).abs();
    dist += (a.nutrient_affinity - b.nutrient_affinity).abs();
    dist += (a.epiphyte_resistance - b.epiphyte_resistance).abs();
    dist += (a.reserve_allocation - b.reserve_allocation).abs();
    dist += (a.complexity - b.complexity).abs();
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_crossover_produces_valid_genome() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = CreatureGenome::random(&mut rng);
            let b = CreatureGenome::random(&mut rng);
            let child = crossover(&a, &b, &mut rng);

            assert!(child.art.body_size >= 0.2 && child.art.body_size <= 5.0);
            assert!(child.behavior.speed_factor >= 0.5 && child.behavior.speed_factor <= 2.0);
            assert!(child.complexity >= 0.0 && child.complexity <= 1.0);
            assert!(child.generation > 0, "Child should have generation > 0");
        }
    }

    #[test]
    fn test_mutation_stays_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();
        for _ in 0..200 {
            let mut g = CreatureGenome::random(&mut rng);
            mutate(&mut g, 0.5, 1.0, &mut rng, &mut tracker);

            assert!(
                g.art.body_size >= 0.2 && g.art.body_size <= 5.0,
                "body_size out of range: {}",
                g.art.body_size
            );
            assert!(g.art.body_elongation >= 0.0 && g.art.body_elongation <= 1.0);
            assert!(g.art.tail_length >= 0.0 && g.art.tail_length <= 1.5);
            assert!(g.behavior.speed_factor >= 0.5 && g.behavior.speed_factor <= 2.0);
            assert!(g.behavior.schooling_affinity >= 0.0 && g.behavior.schooling_affinity <= 1.0);
            assert!(g.behavior.mouth_size >= 0.0 && g.behavior.mouth_size <= 1.0);
            assert!(g.anim.swim_speed >= 0.3 && g.anim.swim_speed <= 2.0);
            assert!(g.complexity >= 0.0 && g.complexity <= 1.0);
        }
    }

    #[test]
    fn test_genomic_distance_self_is_zero() {
        let g = CreatureGenome::random(&mut StdRng::seed_from_u64(42));
        assert!((genomic_distance(&g, &g)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_genomic_distance_different_is_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = CreatureGenome::random(&mut rng);
        let mut b = a.clone();
        b.art.body_elongation = 1.0 - a.art.body_elongation; // flip elongation
        b.behavior.mouth_size = 1.0 - a.behavior.mouth_size; // flip mouth
        assert!(genomic_distance(&a, &b) > 0.0);
    }

    #[test]
    fn test_mutation_eventually_changes_genome() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();
        let original = CreatureGenome::random(&mut rng);
        let mut mutated = original.clone();
        for _ in 0..100 {
            mutate(&mut mutated, 0.8, 1.0, &mut rng, &mut tracker);
        }
        assert!(
            genomic_distance(&original, &mutated) > 0.0,
            "Heavy mutation should change the genome"
        );
    }

    /// Hypothesis test: complexity drifts upward via asexual mutation.
    /// Starting from minimal cells (complexity 0.0–0.1), repeated asexual
    /// reproduction with mutation should produce individuals above 0.3
    /// within 50 generations. This proves the ±0.15 mutation step is
    /// large enough to escape the primordial complexity range.
    #[test]
    fn test_complexity_drifts_upward_via_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();
        let mut max_complexity: f32 = 0.0;

        for _ in 0..20 {
            let mut genome = CreatureGenome::minimal_cell(&mut rng);
            assert!(genome.complexity <= 0.1, "Starts simple");

            for _ in 0..50 {
                genome.generation += 1;
                mutate(&mut genome, 0.25, 1.0, &mut rng, &mut tracker);
            }

            max_complexity = max_complexity.max(genome.complexity);
        }

        assert!(
            max_complexity > 0.3,
            "At least one lineage out of 20 should drift above 0.3 \
             complexity in 50 generations, got max {:.3}",
            max_complexity,
        );
    }

    /// Hypothesis test: brain divergence no longer breaks mate compatibility.
    /// Two lineages diverging independently for 5 generations should still
    /// have genomic distance under MATE_COMPATIBILITY_DISTANCE (8.0).
    /// This proves the brain distance weight reduction (2.0→0.5) works.
    #[test]
    fn test_brain_divergence_preserves_mate_compatibility() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();
        let mate_threshold = 8.0;

        let mut compatible_count = 0;
        let trials = 50;

        for _ in 0..trials {
            let ancestor = CreatureGenome::minimal_cell(&mut rng);
            let mut lineage_a = ancestor.clone();
            let mut lineage_b = ancestor.clone();

            for _ in 0..5 {
                mutate(&mut lineage_a, 0.25, 1.0, &mut rng, &mut tracker);
                mutate(&mut lineage_b, 0.25, 1.0, &mut rng, &mut tracker);
            }

            let dist = genomic_distance(&lineage_a, &lineage_b);
            if dist < mate_threshold {
                compatible_count += 1;
            }
        }

        assert!(
            compatible_count > trials / 2,
            "Most lineages diverging 5 generations should remain compatible. \
             Only {}/{} were compatible (distance < {})",
            compatible_count,
            trials,
            mate_threshold,
        );
    }

    /// Verify evolvable mutation rate: mutation_rate_factor mutates slowly
    /// at a fixed 5% rate, independent of the genome's own mutation_rate_factor.
    #[test]
    fn test_mutation_rate_factor_stays_stable() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();
        let mut changes = 0;
        let trials = 200;

        for _ in 0..trials {
            let mut g = CreatureGenome::random(&mut rng);
            let before = g.behavior.mutation_rate_factor;
            mutate(&mut g, 0.5, 1.0, &mut rng, &mut tracker);
            if (g.behavior.mutation_rate_factor - before).abs() > 0.001 {
                changes += 1;
            }
        }

        // mutation_rate_factor mutates at 5%, so ~10 out of 200 should change
        assert!(
            changes < trials / 3,
            "mutation_rate_factor should mutate slowly (5% rate), but changed {}/{} times",
            changes,
            trials,
        );
        assert!(
            changes > 0,
            "mutation_rate_factor should occasionally mutate",
        );
    }

    #[test]
    fn test_pheromone_sensitivity_mutates_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = CreatureGenome::minimal_cell(&mut rng);
        let mut tracker = InnovationTracker::new();

        for _ in 0..1000 {
            mutate(&mut genome, 1.0, 1.0, &mut rng, &mut tracker);
        }

        assert!(
            genome.behavior.pheromone_sensitivity >= 0.0,
            "pheromone_sensitivity should be >= 0.0, got {}",
            genome.behavior.pheromone_sensitivity,
        );
        assert!(
            genome.behavior.pheromone_sensitivity <= 1.0,
            "pheromone_sensitivity should be <= 1.0, got {}",
            genome.behavior.pheromone_sensitivity,
        );
    }

    #[test]
    fn test_pheromone_sensitivity_inherited_via_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut parent_a = CreatureGenome::minimal_cell(&mut rng);
        let mut parent_b = CreatureGenome::minimal_cell(&mut rng);
        parent_a.behavior.pheromone_sensitivity = 0.1;
        parent_b.behavior.pheromone_sensitivity = 0.9;

        let child = crossover(&parent_a, &parent_b, &mut rng);

        assert!(
            (child.behavior.pheromone_sensitivity - 0.1).abs() < 0.01
                || (child.behavior.pheromone_sensitivity - 0.9).abs() < 0.01,
            "Pheromone sensitivity should be inherited from one parent: got {}",
            child.behavior.pheromone_sensitivity,
        );
    }

    #[test]
    fn test_learning_rate_stays_in_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = CreatureGenome::minimal_cell(&mut rng);
        let mut tracker = InnovationTracker::new();

        for _ in 0..1000 {
            mutate(&mut genome, 1.0, 1.0, &mut rng, &mut tracker);
        }

        assert!(
            genome.behavior.learning_rate >= 0.0,
            "Learning rate should be >= 0.0, got {}",
            genome.behavior.learning_rate,
        );
        assert!(
            genome.behavior.learning_rate <= 0.1,
            "Learning rate should be <= 0.1, got {}",
            genome.behavior.learning_rate,
        );
    }

    /// Verify new genome fields are preserved through crossover.
    #[test]
    fn test_crossover_preserves_new_genes() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut a = CreatureGenome::random(&mut rng);
        let mut b = CreatureGenome::random(&mut rng);
        a.behavior.mutation_rate_factor = 0.5;
        b.behavior.mutation_rate_factor = 2.0;
        a.behavior.mate_preference_hue = 0.0;
        b.behavior.mate_preference_hue = 1.0;

        let child = crossover(&a, &b, &mut rng);
        assert!(
            (child.behavior.mutation_rate_factor - 0.5).abs() < 0.001
                || (child.behavior.mutation_rate_factor - 2.0).abs() < 0.001,
            "Child mutation_rate_factor should be from one parent: {}",
            child.behavior.mutation_rate_factor,
        );
        assert!(
            (child.behavior.mate_preference_hue - 0.0).abs() < 0.001
                || (child.behavior.mate_preference_hue - 1.0).abs() < 0.001,
            "Child mate_preference_hue should be from one parent: {}",
            child.behavior.mate_preference_hue,
        );
    }
}
