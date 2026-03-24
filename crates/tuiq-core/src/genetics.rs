//! Genetic operations: crossover, mutation, speciation.
//! Evolves the full CreatureGenome — appearance, animation, and behavior co-evolve.

use rand::Rng;
use rand::RngExt;

use crate::brain;
use crate::genome::*;

/// Uniform crossover: each gene randomly from parent A or B.
pub fn crossover(a: &CreatureGenome, b: &CreatureGenome, rng: &mut impl Rng) -> CreatureGenome {
    CreatureGenome {
        art: crossover_art(&a.art, &b.art, rng),
        anim: crossover_anim(&a.anim, &b.anim, rng),
        behavior: crossover_behavior(&a.behavior, &b.behavior, rng),
        brain: brain::crossover_brain(&a.brain, &b.brain, rng),
    }
}

fn pick<T: Clone>(a: &T, b: &T, rng: &mut impl Rng) -> T {
    if rng.random_bool(0.5) { a.clone() } else { b.clone() }
}

fn pick_f32(a: f32, b: f32, rng: &mut impl Rng) -> f32 {
    if rng.random_bool(0.5) { a } else { b }
}

fn crossover_art(a: &ArtGenome, b: &ArtGenome, rng: &mut impl Rng) -> ArtGenome {
    ArtGenome {
        body_plan: pick(&a.body_plan, &b.body_plan, rng),
        body_size: pick_f32(a.body_size, b.body_size, rng),
        tail_style: pick(&a.tail_style, &b.tail_style, rng),
        tail_length: pick_f32(a.tail_length, b.tail_length, rng),
        has_dorsal_fin: pick(&a.has_dorsal_fin, &b.has_dorsal_fin, rng),
        has_pectoral_fins: pick(&a.has_pectoral_fins, &b.has_pectoral_fins, rng),
        fill_pattern: pick(&a.fill_pattern, &b.fill_pattern, rng),
        eye_style: pick(&a.eye_style, &b.eye_style, rng),
        primary_color: pick(&a.primary_color, &b.primary_color, rng),
        secondary_color: pick(&a.secondary_color, &b.secondary_color, rng),
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

fn crossover_behavior(a: &BehaviorGenome, b: &BehaviorGenome, rng: &mut impl Rng) -> BehaviorGenome {
    BehaviorGenome {
        schooling_affinity: pick_f32(a.schooling_affinity, b.schooling_affinity, rng),
        aggression: pick_f32(a.aggression, b.aggression, rng),
        timidity: pick_f32(a.timidity, b.timidity, rng),
        speed_factor: pick_f32(a.speed_factor, b.speed_factor, rng),
        metabolism_factor: pick_f32(a.metabolism_factor, b.metabolism_factor, rng),
        diet: pick(&a.diet, &b.diet, rng),
        max_lifespan_factor: pick_f32(a.max_lifespan_factor, b.max_lifespan_factor, rng),
        reproduction_rate: pick_f32(a.reproduction_rate, b.reproduction_rate, rng),
    }
}

/// Mutate a genome in-place. Each float gene has `rate` probability of being perturbed.
pub fn mutate(genome: &mut CreatureGenome, rate: f32, rng: &mut impl Rng) {
    mutate_art(&mut genome.art, rate, rng);
    mutate_anim(&mut genome.anim, rate, rng);
    mutate_behavior(&mut genome.behavior, rate, rng);
    brain::mutate_brain(&mut genome.brain, rate, rng);
}

fn perturb(val: &mut f32, min: f32, max: f32, rate: f32, rng: &mut impl Rng) {
    if rng.random_bool(rate as f64) {
        let delta: f32 = rng.random_range(-0.1..0.1);
        *val = (*val + delta * (max - min)).clamp(min, max);
    }
}

fn mutate_art(art: &mut ArtGenome, rate: f32, rng: &mut impl Rng) {
    if rng.random_bool(rate as f64 * 0.3) {
        art.body_plan = BodyPlan::random(rng);
    }
    perturb(&mut art.body_size, 0.7, 1.5, rate, rng);
    if rng.random_bool(rate as f64 * 0.3) {
        art.tail_style = TailStyle::random(rng);
    }
    perturb(&mut art.tail_length, 0.5, 1.5, rate, rng);
    if rng.random_bool(rate as f64 * 0.2) {
        art.has_dorsal_fin = !art.has_dorsal_fin;
    }
    if rng.random_bool(rate as f64 * 0.2) {
        art.has_pectoral_fins = !art.has_pectoral_fins;
    }
    if rng.random_bool(rate as f64 * 0.3) {
        art.fill_pattern = FillPattern::random(rng);
    }
    if rng.random_bool(rate as f64 * 0.2) {
        art.eye_style = EyeStyle::random(rng);
    }
    if rng.random_bool(rate as f64) {
        art.primary_color = rng.random_range(0..8);
    }
    if rng.random_bool(rate as f64) {
        art.secondary_color = rng.random_range(0..8);
    }
    perturb(&mut art.color_brightness, 0.3, 1.0, rate, rng);
}

fn mutate_anim(anim: &mut AnimGenome, rate: f32, rng: &mut impl Rng) {
    perturb(&mut anim.swim_speed, 0.5, 2.0, rate, rng);
    perturb(&mut anim.tail_amplitude, 0.3, 1.0, rate, rng);
    perturb(&mut anim.idle_sway, 0.0, 1.0, rate, rng);
    perturb(&mut anim.undulation, 0.0, 1.0, rate, rng);
}

fn mutate_behavior(beh: &mut BehaviorGenome, rate: f32, rng: &mut impl Rng) {
    perturb(&mut beh.schooling_affinity, 0.0, 1.0, rate, rng);
    perturb(&mut beh.aggression, 0.0, 1.0, rate, rng);
    perturb(&mut beh.timidity, 0.0, 1.0, rate, rng);
    perturb(&mut beh.speed_factor, 0.5, 2.0, rate, rng);
    perturb(&mut beh.metabolism_factor, 0.5, 2.0, rate, rng);
    if rng.random_bool(rate as f64 * 0.1) {
        beh.diet = DietType::random(rng);
    }
    perturb(&mut beh.max_lifespan_factor, 0.5, 2.0, rate, rng);
    perturb(&mut beh.reproduction_rate, 0.2, 1.0, rate, rng);
}

/// Genomic distance for speciation — how different two genomes are.
pub fn genomic_distance(a: &CreatureGenome, b: &CreatureGenome) -> f32 {
    let mut dist = 0.0_f32;

    // Art genes
    if a.art.body_plan != b.art.body_plan { dist += 1.0; }
    dist += (a.art.body_size - b.art.body_size).abs();
    if a.art.tail_style != b.art.tail_style { dist += 0.5; }
    dist += (a.art.tail_length - b.art.tail_length).abs();
    dist += (a.art.color_brightness - b.art.color_brightness).abs();

    // Behavior genes
    dist += (a.behavior.schooling_affinity - b.behavior.schooling_affinity).abs();
    dist += (a.behavior.aggression - b.behavior.aggression).abs();
    dist += (a.behavior.speed_factor - b.behavior.speed_factor).abs();
    if a.behavior.diet != b.behavior.diet { dist += 1.0; }

    // Brain weight distance (scaled)
    dist += brain::brain_distance(&a.brain, &b.brain) * 2.0;

    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover_produces_valid_genome() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let a = CreatureGenome::random(&mut rng);
            let b = CreatureGenome::random(&mut rng);
            let child = crossover(&a, &b, &mut rng);

            assert!(child.art.body_size >= 0.7 && child.art.body_size <= 1.5);
            assert!(child.behavior.speed_factor >= 0.5 && child.behavior.speed_factor <= 2.0);
        }
    }

    #[test]
    fn test_mutation_stays_in_range() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let mut g = CreatureGenome::random(&mut rng);
            mutate(&mut g, 0.5, &mut rng);

            assert!(g.art.body_size >= 0.7 && g.art.body_size <= 1.5,
                "body_size out of range: {}", g.art.body_size);
            assert!(g.art.tail_length >= 0.5 && g.art.tail_length <= 1.5);
            assert!(g.behavior.speed_factor >= 0.5 && g.behavior.speed_factor <= 2.0);
            assert!(g.behavior.schooling_affinity >= 0.0 && g.behavior.schooling_affinity <= 1.0);
            assert!(g.anim.swim_speed >= 0.5 && g.anim.swim_speed <= 2.0);
        }
    }

    #[test]
    fn test_genomic_distance_self_is_zero() {
        let g = CreatureGenome::random(&mut rand::rng());
        assert!((genomic_distance(&g, &g)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_genomic_distance_different_is_positive() {
        let mut rng = rand::rng();
        let a = CreatureGenome::random(&mut rng);
        let mut b = a.clone();
        b.art.body_plan = if a.art.body_plan == BodyPlan::Slim { BodyPlan::Round } else { BodyPlan::Slim };
        b.behavior.diet = if a.behavior.diet == DietType::Herbivore { DietType::Carnivore } else { DietType::Herbivore };
        assert!(genomic_distance(&a, &b) > 0.0);
    }

    #[test]
    fn test_mutation_eventually_changes_genome() {
        let mut rng = rand::rng();
        let original = CreatureGenome::random(&mut rng);
        let mut mutated = original.clone();
        // With high mutation rate over many iterations, genome should drift
        for _ in 0..100 {
            mutate(&mut mutated, 0.8, &mut rng);
        }
        assert!(
            genomic_distance(&original, &mutated) > 0.0,
            "Heavy mutation should change the genome"
        );
    }
}
