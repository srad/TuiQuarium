//! Phenotype derivation: genome → physical stats.
//! Morphology determines capabilities — appearance IS function.
//! All derivations use continuous interpolation — no enum matches.

use crate::genome::CreatureGenome;

/// Physical capabilities derived entirely from the genome's morphology.
#[derive(Debug, Clone, Default)]
pub struct DerivedPhysics {
    /// Maximum swim speed in cells/sec.
    pub max_speed: f32,
    /// Acceleration in cells/sec².
    pub acceleration: f32,
    /// Turn radius (higher = slower turning).
    pub turn_radius: f32,
    /// Drag coefficient (higher = more deceleration).
    pub drag_coefficient: f32,
    /// Body mass (drives metabolism via allometric scaling).
    pub body_mass: f32,
    /// Maximum energy storage capacity.
    pub max_energy: f32,
    /// Base metabolic rate (energy/sec at rest).
    pub base_metabolism: f32,
    /// Visual profile (how visible to others, 0–1).
    pub visual_profile: f32,
    /// Camouflage effectiveness (0–1, higher = harder to detect).
    pub camouflage: f32,
    /// Sensory range (cells).
    pub sensory_range: f32,
}

/// Feeding capability derived from genome — determines what a creature can eat.
/// Replaces the old TrophicRole enum with emergent, continuous feeding behavior.
#[derive(Debug, Clone, Default)]
pub struct FeedingCapability {
    /// Maximum prey body_mass this creature can consume.
    pub max_prey_mass: f32,
    /// Hunting effectiveness — ability to catch mobile prey (0+).
    pub hunt_skill: f32,
    /// Grazing effectiveness — ability to consume plants/detritus (0+).
    pub graze_skill: f32,
    /// Is this entity a producer (plant/food)? Only true for non-genome entities.
    pub is_producer: bool,
}

/// Derive all physical stats from the creature's genome.
pub fn derive_physics(genome: &CreatureGenome) -> DerivedPhysics {
    let art = &genome.art;
    let beh = &genome.behavior;

    // Body mass from size and body shape (continuous interpolation)
    // body_elongation: 0=round (mass factor 1.2), 1=slim (mass factor 0.6)
    let plan_mass_factor = 0.6 + 0.6 * (1.0 - art.body_elongation);
    let body_mass = art.body_size * art.body_size * plan_mass_factor;

    // Max speed: elongated is fast, round is slow; longer rear protrusion = more thrust
    let plan_speed_factor = 0.7 + 0.7 * art.body_elongation;
    let tail_thrust = 0.6 + art.tail_length * 0.4;
    let max_speed = 3.0 * plan_speed_factor * tail_thrust * beh.speed_factor / art.body_size.sqrt();

    // Acceleration from rear protrusion
    let acceleration = 8.0 * tail_thrust * beh.speed_factor / body_mass.max(0.1);

    // Turn radius: rounder body = slower turning
    let plan_turn = 0.5 + 1.0 * (1.0 - art.body_elongation);
    let turn_radius = plan_turn * art.body_size;

    // Drag: cross-section area
    let drag = 0.3 + 0.5 * (1.0 - art.body_elongation);
    let drag_coefficient = drag * art.body_size;

    // Energy capacity scales with body volume
    let max_energy = 100.0 * body_mass;

    // Metabolism: allometric scaling (mass^0.75), more efficient at higher complexity
    // Complex organisms have more efficient biochemistry (up to 25% reduction)
    let complexity_efficiency = 1.0 - 0.25 * genome.complexity;
    let base_metabolism = 0.5 * body_mass.powf(0.75) * beh.metabolism_factor * complexity_efficiency;

    // Visual profile: bigger + brighter = more visible
    let visual_profile = (art.body_size * art.color_brightness).min(1.0);

    // Camouflage: inverse of brightness
    let camouflage = (1.0 - art.color_brightness).max(0.0);

    // Sensory range from eye size (continuous), boosted by complexity, capped
    // Complex nervous systems process sensory data better (up to 50% bonus)
    let eye_range = 8.0 + 7.0 * art.eye_size;
    let complexity_bonus = 1.0 + 0.5 * genome.complexity;
    let sensory_range = (eye_range * art.body_size.sqrt() * complexity_bonus).min(18.0);

    DerivedPhysics {
        max_speed,
        acceleration,
        turn_radius,
        drag_coefficient,
        body_mass,
        max_energy,
        base_metabolism,
        visual_profile,
        camouflage,
        sensory_range,
    }
}

/// Derive feeding capability from genome and derived physics.
pub fn derive_feeding(genome: &CreatureGenome, physics: &DerivedPhysics) -> FeedingCapability {
    let beh = &genome.behavior;

    // Max prey mass: mouth_size determines what fraction of own body mass can be consumed
    let max_prey_mass = physics.body_mass * beh.mouth_size * 2.0;

    // Hunt skill: fast, aggressive, big-mouthed creatures are good hunters
    let hunt_skill = beh.aggression * beh.mouth_size * (physics.max_speed / 5.0).min(1.0)
        * beh.hunting_instinct;

    // Graze skill: passive, small-mouthed creatures are good grazers
    let graze_skill = (1.0 - beh.aggression) * (1.0 - beh.mouth_size * 0.5) * 0.5 + 0.5;

    FeedingCapability {
        max_prey_mass,
        hunt_skill,
        graze_skill,
        is_producer: false,
    }
}

/// Derive physical stats for a plant from its genome.
/// Uses allometric scaling and LAI-based photosynthesis model.
///
/// Key ecological models applied:
/// - Allometric metabolism: maintenance ∝ mass^0.75
///   (Kleiber, M. "Body Size and Metabolic Rate." Physiological Reviews, 1947)
/// - LAI photosynthesis: P = Pmax · (1 − e^(−c · LAI))
///   (Beer–Lambert law applied to leaf area index light interception)
/// - Complexity efficiency: more complex plants have slightly better biochemistry
///   (+15% at complexity = 1.0)
pub fn derive_plant_physics(genome: &crate::genome::PlantGenome) -> DerivedPhysics {
    let mass = genome.plant_mass();
    let lai = genome.effective_lai();

    // Max energy scales with mass and storage factor
    let max_energy = 15.0 * genome.max_energy_factor * (1.0 + mass);

    // Photosynthesis rate: Beer-Lambert light capture with diminishing returns
    // P = Pmax · (1 − e^(−0.5 · LAI)) · genome_rate
    let light_capture = 1.0 - (-0.5 * lai).exp();
    // base_metabolism is NEGATIVE for producers (photosynthesis)
    // Magnitude is the photosynthesis strength
    let base_photo = 0.25 * genome.photosynthesis_rate * (0.5 + light_capture);

    // Complexity efficiency: more complex plants photosynthesize slightly better
    let complexity_bonus = 1.0 + 0.15 * genome.complexity;

    // Final base_metabolism: negative = net producer
    let base_metabolism = -(base_photo * complexity_bonus);

    DerivedPhysics {
        body_mass: mass,
        max_energy,
        base_metabolism,
        max_speed: 0.0,
        acceleration: 0.0,
        turn_radius: 0.0,
        drag_coefficient: 0.0,
        visual_profile: 0.3 + 0.4 * genome.height_factor,
        camouflage: 0.0,
        sensory_range: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_genome(elongation: f32, size: f32) -> CreatureGenome {
        let mut g = CreatureGenome::random(&mut rand::rng());
        g.art.body_elongation = elongation;
        g.art.body_size = size;
        g.behavior.speed_factor = 1.0;
        g
    }

    #[test]
    fn test_elongated_faster_than_round() {
        let slim = derive_physics(&make_genome(1.0, 1.0));
        let round = derive_physics(&make_genome(0.0, 1.0));
        assert!(
            slim.max_speed > round.max_speed,
            "Elongated ({}) should be faster than round ({})",
            slim.max_speed,
            round.max_speed
        );
    }

    #[test]
    fn test_round_more_energy_than_elongated() {
        let slim = derive_physics(&make_genome(1.0, 1.0));
        let round = derive_physics(&make_genome(0.0, 1.0));
        assert!(
            round.max_energy > slim.max_energy,
            "Round ({}) should store more energy than elongated ({})",
            round.max_energy,
            slim.max_energy
        );
    }

    #[test]
    fn test_larger_body_higher_metabolism() {
        let small = derive_physics(&make_genome(0.5, 0.3));
        let big = derive_physics(&make_genome(0.5, 1.5));
        assert!(
            big.base_metabolism > small.base_metabolism,
            "Larger creature should have higher metabolism"
        );
    }

    #[test]
    fn test_bright_more_visible() {
        let mut bright = CreatureGenome::random(&mut rand::rng());
        bright.art.color_brightness = 1.0;
        let mut dull = bright.clone();
        dull.art.color_brightness = 0.0;
        let pb = derive_physics(&bright);
        let pd = derive_physics(&dull);
        assert!(pb.visual_profile > pd.visual_profile);
        assert!(pd.camouflage > pb.camouflage);
    }

    #[test]
    fn test_all_random_genomes_valid() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let g = CreatureGenome::random(&mut rng);
            let p = derive_physics(&g);
            assert!(p.max_speed > 0.0, "Speed must be positive");
            assert!(p.body_mass > 0.0, "Mass must be positive");
            assert!(p.max_energy > 0.0, "Energy must be positive");
            assert!(p.base_metabolism > 0.0, "Metabolism must be positive");
            assert!(p.sensory_range > 0.0, "Sensory range must be positive");
        }
    }

    #[test]
    fn test_feeding_capability() {
        let mut g = CreatureGenome::random(&mut rand::rng());
        // Big aggressive hunter
        g.behavior.mouth_size = 0.8;
        g.behavior.aggression = 0.9;
        g.behavior.hunting_instinct = 0.9;
        let p = derive_physics(&g);
        let f = derive_feeding(&g, &p);
        assert!(f.hunt_skill > 0.1, "Should be a capable hunter");
        assert!(f.max_prey_mass > 0.0, "Should be able to eat something");

        // Passive grazer
        g.behavior.mouth_size = 0.1;
        g.behavior.aggression = 0.1;
        g.behavior.hunting_instinct = 0.1;
        let p2 = derive_physics(&g);
        let f2 = derive_feeding(&g, &p2);
        assert!(f2.graze_skill > f2.hunt_skill, "Passive creature should be better at grazing");
    }

    #[test]
    fn test_minimal_cells_valid() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let g = CreatureGenome::minimal_cell(&mut rng);
            let p = derive_physics(&g);
            assert!(p.max_speed > 0.0);
            assert!(p.body_mass > 0.0);
            assert!(p.max_energy > 0.0);
            assert!(p.base_metabolism > 0.0);
        }
    }

    /// Hypothesis test: higher complexity gives better sensory range.
    /// A creature at complexity 0.8 should see farther than an identical
    /// creature at complexity 0.0 (due to 1.0 + 0.5*complexity multiplier).
    #[test]
    fn test_complexity_improves_sensory_range() {
        let mut rng = rand::rng();
        for _ in 0..50 {
            let mut g_simple = CreatureGenome::random(&mut rng);
            g_simple.complexity = 0.0;
            let mut g_complex = g_simple.clone();
            g_complex.complexity = 0.8;

            let p_simple = derive_physics(&g_simple);
            let p_complex = derive_physics(&g_complex);

            assert!(
                p_complex.sensory_range >= p_simple.sensory_range,
                "Complex creature (c=0.8) should see at least as far as simple (c=0.0): \
                 complex={:.2} vs simple={:.2}",
                p_complex.sensory_range, p_simple.sensory_range,
            );
        }
    }

    /// Hypothesis test: higher complexity reduces metabolism.
    /// A creature at complexity 0.8 should burn less energy than an identical
    /// creature at complexity 0.0 (due to 1.0 - 0.25*complexity efficiency).
    #[test]
    fn test_complexity_reduces_metabolism() {
        let mut rng = rand::rng();
        for _ in 0..50 {
            let mut g_simple = CreatureGenome::random(&mut rng);
            g_simple.complexity = 0.0;
            let mut g_complex = g_simple.clone();
            g_complex.complexity = 0.8;

            let p_simple = derive_physics(&g_simple);
            let p_complex = derive_physics(&g_complex);

            assert!(
                p_complex.base_metabolism < p_simple.base_metabolism,
                "Complex creature (c=0.8) should have lower metabolism than simple (c=0.0): \
                 complex={:.4} vs simple={:.4}",
                p_complex.base_metabolism, p_simple.base_metabolism,
            );

            // Verify the efficiency is ~20% reduction (0.8 * 0.25 = 0.20)
            let ratio = p_complex.base_metabolism / p_simple.base_metabolism;
            assert!(
                (ratio - 0.80).abs() < 0.01,
                "Metabolism ratio should be ~0.80 at complexity 0.8, got {:.4}",
                ratio,
            );
        }
    }

    #[test]
    fn test_sensory_range_cap_at_18() {
        let mut rng = rand::rng();
        let mut genome = CreatureGenome::random(&mut rng);
        genome.art.eye_size = 1.0;
        genome.art.body_size = 1.5;
        genome.complexity = 1.0;
        let physics = derive_physics(&genome);
        assert!(
            physics.sensory_range > 12.0,
            "With large eyes and high complexity, range should exceed old 12.0 cap: got {}",
            physics.sensory_range,
        );
        assert!(
            physics.sensory_range <= 18.0,
            "Range should be capped at 18.0: got {}",
            physics.sensory_range,
        );
    }

    #[test]
    fn test_complexity_increases_sensory_range() {
        let mut rng = rand::rng();
        let mut simple = CreatureGenome::random(&mut rng);
        simple.complexity = 0.0;
        simple.art.eye_size = 0.5;
        simple.art.body_size = 0.5;

        let mut complex = simple.clone();
        complex.complexity = 0.8;

        let simple_physics = derive_physics(&simple);
        let complex_physics = derive_physics(&complex);

        let improvement = complex_physics.sensory_range / simple_physics.sensory_range;
        assert!(
            improvement > 1.3,
            "Complexity 0.8 should give at least 30% more sensory range: {:.1}x improvement",
            improvement,
        );
    }

    #[test]
    fn test_plant_physics_valid() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let g = crate::genome::PlantGenome::random(&mut rng);
            let p = derive_plant_physics(&g);
            assert!(p.base_metabolism < 0.0, "Plant metabolism should be negative (producer)");
            assert!(p.max_energy > 0.0, "Plant max_energy must be positive");
            assert!(p.body_mass > 0.0, "Plant mass must be positive");
            assert_eq!(p.max_speed, 0.0, "Plants don't move");
        }
    }

    #[test]
    fn test_plant_lai_affects_photosynthesis() {
        let mut rng = rand::rng();
        let mut sparse = crate::genome::PlantGenome::minimal_plant(&mut rng);
        sparse.leaf_area = 0.1;
        sparse.branching = 0.0;
        let mut dense = sparse.clone();
        dense.leaf_area = 0.9;
        dense.branching = 0.8;

        let p_sparse = derive_plant_physics(&sparse);
        let p_dense = derive_plant_physics(&dense);

        assert!(
            p_dense.base_metabolism < p_sparse.base_metabolism,
            "Dense plant should photosynthesize more: {:.4} vs {:.4}",
            p_dense.base_metabolism, p_sparse.base_metabolism,
        );
    }

    #[test]
    fn test_plant_allometric_energy() {
        let mut rng = rand::rng();
        let mut small = crate::genome::PlantGenome::minimal_plant(&mut rng);
        small.stem_thickness = 0.1;
        small.height_factor = 0.2;
        small.max_energy_factor = 1.0;
        let mut large = small.clone();
        large.stem_thickness = 0.9;
        large.height_factor = 0.9;

        let p_small = derive_plant_physics(&small);
        let p_large = derive_plant_physics(&large);

        assert!(
            p_large.max_energy > p_small.max_energy,
            "Larger plant should store more energy: {:.2} vs {:.2}",
            p_large.max_energy, p_small.max_energy,
        );
    }
}
