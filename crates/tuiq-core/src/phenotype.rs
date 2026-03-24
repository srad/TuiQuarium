//! Phenotype derivation: genome → physical stats.
//! Morphology determines capabilities — appearance IS function.

use crate::genome::{BodyPlan, CreatureGenome};

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
    /// Visual profile (how visible to predators, 0–1).
    pub visual_profile: f32,
    /// Camouflage effectiveness (0–1, higher = harder to detect).
    pub camouflage: f32,
    /// Sensory range (cells).
    pub sensory_range: f32,
}

/// Derive all physical stats from the creature's genome.
pub fn derive_physics(genome: &CreatureGenome) -> DerivedPhysics {
    let art = &genome.art;
    let beh = &genome.behavior;

    // Body mass from size and body plan
    let plan_mass_factor = match art.body_plan {
        BodyPlan::Slim => 0.6,
        BodyPlan::Round => 1.2,
        BodyPlan::Flat => 0.8,
        BodyPlan::Tall => 1.0,
    };
    let body_mass = art.body_size * art.body_size * plan_mass_factor;

    // Max speed: slim is fast, round is slow; larger tail = more thrust
    let plan_speed_factor = match art.body_plan {
        BodyPlan::Slim => 1.4,
        BodyPlan::Round => 0.7,
        BodyPlan::Flat => 1.0,
        BodyPlan::Tall => 0.9,
    };
    let tail_thrust = 0.6 + art.tail_length * 0.4;
    let max_speed = 3.0 * plan_speed_factor * tail_thrust * beh.speed_factor / art.body_size.sqrt();

    // Acceleration from tail
    let acceleration = 8.0 * tail_thrust * beh.speed_factor / body_mass.max(0.1);

    // Turn radius: wider body = slower turning
    let plan_turn = match art.body_plan {
        BodyPlan::Slim => 0.5,
        BodyPlan::Round => 1.5,
        BodyPlan::Flat => 1.0,
        BodyPlan::Tall => 1.2,
    };
    let turn_radius = plan_turn * art.body_size;

    // Drag: cross-section area
    let drag_coefficient = match art.body_plan {
        BodyPlan::Slim => 0.3,
        BodyPlan::Round => 0.8,
        BodyPlan::Flat => 0.5,
        BodyPlan::Tall => 0.6,
    } * art.body_size;

    // Energy capacity scales with body volume
    let max_energy = 50.0 * body_mass;

    // Metabolism: allometric scaling (mass^0.75)
    let base_metabolism = 0.5 * body_mass.powf(0.75) * beh.metabolism_factor;

    // Visual profile: bigger + brighter = more visible
    let visual_profile = (art.body_size * art.color_brightness).min(1.0);

    // Camouflage: inverse of brightness
    let camouflage = (1.0 - art.color_brightness).max(0.0);

    // Sensory range from eye style
    let eye_range = match art.eye_style {
        crate::genome::EyeStyle::Dot => 8.0,
        crate::genome::EyeStyle::Circle => 12.0,
        crate::genome::EyeStyle::Star => 10.0,
        crate::genome::EyeStyle::Wide => 15.0,
    };
    let sensory_range = eye_range * art.body_size.sqrt();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::*;

    fn genome_with_plan(plan: BodyPlan) -> CreatureGenome {
        let mut g = CreatureGenome::random(&mut rand::rng());
        g.art.body_plan = plan;
        g.art.body_size = 1.0;
        g.behavior.speed_factor = 1.0;
        g
    }

    #[test]
    fn test_slim_faster_than_round() {
        let slim = derive_physics(&genome_with_plan(BodyPlan::Slim));
        let round = derive_physics(&genome_with_plan(BodyPlan::Round));
        assert!(
            slim.max_speed > round.max_speed,
            "Slim ({}) should be faster than Round ({})",
            slim.max_speed,
            round.max_speed
        );
    }

    #[test]
    fn test_round_more_energy_than_slim() {
        let slim = derive_physics(&genome_with_plan(BodyPlan::Slim));
        let round = derive_physics(&genome_with_plan(BodyPlan::Round));
        assert!(
            round.max_energy > slim.max_energy,
            "Round ({}) should store more energy than Slim ({})",
            round.max_energy,
            slim.max_energy
        );
    }

    #[test]
    fn test_larger_body_higher_metabolism() {
        let mut small = CreatureGenome::random(&mut rand::rng());
        small.art.body_size = 0.7;
        small.behavior.metabolism_factor = 1.0;
        let mut big = small.clone();
        big.art.body_size = 1.5;
        let ps = derive_physics(&small);
        let pb = derive_physics(&big);
        assert!(
            pb.base_metabolism > ps.base_metabolism,
            "Larger creature should have higher metabolism"
        );
    }

    #[test]
    fn test_bright_more_visible() {
        let mut bright = CreatureGenome::random(&mut rand::rng());
        bright.art.color_brightness = 1.0;
        let mut dull = bright.clone();
        dull.art.color_brightness = 0.3;
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
}
