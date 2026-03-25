//! Creature genome: the evolvable blueprint for appearance, animation, and behavior.
//! All genes are continuous floats — no discrete enums. Mutations are always gradual.

use rand::Rng;
use rand::RngExt;

use crate::brain::BrainGenome;

/// The complete genome for a creature — drives appearance, animation, behavior, and ecology.
#[derive(Debug, Clone)]
pub struct CreatureGenome {
    pub art: ArtGenome,
    pub anim: AnimGenome,
    pub behavior: BehaviorGenome,
    pub brain: BrainGenome,
    /// Master complexity gate (0.0–1.0). Controls which morphological features
    /// are expressed. Low complexity = simple cell, high = elaborate organism.
    pub complexity: f32,
    /// Generation number (parent's generation + 1). Starts at 0 for initial cells.
    pub generation: u32,
}

// ============================================================
// Art genome — controls visual appearance and body morphology
// ============================================================

#[derive(Debug, Clone)]
pub struct ArtGenome {
    /// Body elongation: 0.0 = round, 1.0 = elongated/slim.
    pub body_elongation: f32,
    /// Body height ratio: 0.0 = flat, 1.0 = tall.
    pub body_height_ratio: f32,
    /// Overall body size scale factor (0.2–2.0).
    pub body_size: f32,
    /// Rear protrusion fork level: 0.0 = pointed, 1.0 = forked.
    pub tail_fork: f32,
    /// Rear protrusion length (0.0–1.5).
    pub tail_length: f32,
    /// Top appendage size: 0.0 = none, 1.0 = large protrusion.
    pub top_appendage: f32,
    /// Side appendage size: 0.0 = none, 1.0 = large protrusions.
    pub side_appendages: f32,
    /// Interior pattern density: 0.0 = solid, 1.0 = dense pattern.
    pub pattern_density: f32,
    /// Sensor/eye size: 0.0 = tiny dot, 1.0 = large.
    pub eye_size: f32,
    /// Primary hue (0.0–1.0, mapped to terminal color palette).
    pub primary_hue: f32,
    /// Secondary hue (0.0–1.0).
    pub secondary_hue: f32,
    /// Color brightness (0.0–1.0, affects visibility vs camouflage).
    pub color_brightness: f32,
}

impl ArtGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            body_elongation: rng.random_range(0.0..1.0_f32),
            body_height_ratio: rng.random_range(0.0..1.0_f32),
            body_size: rng.random_range(0.2..2.0_f32),
            tail_fork: rng.random_range(0.0..1.0_f32),
            tail_length: rng.random_range(0.0..1.5_f32),
            top_appendage: rng.random_range(0.0..1.0_f32),
            side_appendages: rng.random_range(0.0..1.0_f32),
            pattern_density: rng.random_range(0.0..1.0_f32),
            eye_size: rng.random_range(0.0..1.0_f32),
            primary_hue: rng.random_range(0.0..1.0_f32),
            secondary_hue: rng.random_range(0.0..1.0_f32),
            color_brightness: rng.random_range(0.0..1.0_f32),
        }
    }

    /// Map eye_size to an ASCII character for rendering.
    pub fn eye_char(&self) -> char {
        if self.eye_size < 0.25 {
            '.'
        } else if self.eye_size < 0.5 {
            'o'
        } else if self.eye_size < 0.75 {
            '*'
        } else {
            'O'
        }
    }

    /// Map primary_hue to a color palette index (0–7).
    pub fn color_index(&self) -> u8 {
        ((self.primary_hue * 8.0).floor() as u8).min(7)
    }
}

// ============================================================
// Animation genome — controls movement style
// ============================================================

#[derive(Debug, Clone)]
pub struct AnimGenome {
    pub swim_speed: f32,         // 0.3–2.0 frames/sec multiplier
    pub tail_amplitude: f32,     // 0.0–1.0 how wide rear protrusion swishes
    pub idle_sway: f32,          // 0.0–1.0 body sway at rest
    pub undulation: f32,         // 0.0–1.0 body wave during swim
}

impl AnimGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            swim_speed: rng.random_range(0.3..2.0_f32),
            tail_amplitude: rng.random_range(0.0..1.0_f32),
            idle_sway: rng.random_range(0.0..1.0_f32),
            undulation: rng.random_range(0.0..1.0_f32),
        }
    }
}

// ============================================================
// Behavior genome — controls personality and ecology
// ============================================================

#[derive(Debug, Clone)]
pub struct BehaviorGenome {
    pub schooling_affinity: f32,  // 0.0–1.0
    pub aggression: f32,          // 0.0–1.0
    pub timidity: f32,            // 0.0–1.0
    pub speed_factor: f32,        // 0.5–2.0
    pub metabolism_factor: f32,   // 0.5–2.0
    pub max_lifespan_factor: f32, // 0.5–2.0
    pub reproduction_rate: f32,   // 0.2–1.0
    /// Mouth size relative to body (0.0–1.0). Determines max prey size.
    /// Small = grazer/filter feeder, large = predator.
    pub mouth_size: f32,
    /// Hunting instinct (0.0–1.0). Willingness to pursue mobile prey.
    pub hunting_instinct: f32,
    /// Mutation rate factor (0.5–2.0). Scales the base mutation rate.
    /// Allows evolution to tune its own evolvability (meta-evolution).
    pub mutation_rate_factor: f32,
    /// Mate preference hue (0.0–1.0). Preferred color in a mate.
    /// Drives Fisherian runaway sexual selection.
    pub mate_preference_hue: f32,
    /// Hebbian learning rate (0.0–0.1). Controls lifetime weight plasticity.
    /// 0.0 = no learning, higher = faster adaptation during lifetime.
    /// Learned weights are NOT inherited (Baldwin Effect).
    pub learning_rate: f32,
    /// Pheromone sensitivity (0.0–1.0). Scales perception of chemical signals.
    /// Higher values let creatures detect weaker pheromone concentrations.
    pub pheromone_sensitivity: f32,
}

impl BehaviorGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            schooling_affinity: rng.random_range(0.0..1.0_f32),
            aggression: rng.random_range(0.0..1.0_f32),
            timidity: rng.random_range(0.0..1.0_f32),
            speed_factor: rng.random_range(0.5..2.0_f32),
            metabolism_factor: rng.random_range(0.5..2.0_f32),
            max_lifespan_factor: rng.random_range(0.5..2.0_f32),
            reproduction_rate: rng.random_range(0.2..1.0_f32),
            mouth_size: rng.random_range(0.0..1.0_f32),
            hunting_instinct: rng.random_range(0.0..1.0_f32),
            mutation_rate_factor: rng.random_range(0.5..2.0_f32),
            mate_preference_hue: rng.random_range(0.0..1.0_f32),
            learning_rate: rng.random_range(0.0..0.1_f32),
            pheromone_sensitivity: rng.random_range(0.0..1.0_f32),
        }
    }
}

impl CreatureGenome {
    /// Create a fully random genome (all parameters in full range).
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            art: ArtGenome::random(rng),
            anim: AnimGenome::random(rng),
            behavior: BehaviorGenome::random(rng),
            brain: BrainGenome::random(rng),
            complexity: rng.random_range(0.0..1.0_f32),
            generation: 0,
        }
    }

    /// Create a minimal primordial cell genome — the starting point for evolution.
    /// Small, simple, passive grazers with tiny mouths.
    pub fn minimal_cell(rng: &mut impl Rng) -> Self {
        Self {
            art: ArtGenome {
                body_elongation: rng.random_range(0.3..0.7),
                body_height_ratio: rng.random_range(0.3..0.7),
                body_size: rng.random_range(0.3..0.5),
                tail_fork: 0.0,
                tail_length: 0.0,
                top_appendage: 0.0,
                side_appendages: 0.0,
                pattern_density: 0.0,
                eye_size: rng.random_range(0.0..0.3),
                primary_hue: rng.random_range(0.0..1.0),
                secondary_hue: rng.random_range(0.0..1.0),
                color_brightness: rng.random_range(0.3..0.6),
            },
            anim: AnimGenome {
                swim_speed: rng.random_range(0.3..0.8),
                tail_amplitude: 0.0,
                idle_sway: rng.random_range(0.1..0.4),
                undulation: rng.random_range(0.1..0.5),
            },
            behavior: BehaviorGenome {
                schooling_affinity: rng.random_range(0.0..0.3),
                aggression: rng.random_range(0.0..0.2),
                timidity: rng.random_range(0.2..0.6),
                speed_factor: rng.random_range(0.5..1.0),
                metabolism_factor: rng.random_range(0.8..1.2),
                max_lifespan_factor: rng.random_range(0.6..1.2),
                reproduction_rate: rng.random_range(0.5..1.0),
                mouth_size: rng.random_range(0.0..0.2),
                hunting_instinct: rng.random_range(0.0..0.1),
                mutation_rate_factor: rng.random_range(0.8..1.2),
                mate_preference_hue: rng.random_range(0.0..1.0),
                learning_rate: rng.random_range(0.0..0.05),
                pheromone_sensitivity: rng.random_range(0.0..0.5),
            },
            brain: BrainGenome::random(rng),
            complexity: rng.random_range(0.0..0.1),
            generation: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_genome_in_range() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let g = CreatureGenome::random(&mut rng);
            assert!(g.art.body_size >= 0.2 && g.art.body_size <= 2.0);
            assert!(g.art.body_elongation >= 0.0 && g.art.body_elongation <= 1.0);
            assert!(g.art.body_height_ratio >= 0.0 && g.art.body_height_ratio <= 1.0);
            assert!(g.art.color_brightness >= 0.0 && g.art.color_brightness <= 1.0);
            assert!(g.anim.swim_speed >= 0.3 && g.anim.swim_speed <= 2.0);
            assert!(g.behavior.speed_factor >= 0.5 && g.behavior.speed_factor <= 2.0);
            assert!(g.behavior.schooling_affinity >= 0.0 && g.behavior.schooling_affinity <= 1.0);
            assert!(g.behavior.mouth_size >= 0.0 && g.behavior.mouth_size <= 1.0);
            assert!(g.complexity >= 0.0 && g.complexity <= 1.0);
        }
    }

    #[test]
    fn test_minimal_cell_is_simple() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let cell = CreatureGenome::minimal_cell(&mut rng);
            assert!(cell.complexity <= 0.1, "Cell should be very simple");
            assert!(cell.art.body_size <= 0.5, "Cell should be small");
            assert!(cell.art.tail_length == 0.0, "Cell should have no tail");
            assert!(cell.art.top_appendage == 0.0, "Cell should have no appendages");
            assert!(cell.behavior.mouth_size <= 0.2, "Cell should have tiny mouth");
            assert!(cell.generation == 0);
        }
    }

    #[test]
    fn test_eye_char_ranges() {
        let mut art = ArtGenome::random(&mut rand::rng());
        art.eye_size = 0.1;
        assert_eq!(art.eye_char(), '.');
        art.eye_size = 0.3;
        assert_eq!(art.eye_char(), 'o');
        art.eye_size = 0.6;
        assert_eq!(art.eye_char(), '*');
        art.eye_size = 0.9;
        assert_eq!(art.eye_char(), 'O');
    }

    #[test]
    fn test_color_index() {
        let mut art = ArtGenome::random(&mut rand::rng());
        art.primary_hue = 0.0;
        assert_eq!(art.color_index(), 0);
        art.primary_hue = 0.5;
        assert_eq!(art.color_index(), 4);
        art.primary_hue = 0.99;
        assert_eq!(art.color_index(), 7);
    }
}
