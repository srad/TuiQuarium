//! Creature genome: the evolvable blueprint for appearance, animation, and behavior.

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
}

// ============================================================
// Art genome — controls visual appearance
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPlan {
    Slim,
    Round,
    Flat,
    Tall,
}

impl BodyPlan {
    pub const ALL: [BodyPlan; 4] = [Self::Slim, Self::Round, Self::Flat, Self::Tall];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.random_range(0..Self::ALL.len())]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailStyle {
    Forked,
    Fan,
    Pointed,
    Flowing,
}

impl TailStyle {
    pub const ALL: [TailStyle; 4] = [Self::Forked, Self::Fan, Self::Pointed, Self::Flowing];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.random_range(0..Self::ALL.len())]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillPattern {
    Solid,
    Striped,
    Spotted,
    Scales,
}

impl FillPattern {
    pub const ALL: [FillPattern; 4] = [Self::Solid, Self::Striped, Self::Spotted, Self::Scales];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.random_range(0..Self::ALL.len())]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeStyle {
    Dot,
    Circle,
    Star,
    Wide,
}

impl EyeStyle {
    pub const ALL: [EyeStyle; 4] = [Self::Dot, Self::Circle, Self::Star, Self::Wide];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.random_range(0..Self::ALL.len())]
    }

    pub fn char(&self) -> char {
        match self {
            Self::Dot => '.',
            Self::Circle => 'o',
            Self::Star => '*',
            Self::Wide => 'O',
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArtGenome {
    pub body_plan: BodyPlan,
    pub body_size: f32,          // 0.7–1.5 scale factor
    pub tail_style: TailStyle,
    pub tail_length: f32,        // 0.5–1.5
    pub has_dorsal_fin: bool,
    pub has_pectoral_fins: bool,
    pub fill_pattern: FillPattern,
    pub eye_style: EyeStyle,
    pub primary_color: u8,       // 0–7 index into palette
    pub secondary_color: u8,
    pub color_brightness: f32,   // 0.3–1.0 (affects camouflage vs visibility)
}

impl ArtGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            body_plan: BodyPlan::random(rng),
            body_size: rng.random_range(0.7..1.5_f32),
            tail_style: TailStyle::random(rng),
            tail_length: rng.random_range(0.5..1.5_f32),
            has_dorsal_fin: rng.random_bool(0.6),
            has_pectoral_fins: rng.random_bool(0.5),
            fill_pattern: FillPattern::random(rng),
            eye_style: EyeStyle::random(rng),
            primary_color: rng.random_range(0..8),
            secondary_color: rng.random_range(0..8),
            color_brightness: rng.random_range(0.3..1.0_f32),
        }
    }
}

// ============================================================
// Animation genome — controls movement style
// ============================================================

#[derive(Debug, Clone)]
pub struct AnimGenome {
    pub swim_speed: f32,         // 0.5–2.0 frames/sec multiplier
    pub tail_amplitude: f32,     // 0.3–1.0 how wide tail swishes
    pub idle_sway: f32,          // 0.0–1.0 body sway at rest
    pub undulation: f32,         // 0.0–1.0 body wave during swim
}

impl AnimGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            swim_speed: rng.random_range(0.5..2.0_f32),
            tail_amplitude: rng.random_range(0.3..1.0_f32),
            idle_sway: rng.random_range(0.0..1.0_f32),
            undulation: rng.random_range(0.0..1.0_f32),
        }
    }
}

// ============================================================
// Behavior genome — controls personality and ecology
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DietType {
    Herbivore,
    Omnivore,
    Carnivore,
}

impl DietType {
    pub const ALL: [DietType; 3] = [Self::Herbivore, Self::Omnivore, Self::Carnivore];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.random_range(0..Self::ALL.len())]
    }
}

#[derive(Debug, Clone)]
pub struct BehaviorGenome {
    pub schooling_affinity: f32,  // 0.0–1.0
    pub aggression: f32,          // 0.0–1.0
    pub timidity: f32,            // 0.0–1.0
    pub speed_factor: f32,        // 0.5–2.0
    pub metabolism_factor: f32,   // 0.5–2.0
    pub diet: DietType,
    pub max_lifespan_factor: f32, // 0.5–2.0
    pub reproduction_rate: f32,   // 0.2–1.0
}

impl BehaviorGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            schooling_affinity: rng.random_range(0.0..1.0_f32),
            aggression: rng.random_range(0.0..1.0_f32),
            timidity: rng.random_range(0.0..1.0_f32),
            speed_factor: rng.random_range(0.5..2.0_f32),
            metabolism_factor: rng.random_range(0.5..2.0_f32),
            diet: DietType::random(rng),
            max_lifespan_factor: rng.random_range(0.5..2.0_f32),
            reproduction_rate: rng.random_range(0.2..1.0_f32),
        }
    }
}

impl CreatureGenome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            art: ArtGenome::random(rng),
            anim: AnimGenome::random(rng),
            behavior: BehaviorGenome::random(rng),
            brain: BrainGenome::random(rng),
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
            assert!(g.art.body_size >= 0.7 && g.art.body_size <= 1.5);
            assert!(g.art.tail_length >= 0.5 && g.art.tail_length <= 1.5);
            assert!(g.art.color_brightness >= 0.3 && g.art.color_brightness <= 1.0);
            assert!(g.art.primary_color < 8);
            assert!(g.art.secondary_color < 8);
            assert!(g.anim.swim_speed >= 0.5 && g.anim.swim_speed <= 2.0);
            assert!(g.behavior.speed_factor >= 0.5 && g.behavior.speed_factor <= 2.0);
            assert!(g.behavior.schooling_affinity >= 0.0 && g.behavior.schooling_affinity <= 1.0);
        }
    }
}
