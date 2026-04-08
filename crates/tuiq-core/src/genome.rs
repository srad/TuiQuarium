//! Creature and producer genomes: evolvable blueprints for all organisms.
//! All genes are continuous floats — no discrete enums. Mutations are always gradual.
//!
//! # Creature genome
//! 35+ genes controlling body plan, appendages, behavior, and NEAT brain topology.
//!
//! # Producer genome
//! 16 genes controlling producer geometry and physiology, informed by ecological theory:
//! - L-systems (Lindenmayer, 1968): branching and curvature → visual shape
//! - Allometric scaling (Kleiber, 1947): metabolism ∝ mass^0.75
//! - Beer–Lambert law: photosynthesis via Leaf Area Index (LAI)
//! - Grime's C-S-R Triangle (1977): Competitor/Stress-tolerator/Ruderal strategies
//! - r/K selection (MacArthur & Wilson, 1967): seed count vs seed size trade-off
//! - Trait-mediated light-depth responses in submerged macrophytes (He et al., 2019)
//! - Propagule pressure and depth-limited establishment (Li et al., 2015)

use rand::Rng;
use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::brain::BrainGenome;

/// The complete genome for a creature — drives appearance, animation, behavior, and ecology.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtGenome {
    /// Body elongation: 0.0 = round, 1.0 = elongated/slim.
    pub body_elongation: f32,
    /// Body height ratio: 0.0 = flat, 1.0 = tall.
    pub body_height_ratio: f32,
    /// Overall body size scale factor (0.2–5.0).
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
            body_size: rng.random_range(0.2..5.0_f32),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimGenome {
    pub swim_speed: f32,     // 0.3–2.0 frames/sec multiplier
    pub tail_amplitude: f32, // 0.0–1.0 how wide rear protrusion swishes
    pub idle_sway: f32,      // 0.0–1.0 body sway at rest
    pub undulation: f32,     // 0.0–1.0 body wave during swim
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

// ============================================================
// Producer genome — evolvable blueprint for aquatic producer geometry and physiology.
// Producer colonies are sessile photosynthesizers: no brain, no movement.
//
// Ecological foundations (see README.md § References for full citations):
//
//   - Allometric scaling (Kleiber, 1947):
//     Kleiber, M. "Body Size and Metabolic Rate." Physiological Reviews, 27(4), 511–541.
//     Metabolism ∝ mass^0.75. Used for plant maintenance cost calculation.
//
//   - Leaf Area Index / Beer–Lambert law:
//     Photosynthesis = Pmax·(1 − e^(−c·LAI)). LAI is a proxy for canopy light
//     interception. High leaf_area gene → higher LAI → more photosynthesis with
//     diminishing returns.
//
//   - Grime's C-S-R Triangle (Grime, 1977):
//     Grime, J.P. "Evidence for the Existence of Three Primary Strategies in Plants."
//     The American Naturalist, 111(982), 1169–1194.
//     Competitor (high photosynthesis_rate) / Stress-tolerator (high hardiness) /
//     Ruderal (high seed_count) strategies emerge from gene trade-offs.
//
//   - r/K selection (MacArthur & Wilson, 1967):
//     MacArthur, R.H. & Wilson, E.O. "The Theory of Island Biogeography."
//     Princeton University Press.
//     seed_count (r-strategy) trades off with seed_size (K-strategy).
//
//   - L-systems (Lindenmayer, 1968):
//     Lindenmayer, A. "Mathematical Models for Cellular Interaction in Development."
//     Journal of Theoretical Biology, 18(3), 280–315.
//     Branching and curvature genes parameterize production rules for
//     procedural plant ASCII art: F → F[+F][-F].
// ============================================================

/// The complete genome for an aquatic producer colony — drives appearance and physiology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerGenome {
    // ── Morphology (L-system parameters) ───────────────────────
    /// Support density / attachment strength proxy.
    pub stem_thickness: f32,
    /// Vertical profile: low = mat-like, high = plume-like.
    pub height_factor: f32,
    /// Active photosynthetic surface proxy.
    pub leaf_area: f32,
    /// Colony roughness / branching heterogeneity.
    pub branching: f32,
    /// Irregularity of the producer colony outline.
    pub curvature: f32,
    /// Primary hue (0.0–1.0, mapped to green/teal/rust palette).
    pub primary_hue: f32,

    // ── Physiology (ecological trade-offs) ─────────────────────
    /// Photosynthetic productivity (0.5–2.0): rate of light → reserve conversion.
    pub photosynthesis_rate: f32,
    /// Storage capacity factor (0.5–2.0): scales base reserve capacity.
    pub max_energy_factor: f32,
    /// Stress tolerance (0.0–1.0): resistance to darkness and resource stress.
    pub hardiness: f32,
    /// Broadcast propagule range factor (0.5–2.0): how far dispersal reaches.
    pub seed_range: f32,
    /// Broadcast propagule count factor (0.3–2.0): many cheap dispersers vs fewer large propagules.
    pub seed_count: f32,
    /// Propagule size factor (0.3–2.0): investment per dispersal unit.
    pub seed_size: f32,
    /// Turnover resistance factor (0.5–2.0): scales base persistence.
    pub lifespan_factor: f32,
    /// Nutritional value (0.3–1.5): grazer reward from consuming active biomass.
    pub nutritional_value: f32,
    /// Local fragmentation / lateral spread tendency (0.0–1.0).
    pub clonal_spread: f32,
    /// Nutrient affinity (0.3–1.5): ability to keep growing under low dissolved N/P.
    pub nutrient_affinity: f32,
    /// Fouling resistance (0.0–1.0): resistance to attached/suspended shading load.
    pub epiphyte_resistance: f32,
    /// Reserve allocation (0.0–1.0): fraction of surplus reserve diverted into propagules.
    pub reserve_allocation: f32,

    // ── Evolution tracking ─────────────────────────────────────
    /// Master complexity gate (0.0–1.0): controls visual tier and efficiency.
    pub complexity: f32,
    /// Generation number.
    pub generation: u32,
    /// Mutation rate factor (0.5–2.0): meta-evolution of evolvability.
    pub mutation_rate_factor: f32,
}

impl ProducerGenome {
    /// Create a fully random producer genome.
    pub fn random(rng: &mut impl rand::Rng) -> Self {
        use rand::RngExt;
        Self {
            stem_thickness: rng.random_range(0.0..1.0_f32),
            height_factor: rng.random_range(0.0..1.0_f32),
            leaf_area: rng.random_range(0.0..1.0_f32),
            branching: rng.random_range(0.0..1.0_f32),
            curvature: rng.random_range(0.0..1.0_f32),
            primary_hue: rng.random_range(0.0..1.0_f32),
            photosynthesis_rate: rng.random_range(0.5..2.0_f32),
            max_energy_factor: rng.random_range(0.5..2.0_f32),
            hardiness: rng.random_range(0.0..1.0_f32),
            seed_range: rng.random_range(0.5..2.0_f32),
            seed_count: rng.random_range(0.3..2.0_f32),
            seed_size: rng.random_range(0.3..2.0_f32),
            lifespan_factor: rng.random_range(0.5..2.0_f32),
            nutritional_value: rng.random_range(0.3..1.5_f32),
            clonal_spread: rng.random_range(0.0..1.0_f32),
            nutrient_affinity: rng.random_range(0.3..1.5_f32),
            epiphyte_resistance: rng.random_range(0.0..1.0_f32),
            reserve_allocation: rng.random_range(0.0..1.0_f32),
            complexity: rng.random_range(0.0..1.0_f32),
            generation: 0,
            mutation_rate_factor: rng.random_range(0.5..2.0_f32),
        }
    }

    /// Create a minimal aquatic producer colony genome.
    ///
    /// Research note: the unified model starts from simple phototrophic colonies
    /// rather than from rooted macrophytes, so default founders stay low-profile
    /// and fast-turnover instead of being prebuilt higher plants.
    pub fn minimal_producer(rng: &mut impl rand::Rng) -> Self {
        use rand::RngExt;
        Self {
            stem_thickness: rng.random_range(0.2..0.6_f32),
            height_factor: rng.random_range(0.1..0.5_f32),
            leaf_area: rng.random_range(0.3..0.7_f32),
            branching: rng.random_range(0.1..0.5_f32),
            curvature: rng.random_range(0.1..0.6_f32),
            primary_hue: rng.random_range(0.0..1.0_f32),
            photosynthesis_rate: rng.random_range(0.8..1.2_f32),
            max_energy_factor: rng.random_range(0.8..1.2_f32),
            hardiness: rng.random_range(0.2..0.6_f32),
            seed_range: rng.random_range(0.7..1.4_f32),
            seed_count: rng.random_range(0.5..1.1_f32),
            seed_size: rng.random_range(0.3..0.8_f32),
            lifespan_factor: rng.random_range(0.8..1.2_f32),
            nutritional_value: rng.random_range(0.5..1.0_f32),
            clonal_spread: rng.random_range(0.2..0.8_f32),
            nutrient_affinity: rng.random_range(0.7..1.1_f32),
            epiphyte_resistance: rng.random_range(0.2..0.6_f32),
            reserve_allocation: rng.random_range(0.3..0.7_f32),
            complexity: rng.random_range(0.05..0.25_f32),
            generation: 0,
            mutation_rate_factor: rng.random_range(0.8..1.2_f32),
        }
    }

    /// Allometric producer mass from morphology genes.
    /// Mass = stem_thickness × height_factor × scale.
    /// Used for metabolism scaling (Kleiber's law: rate ∝ mass^0.75).
    pub fn producer_mass(&self) -> f32 {
        let base = self.stem_thickness * 0.5 + 0.1;
        let height = self.height_factor * 0.8 + 0.2;
        base * height
    }

    /// Effective capture area for photosynthesis calculation.
    /// Higher leaf_area and branching means more light interception,
    /// but follows Beer-Lambert law with diminishing returns.
    pub fn effective_capture_area(&self) -> f32 {
        self.leaf_area * (1.0 + self.branching * 0.5) * (0.5 + self.height_factor * 0.5)
    }

    /// Target support biomass for this morphology.
    ///
    /// Research note: aquatic primary-producer performance still depends on
    /// realized colony geometry and exposed structure, not on a single reserve
    /// pool, so we keep morphology-driven biomass targets in the unified model.
    pub fn support_target_biomass(&self) -> f32 {
        0.6 + self.producer_mass() * 5.0 + self.height_factor * 1.5
    }

    /// Target active biomass for this morphology.
    pub fn active_target_biomass(&self) -> f32 {
        0.5 + self.effective_capture_area() * 3.5
    }

    /// Map primary_hue to a color palette index for producer rendering.
    pub fn color_index(&self) -> u8 {
        if self.primary_hue < 0.2 {
            5 // Green
        } else if self.primary_hue < 0.4 {
            8 // Dark green (new palette entry)
        } else if self.primary_hue < 0.55 {
            9 // Yellow-green / lime (new palette entry)
        } else if self.primary_hue < 0.7 {
            10 // Teal / sea green (new palette entry)
        } else if self.primary_hue < 0.85 {
            6 // Orange / brown
        } else {
            11 // Red-brown / rust (new palette entry)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_random_genome_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let g = CreatureGenome::random(&mut rng);
            assert!(g.art.body_size >= 0.2 && g.art.body_size <= 5.0);
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
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let cell = CreatureGenome::minimal_cell(&mut rng);
            assert!(cell.complexity <= 0.1, "Cell should be very simple");
            assert!(cell.art.body_size <= 0.5, "Cell should be small");
            assert!(cell.art.tail_length == 0.0, "Cell should have no tail");
            assert!(
                cell.art.top_appendage == 0.0,
                "Cell should have no appendages"
            );
            assert!(
                cell.behavior.mouth_size <= 0.2,
                "Cell should have tiny mouth"
            );
            assert!(cell.generation == 0);
        }
    }

    #[test]
    fn test_eye_char_ranges() {
        let mut art = ArtGenome::random(&mut StdRng::seed_from_u64(42));
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
        let mut art = ArtGenome::random(&mut StdRng::seed_from_u64(42));
        art.primary_hue = 0.0;
        assert_eq!(art.color_index(), 0);
        art.primary_hue = 0.5;
        assert_eq!(art.color_index(), 4);
        art.primary_hue = 0.99;
        assert_eq!(art.color_index(), 7);
    }

    #[test]
    fn test_producer_genome_random_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let g = ProducerGenome::random(&mut rng);
            assert!(g.stem_thickness >= 0.0 && g.stem_thickness <= 1.0);
            assert!(g.height_factor >= 0.0 && g.height_factor <= 1.0);
            assert!(g.leaf_area >= 0.0 && g.leaf_area <= 1.0);
            assert!(g.photosynthesis_rate >= 0.5 && g.photosynthesis_rate <= 2.0);
            assert!(g.complexity >= 0.0 && g.complexity <= 1.0);
            assert!(g.seed_count >= 0.3 && g.seed_count <= 2.0);
            assert!(g.seed_size >= 0.3 && g.seed_size <= 2.0);
            assert!(g.lifespan_factor >= 0.5 && g.lifespan_factor <= 2.0);
            assert!(g.clonal_spread >= 0.0 && g.clonal_spread <= 1.0);
            assert!(g.nutrient_affinity >= 0.3 && g.nutrient_affinity <= 1.5);
            assert!(g.epiphyte_resistance >= 0.0 && g.epiphyte_resistance <= 1.0);
            assert!(g.reserve_allocation >= 0.0 && g.reserve_allocation <= 1.0);
        }
    }

    #[test]
    fn test_minimal_producer_is_simple() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let g = ProducerGenome::minimal_producer(&mut rng);
            assert!(
                g.complexity <= 0.4,
                "Minimal producer complexity should be moderate"
            );
            assert!(g.generation == 0);
            assert!(g.branching <= 0.5);
            assert!(g.producer_mass() > 0.0, "Plant mass should be positive");
            assert!(g.effective_capture_area() > 0.0, "LAI should be positive");
            assert!(g.support_target_biomass() > 0.0);
            assert!(g.active_target_biomass() > 0.0);
        }
    }

    #[test]
    fn test_producer_color_index() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = ProducerGenome::minimal_producer(&mut rng);
        g.primary_hue = 0.1;
        assert_eq!(g.color_index(), 5); // green
        g.primary_hue = 0.3;
        assert_eq!(g.color_index(), 8); // dark green
        g.primary_hue = 0.5;
        assert_eq!(g.color_index(), 9); // yellow-green
        g.primary_hue = 0.65;
        assert_eq!(g.color_index(), 10); // teal
        g.primary_hue = 0.75;
        assert_eq!(g.color_index(), 6); // orange/brown
        g.primary_hue = 0.9;
        assert_eq!(g.color_index(), 11); // rust brown
    }

    #[test]
    fn test_producer_allometric_mass() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut small = ProducerGenome::minimal_producer(&mut rng);
        small.stem_thickness = 0.1;
        small.height_factor = 0.2;
        let mut large = small.clone();
        large.stem_thickness = 0.9;
        large.height_factor = 0.9;
        assert!(
            large.producer_mass() > small.producer_mass(),
            "Larger producer should have more mass: {} vs {}",
            large.producer_mass(),
            small.producer_mass()
        );
    }

    #[test]
    fn test_producer_capture_area_increases_with_leaf_area() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut sparse = ProducerGenome::minimal_producer(&mut rng);
        sparse.leaf_area = 0.1;
        sparse.branching = 0.0;
        let mut dense = sparse.clone();
        dense.leaf_area = 0.9;
        dense.branching = 0.8;
        assert!(
            dense.effective_capture_area() > sparse.effective_capture_area(),
            "Dense producer should have higher LAI"
        );
    }
}
