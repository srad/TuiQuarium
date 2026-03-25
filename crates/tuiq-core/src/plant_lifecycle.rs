//! Plant lifecycle: genome-driven growth stages and appearance updates.
//! Plants are sessile producers whose visual appearance and behavior
//! emerge from their PlantGenome rather than hardcoded species types.
//!
//! Stage derivation uses energy fraction and age (same thresholds for all
//! plants — the genome affects HOW they look at each stage, not WHEN they
//! transition).
//!
//! ASCII art generation uses an L-system-inspired approach
//! (Lindenmayer, 1968) where branching probability and turning angle
//! are parameterized by genome genes. Visual complexity is gated by the
//! complexity gene, mirroring the creature art system.

use crate::components::*;
use crate::ecosystem::Energy;
use crate::genome::PlantGenome;

/// Thresholds for stage transitions (in simulation seconds).
const SEEDLING_DURATION: f32 = 15.0;
const YOUNG_DURATION: f32 = 40.0;
const WILTING_ENERGY_THRESHOLD: f32 = 0.20;
const FLOWERING_ENERGY_THRESHOLD: f32 = 0.85;

/// Compute the current plant stage from energy fraction and age.
pub fn compute_stage(energy: &Energy, age: &PlantAge) -> PlantStage {
    let frac = energy.fraction();
    if frac < WILTING_ENERGY_THRESHOLD {
        PlantStage::Wilting
    } else if age.seconds < SEEDLING_DURATION {
        PlantStage::Seedling
    } else if age.seconds < YOUNG_DURATION {
        PlantStage::Young
    } else if frac >= FLOWERING_ENERGY_THRESHOLD {
        PlantStage::Flowering
    } else {
        PlantStage::Mature
    }
}

/// ECS system: advance plant age, recompute stage, update appearance when stage changes.
pub fn plant_lifecycle_system(world: &mut hecs::World, dt: f32) {
    let mut updates: Vec<(hecs::Entity, PlantGenome, PlantStage)> = Vec::new();

    for (entity, age, genome, stage, energy) in
        &mut world.query::<(hecs::Entity, &mut PlantAge, &PlantGenome, &mut PlantStage, &Energy)>()
    {
        age.seconds += dt;
        let new_stage = compute_stage(energy, age);
        if new_stage != *stage {
            *stage = new_stage;
            updates.push((entity, genome.clone(), new_stage));
        }
    }

    // Apply appearance updates (separate pass to avoid borrow conflict)
    for (entity, genome, stage) in updates {
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage);
        let _ = world.insert(entity, (appearance, bbox, AnimationState::new(0.8)));
    }
}

/// Build full appearance for a plant from its genome and current stage.
pub fn build_appearance_from_genome(genome: &PlantGenome, stage: PlantStage) -> (Appearance, BoundingBox) {
    let (rows, sway_rows) = generate_plant_art(genome, stage);

    let w = rows.iter().map(|r| r.len()).max().unwrap_or(1) as f32;
    let h = rows.len() as f32;

    let frame = AsciiFrame::from_rows(rows.iter().map(|s| s.as_str()).collect());
    let sway_frame = AsciiFrame::from_rows(sway_rows.iter().map(|s| s.as_str()).collect());

    let frame_sets = vec![
        vec![frame.clone(), sway_frame], // swim/sway frames
        vec![frame],                      // idle frames
    ];

    let appearance = Appearance {
        frame_sets,
        facing: Direction::Right,
        color_index: genome.color_index(),
    };

    (appearance, BoundingBox { w, h })
}

/// Generate plant ASCII art from genome using L-system-inspired approach.
///
/// L-systems (Lindenmayer, A. "Mathematical Models for Cellular Interaction
/// in Development." Journal of Theoretical Biology, 1968) define recursive
/// branching rules like F → F[+F][-F]. Here, the genome's `branching` and
/// `curvature` genes control the probability and character choice for branches.
///
/// Complexity gates visual tier (like creatures):
///   < 0.15: single character (spore/cell)
///   0.15–0.35: 2-3 line sprout
///   0.35–0.6: medium plant with branching
///   0.6+: elaborate multi-line plant
///
/// **Stage scaling**: the tier is determined by raw `genome.complexity`, but
/// the *size within each tier* grows with the plant's lifecycle stage. This
/// ensures that Seedling → Young → Mature produces visibly different art even
/// when all stages fall within the same complexity tier.
///
/// Genome traits affect the generated shape:
///   height_factor → number of rows
///   stem_thickness → stem character width
///   leaf_area → density of leaf characters
///   branching → width spread, branch count
///   curvature → use of curved vs straight characters
fn generate_plant_art(genome: &PlantGenome, stage: PlantStage) -> (Vec<String>, Vec<String>) {
    // Use raw complexity for tier selection — the stage controls size WITHIN each tier
    let complexity = genome.complexity.max(0.08);

    if complexity < 0.15 {
        generate_spore_art(genome, stage)
    } else if complexity < 0.35 {
        generate_sprout_art(genome, stage)
    } else if complexity < 0.6 {
        generate_medium_plant_art(genome, stage)
    } else {
        generate_complex_plant_art(genome, stage)
    }
}

// ── Spore tier: complexity < 0.15 ────────────────────────────

fn generate_spore_art(genome: &PlantGenome, stage: PlantStage) -> (Vec<String>, Vec<String>) {
    let ch = if stage == PlantStage::Wilting {
        '.'
    } else if stage == PlantStage::Flowering {
        '*'
    } else if genome.leaf_area > 0.5 {
        ','
    } else {
        '.'
    };
    let s = String::from(ch);
    (vec![s.clone()], vec![s])
}

// ── Sprout tier: complexity 0.15–0.35 ────────────────────────

fn generate_sprout_art(genome: &PlantGenome, stage: PlantStage) -> (Vec<String>, Vec<String>) {
    let stem = if genome.curvature > 0.5 { ")" } else { "|" };
    let leaf_ch = if genome.leaf_area > 0.6 { "}" } else if genome.leaf_area > 0.3 { ")" } else { "'" };

    let (primary, sway) = match stage {
        PlantStage::Seedling => (
            vec![format!(" {leaf_ch}"), format!(" {stem}")],
            vec![format!("{leaf_ch} "), format!(" {stem}")],
        ),
        PlantStage::Young => (
            vec![format!(" {leaf_ch}"), format!("{leaf_ch}{stem}"), format!("_{stem}_")],
            vec![format!("{leaf_ch} "), format!("{stem}{leaf_ch}"), format!("_{stem}_")],
        ),
        PlantStage::Mature => (
            vec![
                format!("{leaf_ch} {leaf_ch}"),
                format!("{leaf_ch}{stem}{leaf_ch}"),
                format!(" {stem} "),
                format!("_{stem}_"),
            ],
            vec![
                format!(" {leaf_ch}{leaf_ch}"),
                format!("{leaf_ch}{stem}{leaf_ch}"),
                format!(" {stem} "),
                format!("_{stem}_"),
            ],
        ),
        PlantStage::Flowering => (
            vec![
                format!(" * "),
                format!("{leaf_ch}*{leaf_ch}"),
                format!("{leaf_ch}{stem}{leaf_ch}"),
                format!(" {stem} "),
                format!("_{stem}_"),
            ],
            vec![
                format!("  *"),
                format!("{leaf_ch}*{leaf_ch}"),
                format!("{leaf_ch}{stem}{leaf_ch}"),
                format!(" {stem} "),
                format!("_{stem}_"),
            ],
        ),
        PlantStage::Wilting => (
            vec![format!(" ."), format!(".{stem}"), format!("_{stem}_")],
            vec![format!(". "), format!("{stem}."), format!("_{stem}_")],
        ),
    };
    (primary, sway)
}

// ── Medium tier: complexity 0.35–0.6 ─────────────────────────

fn generate_medium_plant_art(genome: &PlantGenome, stage: PlantStage) -> (Vec<String>, Vec<String>) {
    // Stage-dependent height scaling
    let stage_height_scale = match stage {
        PlantStage::Seedling => 0.5,
        PlantStage::Young => 0.7,
        PlantStage::Mature | PlantStage::Flowering => 1.0,
        PlantStage::Wilting => 0.85,
    };

    let base_height = 3.0 + genome.height_factor * 2.0;
    let height = ((base_height * stage_height_scale) as usize).max(2).min(5);
    let width = (3.0 + genome.branching * 2.0) as usize;
    let width = width.max(3).min(5);

    // Choose characters based on genome
    let stem_ch = if genome.stem_thickness > 0.6 { "||" } else { "|" };
    let leaf_l = if genome.curvature > 0.5 { "(" } else { "{" };
    let leaf_r = if genome.curvature > 0.5 { ")" } else { "}" };
    let fill = if genome.leaf_area > 0.7 { "#" } else if genome.leaf_area > 0.4 { "~" } else { " " };

    let flower = if stage == PlantStage::Flowering { "*" } else if stage == PlantStage::Wilting { "." } else { "" };
    let wilt = stage == PlantStage::Wilting;

    let mut primary = Vec::new();
    let mut sway = Vec::new();

    for row in 0..height {
        if row == 0 {
            // Top/crown
            if !flower.is_empty() {
                let pad = " ".repeat(width / 2);
                primary.push(format!("{pad}{flower}{}", leaf_r.repeat(if wilt { 0 } else { 1 })));
                sway.push(format!("{}{flower}{pad}", if wilt { "" } else { leaf_l }));
            } else {
                let pad = " ".repeat(width / 2);
                primary.push(format!("{pad}{leaf_r}"));
                sway.push(format!("{leaf_l}{pad}"));
            }
        } else if row == height - 1 {
            // Base/root
            let base_w = if genome.stem_thickness > 0.5 { 4 } else { 3 };
            let base = "=".repeat(base_w);
            primary.push(base.clone());
            sway.push(base);
        } else {
            // Body rows with branching
            let has_branch = genome.branching > 0.3 && (row % 2 == 1);
            if has_branch {
                let body = format!("{leaf_l}{fill}{stem_ch}{fill}{leaf_r}");
                let body_sway = format!("{leaf_r}{fill}{stem_ch}{fill}{leaf_l}");
                primary.push(body);
                sway.push(body_sway);
            } else {
                let pad = " ".repeat(1);
                primary.push(format!("{pad}{stem_ch}{pad}"));
                sway.push(format!("{pad}{stem_ch}{pad}"));
            }
        }
    }

    (primary, sway)
}

// ── Complex tier: complexity 0.6+ ────────────────────────────

fn generate_complex_plant_art(genome: &PlantGenome, stage: PlantStage) -> (Vec<String>, Vec<String>) {
    let stage_height_scale = match stage {
        PlantStage::Seedling => 0.5,
        PlantStage::Young => 0.7,
        PlantStage::Mature | PlantStage::Flowering => 1.0,
        PlantStage::Wilting => 0.85,
    };

    let base_height = 4.0 + genome.height_factor * 3.0;
    let height = ((base_height * stage_height_scale) as usize).max(3).min(7);
    let width = (4.0 + genome.branching * 3.0) as usize;
    let width = width.max(4).min(7);

    let stem = if genome.stem_thickness > 0.7 { "||" } else { "|" };
    let leaf_l = if genome.curvature > 0.5 { "(" } else { "{" };
    let leaf_r = if genome.curvature > 0.5 { ")" } else { "}" };
    let fill = if genome.leaf_area > 0.7 { "#" }
        else if genome.leaf_area > 0.4 { "~" }
        else { ":" };

    let flower_ch = if stage == PlantStage::Flowering { "*" } else { "" };
    let wilt = stage == PlantStage::Wilting;
    let wilt_fill = if wilt { "." } else { fill };

    let mut primary = Vec::new();
    let mut sway = Vec::new();

    for row in 0..height {
        if row == 0 {
            let crown_w = (width as f32 * 0.6) as usize;
            let pad = " ".repeat((width - crown_w) / 2);
            if !flower_ch.is_empty() {
                let flowers = format!("{flower_ch}{}", leaf_r.repeat(crown_w.saturating_sub(1)));
                let flowers_sway = format!("{}{flower_ch}", leaf_l.repeat(crown_w.saturating_sub(1)));
                primary.push(format!("{pad}{flowers}"));
                sway.push(format!("{flowers_sway}{pad}"));
            } else {
                let crown = format!("{}{}", leaf_r.repeat(crown_w / 2), leaf_l.repeat(crown_w / 2));
                let crown_sway = format!("{}{}", leaf_l.repeat(crown_w / 2), leaf_r.repeat(crown_w / 2));
                primary.push(format!("{pad}{crown}"));
                sway.push(format!("{crown_sway}{pad}"));
            }
        } else if row == height - 1 {
            let base_w = if genome.stem_thickness > 0.5 { width } else { width - 1 };
            primary.push("=".repeat(base_w));
            sway.push("=".repeat(base_w));
        } else if row == height - 2 {
            let pad = " ".repeat((width - stem.len()) / 2);
            primary.push(format!("{pad}{stem}"));
            sway.push(format!("{pad}{stem}"));
        } else {
            let mid = (height - 2) / 2;
            let dist_from_mid = ((row as i32 - mid as i32).unsigned_abs()) as usize;
            let row_w = width.saturating_sub(dist_from_mid);
            let row_w = row_w.max(2);

            let inner_w = row_w.saturating_sub(2);
            let inner = wilt_fill.repeat(inner_w);
            let pad = " ".repeat((width - row_w) / 2);
            let body = format!("{pad}{leaf_l}{inner}{leaf_r}");
            let body_sway = format!("{pad}{leaf_r}{inner}{leaf_l}");
            primary.push(body);
            sway.push(body_sway);
        }
    }

    (primary, sway)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecosystem::Energy;
    use crate::genome::PlantGenome;

    fn test_genome() -> PlantGenome {
        let mut rng = rand::rng();
        PlantGenome::minimal_plant(&mut rng)
    }

    fn test_genome_with_complexity(complexity: f32) -> PlantGenome {
        let mut g = test_genome();
        g.complexity = complexity;
        g
    }

    // ── compute_stage unit tests ──────────────────────────────────

    #[test]
    fn test_seedling_stage_when_young() {
        let energy = Energy::new_with(15.0, 20.0); // 75% full
        let age = PlantAge { seconds: 5.0 };
        assert_eq!(compute_stage(&energy, &age), PlantStage::Seedling);
    }

    #[test]
    fn test_young_stage_after_seedling_duration() {
        let energy = Energy::new_with(15.0, 20.0);
        let age = PlantAge { seconds: 20.0 };
        assert_eq!(compute_stage(&energy, &age), PlantStage::Young);
    }

    #[test]
    fn test_mature_stage_after_young_duration() {
        let energy = Energy::new_with(15.0, 20.0); // 75% < 85%
        let age = PlantAge { seconds: 50.0 };
        assert_eq!(compute_stage(&energy, &age), PlantStage::Mature);
    }

    #[test]
    fn test_flowering_when_energy_high() {
        let energy = Energy::new_with(18.0, 20.0); // 90% >= 85%
        let age = PlantAge { seconds: 50.0 };
        assert_eq!(compute_stage(&energy, &age), PlantStage::Flowering);
    }

    #[test]
    fn test_wilting_overrides_all_stages() {
        let energy = Energy::new_with(2.0, 20.0); // 10% < 20%
        for age_secs in [0.0, 10.0, 30.0, 50.0] {
            let age = PlantAge { seconds: age_secs };
            assert_eq!(
                compute_stage(&energy, &age),
                PlantStage::Wilting,
                "Expected Wilting at age={age_secs}s with 10% energy"
            );
        }
    }

    #[test]
    fn test_full_stage_progression() {
        let max_e = 20.0;
        let mut age = PlantAge { seconds: 0.0 };

        // Phase 1: Seedling (age < 15s, energy >= 20%)
        let energy = Energy::new_with(max_e * 0.5, max_e);
        assert_eq!(compute_stage(&energy, &age), PlantStage::Seedling);

        // Phase 2: Young (age 15-40s)
        age.seconds = 25.0;
        assert_eq!(compute_stage(&energy, &age), PlantStage::Young);

        // Phase 3: Mature (age > 40s, energy < 85%)
        age.seconds = 50.0;
        let energy_moderate = Energy::new_with(max_e * 0.7, max_e);
        assert_eq!(compute_stage(&energy_moderate, &age), PlantStage::Mature);

        // Phase 4: Flowering (age > 40s, energy >= 85%)
        let energy_high = Energy::new_with(max_e * 0.9, max_e);
        assert_eq!(compute_stage(&energy_high, &age), PlantStage::Flowering);

        // Phase 5: Wilting (energy < 20%)
        let energy_low = Energy::new_with(max_e * 0.1, max_e);
        assert_eq!(compute_stage(&energy_low, &age), PlantStage::Wilting);
    }

    // ── ECS lifecycle system tests ────────────────────────────────

    fn spawn_ecs_plant(world: &mut hecs::World, energy_frac: f32) -> hecs::Entity {
        let genome = test_genome();
        let max_e = 20.0;
        let stage = PlantStage::Seedling;
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage);
        world.spawn((
            genome,
            stage,
            PlantAge { seconds: 0.0 },
            Energy::new_with(max_e * energy_frac, max_e),
            appearance,
            bbox,
            AnimationState::new(0.8),
        ))
    }

    #[test]
    fn test_ecs_seedling_to_young_transition() {
        let mut world = hecs::World::new();
        let entity = spawn_ecs_plant(&mut world, 0.5);

        for _ in 0..16 {
            plant_lifecycle_system(&mut world, 1.0);
        }

        let stage = *world.get::<&PlantStage>(entity).unwrap();
        assert_eq!(stage, PlantStage::Young);
    }

    #[test]
    fn test_ecs_young_to_mature_transition() {
        let mut world = hecs::World::new();
        let entity = spawn_ecs_plant(&mut world, 0.5);

        for _ in 0..41 {
            plant_lifecycle_system(&mut world, 1.0);
        }

        let stage = *world.get::<&PlantStage>(entity).unwrap();
        assert_eq!(stage, PlantStage::Mature);
    }

    #[test]
    fn test_ecs_mature_to_flowering_with_high_energy() {
        let mut world = hecs::World::new();
        let entity = spawn_ecs_plant(&mut world, 0.9);

        for _ in 0..41 {
            plant_lifecycle_system(&mut world, 1.0);
        }

        let stage = *world.get::<&PlantStage>(entity).unwrap();
        assert_eq!(stage, PlantStage::Flowering);
    }

    #[test]
    fn test_ecs_wilting_on_energy_drain() {
        let mut world = hecs::World::new();
        let entity = spawn_ecs_plant(&mut world, 0.5);

        // Advance past seedling
        for _ in 0..20 {
            plant_lifecycle_system(&mut world, 1.0);
        }
        assert_eq!(*world.get::<&PlantStage>(entity).unwrap(), PlantStage::Young);

        // Drain energy below wilting threshold
        {
            let mut energy = world.get::<&mut Energy>(entity).unwrap();
            energy.current = energy.max * 0.1;
        }

        plant_lifecycle_system(&mut world, 1.0);
        assert_eq!(*world.get::<&PlantStage>(entity).unwrap(), PlantStage::Wilting);
    }

    #[test]
    fn test_ecs_full_lifecycle_seedling_to_flowering() {
        let mut world = hecs::World::new();
        let entity = spawn_ecs_plant(&mut world, 0.9);

        let mut stages_seen = vec![PlantStage::Seedling];

        for _ in 0..50 {
            plant_lifecycle_system(&mut world, 1.0);
            let stage = *world.get::<&PlantStage>(entity).unwrap();
            if stages_seen.last() != Some(&stage) {
                stages_seen.push(stage);
            }
        }

        assert!(
            stages_seen.contains(&PlantStage::Seedling),
            "Never saw Seedling: {stages_seen:?}"
        );
        assert!(
            stages_seen.contains(&PlantStage::Young),
            "Never saw Young: {stages_seen:?}"
        );
        assert!(
            stages_seen.contains(&PlantStage::Flowering),
            "Never saw Flowering (energy was 90%): {stages_seen:?}"
        );
    }

    // ── Appearance change tests ───────────────────────────────────

    #[test]
    fn test_appearance_differs_between_stages() {
        let genome = test_genome_with_complexity(0.25);
        let stages = [
            PlantStage::Seedling,
            PlantStage::Young,
            PlantStage::Mature,
            PlantStage::Flowering,
            PlantStage::Wilting,
        ];

        let arts: Vec<(Vec<String>, Vec<String>)> = stages
            .iter()
            .map(|&s| generate_plant_art(&genome, s))
            .collect();

        for i in 0..stages.len() - 1 {
            assert_ne!(
                arts[i].0, arts[i + 1].0,
                "Art for {:?} and {:?} should differ (complexity=0.25 sprout tier)",
                stages[i], stages[i + 1]
            );
        }
    }

    #[test]
    fn test_sprout_height_grows_with_stage() {
        let genome = test_genome_with_complexity(0.25);
        let seedling = generate_plant_art(&genome, PlantStage::Seedling);
        let young = generate_plant_art(&genome, PlantStage::Young);
        let mature = generate_plant_art(&genome, PlantStage::Mature);
        let flowering = generate_plant_art(&genome, PlantStage::Flowering);

        assert!(
            seedling.0.len() < young.0.len(),
            "Seedling ({} rows) should be shorter than Young ({} rows)",
            seedling.0.len(),
            young.0.len()
        );
        assert!(
            young.0.len() < mature.0.len(),
            "Young ({} rows) should be shorter than Mature ({} rows)",
            young.0.len(),
            mature.0.len()
        );
        assert!(
            mature.0.len() <= flowering.0.len(),
            "Mature ({} rows) should not be taller than Flowering ({} rows)",
            mature.0.len(),
            flowering.0.len()
        );
    }

    #[test]
    fn test_medium_height_grows_with_stage() {
        let genome = test_genome_with_complexity(0.45);
        let seedling = generate_plant_art(&genome, PlantStage::Seedling);
        let mature = generate_plant_art(&genome, PlantStage::Mature);

        assert!(
            seedling.0.len() < mature.0.len(),
            "Medium seedling ({} rows) should be shorter than mature ({} rows)",
            seedling.0.len(),
            mature.0.len()
        );
    }

    #[test]
    fn test_complex_height_grows_with_stage() {
        let genome = test_genome_with_complexity(0.7);
        let seedling = generate_plant_art(&genome, PlantStage::Seedling);
        let mature = generate_plant_art(&genome, PlantStage::Mature);

        assert!(
            seedling.0.len() < mature.0.len(),
            "Complex seedling ({} rows) should be shorter than mature ({} rows)",
            seedling.0.len(),
            mature.0.len()
        );
    }

    #[test]
    fn test_ecs_appearance_updates_on_stage_change() {
        let mut world = hecs::World::new();
        let entity = spawn_ecs_plant(&mut world, 0.5);

        let initial_h = world.get::<&BoundingBox>(entity).unwrap().h;

        // Tick past seedling → young (15s)
        for _ in 0..16 {
            plant_lifecycle_system(&mut world, 1.0);
        }

        let young_h = world.get::<&BoundingBox>(entity).unwrap().h;

        // Tick past young → mature (40s)
        for _ in 0..25 {
            plant_lifecycle_system(&mut world, 1.0);
        }

        let mature_h = world.get::<&BoundingBox>(entity).unwrap().h;

        assert!(
            young_h > initial_h,
            "Young bbox height ({young_h}) should be > Seedling ({initial_h})"
        );
        assert!(
            mature_h > young_h,
            "Mature bbox height ({mature_h}) should be > Young ({young_h})"
        );
    }

    // ── Tier selection tests ──────────────────────────────────────

    #[test]
    fn test_spore_tier_for_low_complexity() {
        let genome = test_genome_with_complexity(0.10);
        let (art, _) = generate_plant_art(&genome, PlantStage::Mature);
        assert_eq!(art.len(), 1, "Spore tier should produce exactly 1 row");
    }

    #[test]
    fn test_sprout_tier_at_all_stages() {
        let genome = test_genome_with_complexity(0.25);
        for stage in [PlantStage::Seedling, PlantStage::Young, PlantStage::Mature, PlantStage::Flowering, PlantStage::Wilting] {
            let (art, _) = generate_plant_art(&genome, stage);
            assert!(
                art.len() >= 2,
                "Sprout tier at {stage:?} should have >= 2 rows, got {}",
                art.len()
            );
        }
    }

    #[test]
    fn test_tier_selection_uses_raw_complexity() {
        // A plant with complexity 0.3 should ALWAYS be in sprout tier (0.15-0.35)
        // regardless of stage — no more stage-scaled tier jumping
        let genome = test_genome_with_complexity(0.30);
        for stage in [PlantStage::Seedling, PlantStage::Young, PlantStage::Mature] {
            let (art, _) = generate_plant_art(&genome, stage);
            assert!(
                art.len() >= 2,
                "Complexity 0.30 plant at {stage:?} should be sprout tier (>=2 rows), got {}",
                art.len()
            );
        }
    }

    // ── Death system tests ────────────────────────────────────────

    #[test]
    fn test_plant_dies_by_age_via_death_system() {
        use crate::ecosystem::{Age, death_system};

        let mut world = hecs::World::new();
        let genome = test_genome();
        let stage = PlantStage::Seedling;
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage);

        let entity = world.spawn((
            genome,
            stage,
            PlantAge { seconds: 0.0 },
            Energy::new_with(10.0, 20.0),
            appearance,
            bbox,
            AnimationState::new(0.8),
            Age { ticks: 100, max_ticks: 100 },
        ));

        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 1, "Plant at max age should be removed");
        assert!(world.get::<&PlantGenome>(entity).is_err(), "Entity should be despawned");
    }

    #[test]
    fn test_plant_dies_by_energy_depletion() {
        use crate::ecosystem::{Age, death_system};

        let mut world = hecs::World::new();
        let genome = test_genome();
        let stage = PlantStage::Wilting;
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage);

        let entity = world.spawn((
            genome,
            stage,
            PlantAge { seconds: 50.0 },
            Energy::new_with(0.0, 20.0),
            appearance,
            bbox,
            AnimationState::new(0.8),
            Age { ticks: 0, max_ticks: 10000 },
        ));

        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 1, "Plant with zero energy should be removed");
        assert!(world.get::<&PlantGenome>(entity).is_err(), "Entity should be despawned");
    }

    #[test]
    fn test_plant_survives_when_healthy() {
        use crate::ecosystem::{Age, death_system};

        let mut world = hecs::World::new();
        let genome = test_genome();
        let stage = PlantStage::Mature;
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage);

        let entity = world.spawn((
            genome,
            stage,
            PlantAge { seconds: 50.0 },
            Energy::new_with(15.0, 20.0),
            appearance,
            bbox,
            AnimationState::new(0.8),
            Age { ticks: 500, max_ticks: 10000 },
        ));

        let result = death_system(&mut world);
        assert_eq!(result.total_removed, 0, "Healthy plant should survive");
        assert!(world.get::<&PlantGenome>(entity).is_ok(), "Entity should still exist");
    }

    #[test]
    fn test_plant_energy_drains_via_maintenance() {
        use crate::ecosystem;
        use crate::environment::Environment;
        use crate::phenotype;

        let mut world = hecs::World::new();
        let genome = test_genome();
        let physics = phenotype::derive_plant_physics(&genome);
        let max_e = physics.max_energy;

        world.spawn((
            genome.clone(),
            physics,
            crate::components::Position { x: 10.0, y: 10.0 },
            Energy::new_with(max_e, max_e),
        ));

        // Use zero light so photosynthesis gain = 0 → only maintenance drains
        let mut env = Environment::default();
        env.light_level = 0.0;

        for _ in 0..2000 {
            ecosystem::metabolism_system(&mut world, 0.05, &env, 20.0);
        }

        let energy = world.query_mut::<&Energy>().into_iter().next().unwrap();
        assert!(
            energy.fraction() < 0.90,
            "After 2000 ticks in darkness, energy should drain via maintenance, got {:.1}%",
            energy.fraction() * 100.0
        );
    }

    #[test]
    fn test_plant_energy_does_not_stay_at_flowering_threshold() {
        use crate::ecosystem;
        use crate::environment::Environment;
        use crate::phenotype;

        let mut world = hecs::World::new();
        // Fully deterministic genome so equilibrium is predictable (~81%)
        let mut genome = test_genome();
        genome.photosynthesis_rate = 1.0;
        genome.stem_thickness = 0.4;
        genome.height_factor = 0.5;
        genome.leaf_area = 0.5;
        genome.branching = 0.3;
        genome.complexity = 0.25;
        genome.max_energy_factor = 1.0;
        let physics = phenotype::derive_plant_physics(&genome);
        let max_e = physics.max_energy;

        world.spawn((
            genome.clone(),
            physics,
            crate::components::Position { x: 10.0, y: 5.0 },
            // Start at 50% energy — equilibrium should approach ~81% from below
            Energy::new_with(max_e * 0.5, max_e),
        ));

        let env = Environment::default();

        // Run metabolism for 20000 ticks (1000s sim time) — enough to approach equilibrium
        for _ in 0..20000 {
            ecosystem::metabolism_system(&mut world, 0.05, &env, 20.0);
        }

        let energy = world.query_mut::<&Energy>().into_iter().next().unwrap();
        let frac = energy.fraction();
        assert!(
            frac < 0.85,
            "Plant energy should equilibrate below flowering threshold (85%), got {:.1}%",
            frac * 100.0
        );
        assert!(
            frac > 0.50,
            "Plant energy should recover from 50% toward equilibrium (~81%), got {:.1}%",
            frac * 100.0
        );
    }
}
