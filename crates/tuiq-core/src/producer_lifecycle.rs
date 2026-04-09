//! Producer lifecycle: colony-driven growth stages and appearance updates.
//! Producer colonies are sessile photosynthesizers whose visible form emerges
//! from their genome and current biomass state rather than from hardcoded taxa.

use crate::components::*;
use crate::ecosystem::Energy;
use crate::genome::ProducerGenome;

const MIN_BROADCASTING_AGE: f32 = 18.0;
const COLLAPSING_ENERGY_THRESHOLD: f32 = 0.12;
const BROADCASTING_ENERGY_THRESHOLD: f32 = 0.70;

/// Compute the current producer stage from reserve status, biomass fill, and age.
pub fn compute_stage(
    genome: &ProducerGenome,
    state: &ProducerState,
    energy: &Energy,
    age: &ProducerAge,
) -> ProducerStage {
    let reserve_fraction = energy.fraction();
    let support_fill = state.structural_biomass / genome.support_target_biomass().max(0.1);
    let active_fill = state.leaf_biomass / genome.active_target_biomass().max(0.1);
    let maturity = ((support_fill + active_fill) * 0.5).clamp(0.0, 1.5);

    if reserve_fraction < COLLAPSING_ENERGY_THRESHOLD
        || active_fill < 0.10
        || state.stress_time > 6.0
    {
        ProducerStage::Collapsing
    } else if maturity < 0.25 {
        ProducerStage::Cell
    } else if maturity < 0.55 {
        ProducerStage::Patch
    } else if age.seconds >= MIN_BROADCASTING_AGE
        && reserve_fraction >= BROADCASTING_ENERGY_THRESHOLD
    {
        ProducerStage::Broadcasting
    } else {
        ProducerStage::Mature
    }
}

/// ECS system: advance producer age, recompute stage, update appearance when stage changes.
pub fn producer_lifecycle_system(world: &mut hecs::World, dt: f32) {
    let mut updates: Vec<(hecs::Entity, ProducerGenome, ProducerStage, Option<f32>)> = Vec::new();

    for (entity, age, genome, state, stage, energy, pos, bbox, rooted) in &mut world.query::<(
        hecs::Entity,
        &mut ProducerAge,
        &ProducerGenome,
        &ProducerState,
        &mut ProducerStage,
        &Energy,
        &Position,
        &BoundingBox,
        Option<&RootedMacrophyte>,
    )>() {
        age.seconds += dt;
        let new_stage = compute_stage(genome, state, energy, age);
        if new_stage != *stage {
            *stage = new_stage;
            let bottom_anchor = rooted.map(|_| pos.y + bbox.h - 1.0);
            updates.push((entity, genome.clone(), new_stage, bottom_anchor));
        }
    }

    for (entity, genome, stage, bottom_anchor) in updates {
        let rooted = bottom_anchor.is_some();
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage, rooted);
        let new_bottom = bottom_anchor.map(|bottom| (bottom - bbox.h + 1.0).max(0.0));
        let _ = world.insert(entity, (appearance, bbox, AnimationState::new(0.8)));
        if let Some(new_y) = new_bottom {
            if let Ok(mut pos) = world.get::<&mut Position>(entity) {
                pos.y = new_y;
            }
        }
    }
}

/// Build full appearance for a producer colony from its genome and current stage.
pub fn build_appearance_from_genome(
    genome: &ProducerGenome,
    stage: ProducerStage,
    rooted: bool,
) -> (Appearance, BoundingBox) {
    let (rows, sway_rows) = generate_producer_art(genome, stage, rooted);

    let w = rows.iter().map(|r| r.len()).max().unwrap_or(1) as f32;
    let h = rows.len() as f32;

    let frame = AsciiFrame::from_rows(rows.iter().map(|s| s.as_str()).collect());
    let sway_frame = AsciiFrame::from_rows(sway_rows.iter().map(|s| s.as_str()).collect());

    let appearance = Appearance {
        frame_sets: vec![vec![frame.clone(), sway_frame], vec![frame]],
        facing: Direction::Right,
        color_index: genome.color_index(),
    };

    (appearance, BoundingBox { w, h })
}

fn generate_producer_art(
    genome: &ProducerGenome,
    stage: ProducerStage,
    rooted: bool,
) -> (Vec<String>, Vec<String>) {
    if rooted && matches!(stage, ProducerStage::Cell | ProducerStage::Patch) {
        return rooted_sprout_art(stage);
    }
    let complexity = genome.complexity.max(0.04);
    if complexity < 0.12 {
        speck_art(stage)
    } else if complexity < 0.35 {
        tuft_art(genome, stage)
    } else if complexity < 0.65 {
        mat_art(genome, stage)
    } else {
        plume_art(genome, stage)
    }
}

fn rooted_sprout_art(stage: ProducerStage) -> (Vec<String>, Vec<String>) {
    match stage {
        ProducerStage::Cell => (
            vec!["'".to_string(), "|".to_string()],
            vec!["`".to_string(), "|".to_string()],
        ),
        ProducerStage::Patch => (
            vec![" ' ".to_string(), "\\|/".to_string()],
            vec![" ` ".to_string(), "/|\\".to_string()],
        ),
        _ => speck_art(stage),
    }
}

fn speck_art(stage: ProducerStage) -> (Vec<String>, Vec<String>) {
    let ch = match stage {
        ProducerStage::Cell => '.',
        ProducerStage::Patch => ':',
        ProducerStage::Mature => 'o',
        ProducerStage::Broadcasting => '*',
        ProducerStage::Collapsing => '\'',
    };
    let s = ch.to_string();
    (vec![s.clone()], vec![s])
}

fn tuft_art(genome: &ProducerGenome, stage: ProducerStage) -> (Vec<String>, Vec<String>) {
    let body = if genome.leaf_area > 0.55 { "~" } else { ":" };
    let top = match stage {
        ProducerStage::Broadcasting => "*",
        ProducerStage::Collapsing => "'",
        _ => body,
    };
    match stage {
        ProducerStage::Cell => (vec![body.to_string()], vec![body.to_string()]),
        ProducerStage::Patch => (
            vec![format!(" {top}"), format!("{body}{body}")],
            vec![format!("{top} "), format!("{body}{body}")],
        ),
        ProducerStage::Mature | ProducerStage::Broadcasting => (
            vec![
                format!(" {top} "),
                format!("{body}{body}{body}"),
                format!(" {body} "),
            ],
            vec![
                format!("  {top}"),
                format!("{body}{body}{body}"),
                format!(" {body} "),
            ],
        ),
        ProducerStage::Collapsing => (
            vec![format!(" {top}"), format!(" {body}")],
            vec![format!("{top} "), format!(" {body}")],
        ),
    }
}

fn mat_art(genome: &ProducerGenome, stage: ProducerStage) -> (Vec<String>, Vec<String>) {
    let spread = if genome.branching > 0.5 { "#" } else { "~" };
    let edge = if genome.curvature > 0.5 { "(" } else { "{" };
    let cap = match stage {
        ProducerStage::Broadcasting => "*",
        ProducerStage::Collapsing => ".",
        _ => spread,
    };
    match stage {
        ProducerStage::Cell => speck_art(stage),
        ProducerStage::Patch => (
            vec![format!(" {cap} "), format!("{edge}{spread}{edge}")],
            vec![format!("  {cap}"), format!("{edge}{spread}{edge}")],
        ),
        ProducerStage::Mature | ProducerStage::Broadcasting => (
            vec![
                format!(" {cap}{cap} "),
                format!("{edge}{spread}{spread}{edge}"),
                format!(" {spread}{spread} "),
            ],
            vec![
                format!("  {cap}{cap}"),
                format!("{edge}{spread}{spread}{edge}"),
                format!(" {spread}{spread} "),
            ],
        ),
        ProducerStage::Collapsing => (
            vec![format!(" {cap} "), format!(" {spread} ")],
            vec![format!("{cap}  "), format!(" {spread} ")],
        ),
    }
}

fn plume_art(genome: &ProducerGenome, stage: ProducerStage) -> (Vec<String>, Vec<String>) {
    let branch = if genome.curvature > 0.5 { "/" } else { "|" };
    let plume = if genome.leaf_area > 0.65 { "*" } else { "~" };
    let cap = match stage {
        ProducerStage::Broadcasting => "*",
        ProducerStage::Collapsing => ".",
        _ => plume,
    };
    match stage {
        ProducerStage::Cell => speck_art(stage),
        ProducerStage::Patch => (
            vec![
                format!(" {cap} "),
                format!(" {branch} "),
                format!("~{branch}~"),
            ],
            vec![
                format!("  {cap}"),
                format!(" {branch} "),
                format!("~{branch}~"),
            ],
        ),
        ProducerStage::Mature | ProducerStage::Broadcasting => (
            vec![
                format!(" {cap}{cap} "),
                format!("{plume}{branch}{plume}"),
                format!(" {branch}{branch} "),
                format!("~{branch}{branch}~"),
            ],
            vec![
                format!("  {cap}{cap}"),
                format!("{plume}{branch}{plume}"),
                format!(" {branch}{branch} "),
                format!("~{branch}{branch}~"),
            ],
        ),
        ProducerStage::Collapsing => (
            vec![
                format!(" {cap} "),
                format!(" {branch} "),
                format!(" {branch} "),
            ],
            vec![
                format!("{cap}  "),
                format!(" {branch} "),
                format!(" {branch} "),
            ],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::ProducerGenome;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_genome() -> ProducerGenome {
        let mut genome = ProducerGenome::minimal_producer(&mut StdRng::seed_from_u64(42));
        genome.complexity = 0.45;
        genome
    }

    fn test_state(genome: &ProducerGenome, support_fill: f32, active_fill: f32) -> ProducerState {
        ProducerState {
            structural_biomass: genome.support_target_biomass() * support_fill,
            leaf_biomass: genome.active_target_biomass() * active_fill,
            belowground_reserve: genome.support_target_biomass() * 0.18,
            meristem_bank: 0.25,
            epiphyte_load: 0.0,
            seed_cooldown: 0.0,
            clonal_cooldown: 0.0,
            stress_time: 0.0,
            propagule_kind: None,
        }
    }

    #[test]
    fn stage_progresses_from_cell_to_mature() {
        let genome = test_genome();
        let age = ProducerAge { seconds: 6.0 };
        let energy = Energy::new_with(5.0, 8.0);
        assert_eq!(
            compute_stage(&genome, &test_state(&genome, 0.12, 0.12), &energy, &age),
            ProducerStage::Cell
        );
        assert_eq!(
            compute_stage(&genome, &test_state(&genome, 0.45, 0.45), &energy, &age),
            ProducerStage::Patch
        );
        assert_eq!(
            compute_stage(&genome, &test_state(&genome, 0.90, 0.90), &energy, &age),
            ProducerStage::Mature
        );
    }

    #[test]
    fn broadcasting_requires_age_and_reserve() {
        let genome = test_genome();
        let state = test_state(&genome, 0.95, 0.95);
        let young = ProducerAge { seconds: 4.0 };
        let adult = ProducerAge { seconds: 24.0 };
        let high_energy = Energy::new_with(7.5, 8.0);
        assert_eq!(
            compute_stage(&genome, &state, &high_energy, &young),
            ProducerStage::Mature
        );
        assert_eq!(
            compute_stage(&genome, &state, &high_energy, &adult),
            ProducerStage::Broadcasting
        );
    }

    #[test]
    fn collapsing_stage_triggers_on_low_energy() {
        let genome = test_genome();
        let state = test_state(&genome, 0.8, 0.8);
        let age = ProducerAge { seconds: 24.0 };
        let energy = Energy::new_with(0.2, 8.0);
        assert_eq!(
            compute_stage(&genome, &state, &energy, &age),
            ProducerStage::Collapsing
        );
    }

    #[test]
    fn lifecycle_system_updates_stage_component() {
        let mut world = hecs::World::new();
        let genome = test_genome();
        let stage = ProducerStage::Cell;
        let entity = world.spawn((
            ProducerAge { seconds: 24.0 },
            genome.clone(),
            test_state(&genome, 0.9, 0.9),
            stage,
            Energy::new_with(7.5, 8.0),
            Appearance {
                frame_sets: vec![
                    vec![AsciiFrame::from_rows(vec!["."])],
                    vec![AsciiFrame::from_rows(vec!["."])],
                ],
                facing: Direction::Right,
                color_index: 0,
            },
            BoundingBox { w: 1.0, h: 1.0 },
            Position { x: 6.0, y: 12.0 },
        ));

        producer_lifecycle_system(&mut world, 1.0);
        let live_stage = *world.get::<&ProducerStage>(entity).unwrap();
        assert_eq!(live_stage, ProducerStage::Broadcasting);
    }

    #[test]
    fn rooted_macrophyte_preserves_bottom_anchor_when_stage_changes() {
        let mut world = hecs::World::new();
        let genome = test_genome();
        let stage = ProducerStage::Cell;
        let initial_bbox = BoundingBox { w: 1.0, h: 1.0 };
        let initial_pos = Position { x: 8.0, y: 16.0 };
        let initial_bottom = initial_pos.y + initial_bbox.h - 1.0;

        let entity = world.spawn((
            ProducerAge { seconds: 24.0 },
            genome.clone(),
            test_state(&genome, 0.9, 0.9),
            stage,
            Energy::new_with(7.5, 8.0),
            Appearance {
                frame_sets: vec![
                    vec![AsciiFrame::from_rows(vec!["."])],
                    vec![AsciiFrame::from_rows(vec!["."])],
                ],
                facing: Direction::Right,
                color_index: 0,
            },
            initial_bbox,
            initial_pos,
            RootedMacrophyte,
        ));

        producer_lifecycle_system(&mut world, 1.0);
        let pos = world.get::<&Position>(entity).unwrap();
        let bbox = world.get::<&BoundingBox>(entity).unwrap();
        let new_bottom = pos.y + bbox.h - 1.0;
        assert!(
            (new_bottom - initial_bottom).abs() < 0.01,
            "Rooted producer should preserve bottom anchor: initial={initial_bottom:.2} new={new_bottom:.2}"
        );
    }

    #[test]
    fn rooted_low_stage_art_rises_above_substrate() {
        let genome = test_genome();

        let (cell_appearance, cell_bbox) =
            build_appearance_from_genome(&genome, ProducerStage::Cell, true);
        let cell_rows = &cell_appearance.frame_sets[0][0].rows;
        assert!(
            cell_bbox.h >= 2.0,
            "Rooted cell stage should render above the substrate, got height {}",
            cell_bbox.h
        );
        assert_eq!(cell_rows, &vec!["'".to_string(), "|".to_string()]);

        let (patch_appearance, patch_bbox) =
            build_appearance_from_genome(&genome, ProducerStage::Patch, true);
        let patch_rows = &patch_appearance.frame_sets[0][0].rows;
        assert!(
            patch_bbox.h >= 2.0,
            "Rooted patch stage should render above the substrate, got height {}",
            patch_bbox.h
        );
        assert_eq!(patch_rows, &vec![" ' ".to_string(), "\\|/".to_string()]);
    }
}
