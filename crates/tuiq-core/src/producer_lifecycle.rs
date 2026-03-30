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
    let mut updates: Vec<(hecs::Entity, ProducerGenome, ProducerStage)> = Vec::new();

    for (entity, age, genome, state, stage, energy) in &mut world.query::<(
        hecs::Entity,
        &mut ProducerAge,
        &ProducerGenome,
        &ProducerState,
        &mut ProducerStage,
        &Energy,
    )>() {
        age.seconds += dt;
        let new_stage = compute_stage(genome, state, energy, age);
        if new_stage != *stage {
            *stage = new_stage;
            updates.push((entity, genome.clone(), new_stage));
        }
    }

    for (entity, genome, stage) in updates {
        let (appearance, bbox) = build_appearance_from_genome(&genome, stage);
        let _ = world.insert(entity, (appearance, bbox, AnimationState::new(0.8)));
    }
}

/// Build full appearance for a producer colony from its genome and current stage.
pub fn build_appearance_from_genome(
    genome: &ProducerGenome,
    stage: ProducerStage,
) -> (Appearance, BoundingBox) {
    let (rows, sway_rows) = generate_producer_art(genome, stage);

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
) -> (Vec<String>, Vec<String>) {
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use crate::genome::ProducerGenome;

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
        ));

        producer_lifecycle_system(&mut world, 1.0);
        let live_stage = *world.get::<&ProducerStage>(entity).unwrap();
        assert_eq!(live_stage, ProducerStage::Broadcasting);
    }
}
