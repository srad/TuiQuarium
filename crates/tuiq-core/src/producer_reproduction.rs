//! Producer (plant) reproduction: seed dispersal, clonal spread, and establishment.

use crate::components::*;
use crate::ecosystem::{Detritus, Energy};
use crate::genetics;
use crate::genome::{self, ProducerGenome};

use hecs::Entity;
use rand::seq::SliceRandom;
use rand::RngExt;

impl super::AquariumSim {
    pub(crate) fn spawn_labile_detritus_from_producers(&mut self) {
        const DETRITUS_PACKET_ENERGY: f32 = 1.4;
        if self.pending_labile_detritus_energy < DETRITUS_PACKET_ENERGY {
            return;
        }

        let producer_positions: Vec<Position> = (&mut self.world.query::<(Entity, &Position)>())
            .into_iter()
            .filter_map(|(entity, pos)| {
                if self.world.get::<&ProducerGenome>(entity).is_ok() {
                    Some(pos.clone())
                } else {
                    None
                }
            })
            .collect();
        if producer_positions.is_empty() {
            return;
        }

        let existing_detritus = (&mut self.world.query::<&Detritus>()).into_iter().count();
        let max_packets = ((self.tank_width as usize * self.tank_height as usize) / 160).max(6);
        let available_slots = max_packets.saturating_sub(existing_detritus);
        if available_slots == 0 {
            return;
        }

        let spawn_count = ((self.pending_labile_detritus_energy / DETRITUS_PACKET_ENERGY).floor()
            as usize)
            .min(available_slots)
            .min(4);
        for _ in 0..spawn_count {
            let anchor = &producer_positions[self.rng.random_range(0..producer_positions.len())];
            let x = (anchor.x + self.rng.random_range(-1.5..1.5))
                .clamp(2.0, self.tank_width as f32 - 2.0);
            let y = (anchor.y + self.rng.random_range(-1.0..1.0))
                .clamp(2.0, self.tank_height as f32 - 2.0);
            self.spawn_detritus(x, y, DETRITUS_PACKET_ENERGY);
            self.pending_labile_detritus_energy =
                (self.pending_labile_detritus_energy - DETRITUS_PACKET_ENERGY).max(0.0);
        }
        self.pending_labile_detritus_energy = self.pending_labile_detritus_energy.min(12.0);
    }

    pub(crate) fn producer_establishment_chance(&self, x: f32, y: f32, kind: ProducerOrigin, hardiness: f32) -> f32 {
        let local_light = self.light_field.sample_light(
            x,
            y,
            2.0,
            self.env.light_level,
            self.tank_height as f32,
            &self.nutrients,
            &self.calibration.ecology,
        );
        let local_crowding = self
            .grid
            .neighbors(x, y, 2.5)
            .into_iter()
            .filter(|&entity| self.world.get::<&genome::ProducerGenome>(entity).is_ok())
            .count()
            .saturating_sub(1) as f32;
        let patch_crowding = self
            .grid
            .neighbors(x, y, 6.0)
            .into_iter()
            .filter(|&entity| self.world.get::<&genome::ProducerGenome>(entity).is_ok())
            .count()
            .saturating_sub(1) as f32;
        let base = match kind {
            // Simulation assumption: local fragments establish more reliably but
            // disperse less far than broadcast propagules in the unified model.
            ProducerOrigin::Broadcast => 0.46,
            ProducerOrigin::Fragment => 0.80,
        };

        // Research note: dense established producer stands suppress local
        // recruitment, so establishment is penalized more strongly inside an
        // existing patch than in open water/light gaps.
        let local_open_factor = match kind {
            ProducerOrigin::Broadcast => 1.0 / (1.0 + local_crowding * 0.75),
            ProducerOrigin::Fragment => 1.0 / (1.0 + local_crowding * 1.35),
        };
        let patch_open_factor = match kind {
            ProducerOrigin::Broadcast => 1.0 / (1.0 + patch_crowding * 0.20),
            ProducerOrigin::Fragment => 1.0 / (1.0 + patch_crowding * 0.32),
        };
        let depth_fraction = (y / self.tank_height.max(1) as f32).clamp(0.0, 1.0);
        // Research note: submerged macrophyte establishment falls off sharply
        // below the productive light band, so recruitment should be penalized
        // well before the darkest bottom cells.
        let depth_factor = (1.0 - (depth_fraction - 0.62).max(0.0) * 1.8).clamp(0.20, 1.0);

        // Substrate affects establishment: rocky favors hardy producers, planted
        // zones give a general establishment bonus.
        let substrate_mod = self.substrate.establishment_modifier(x, hardiness);

        (base * (0.28 + local_light * 0.82) * local_open_factor * patch_open_factor * depth_factor * substrate_mod)
            .clamp(0.0, 0.95)
    }

    pub(crate) fn reproduce_producers(&mut self) -> u64 {
        #[derive(Clone)]
        struct Candidate {
            entity: hecs::Entity,
            x: f32,
            y: f32,
            genome: genome::ProducerGenome,
            state: ProducerState,
            energy_current: f32,
            energy_max: f32,
        }

        #[derive(Clone, Copy)]
        struct ParentUpdate {
            entity: hecs::Entity,
            reserve_cost: f32,
            seed_cd: f32,
            clonal_cd: f32,
        }

        #[derive(Clone)]
        struct SpawnRequest {
            pos: Position,
            genome: genome::ProducerGenome,
            kind: ProducerOrigin,
        }

        let mut raw_producer_count = 0usize;
        let mut active_producer_count = 0usize;
        for (_, state) in &mut self
            .world
            .query::<(&genome::ProducerGenome, &ProducerState)>()
        {
            raw_producer_count += 1;
            if state.total_biomass() > 0.45 {
                active_producer_count += 1;
            }
        }
        let max_producers = self.max_producers();
        let hard_max_producers = max_producers * 2;
        // Research note: stand-forming aquatic plants replace weakening ramets
        // before the entire patch disappears, so recruitment is limited by
        // active occupancy rather than by a strict entity-count freeze.
        if active_producer_count >= max_producers || raw_producer_count >= hard_max_producers {
            return 0;
        }

        let mut candidates: Vec<Candidate> = (&mut self.world.query::<(
            hecs::Entity,
            &Position,
            &genome::ProducerGenome,
            &ProducerState,
            &Energy,
        )>())
            .into_iter()
            .map(|(entity, pos, genome, state, energy)| Candidate {
                entity,
                x: pos.x,
                y: pos.y,
                genome: genome.clone(),
                state: state.clone(),
                energy_current: energy.current,
                energy_max: energy.max,
            })
            .collect();
        candidates.shuffle(&mut self.rng);

        let tw = self.tank_width as f32;
        let th = self.tank_height as f32;
        let mut parent_updates = Vec::new();
        let mut spawns = Vec::new();

        for candidate in candidates {
            if active_producer_count + spawns.len() >= max_producers
                || raw_producer_count + spawns.len() >= hard_max_producers
            {
                break;
            }

            let structural_fill = candidate.state.structural_biomass
                / candidate.genome.support_target_biomass().max(0.1);
            let leaf_fill =
                candidate.state.leaf_biomass / candidate.genome.active_target_biomass().max(0.1);
            let belowground_target = candidate.genome.support_target_biomass()
                * (0.22
                    + candidate.genome.hardiness * 0.22
                    + candidate.genome.clonal_spread * 0.18);
            let below_fill = candidate.state.belowground_reserve / belowground_target.max(0.05);
            let maturity = ((structural_fill + leaf_fill + below_fill) / 3.0).clamp(0.0, 1.5);
            let reserve_ratio =
                (candidate.energy_current / candidate.energy_max.max(1e-3)).clamp(0.0, 1.0);
            if maturity < 0.55 || reserve_ratio < 0.44 {
                continue;
            }

            let mut reserve_cost = 0.0;
            let mut seed_cd = candidate.state.seed_cooldown;
            let mut clonal_cd = candidate.state.clonal_cooldown;

            // Research note: reproductive effort is derived from realized biomass
            // maturity and reserve surplus instead of a lifecycle-stage switch,
            // matching plant allocation models more closely than a global timer gate.
            let surplus = (candidate.energy_current - candidate.energy_max * 0.35).max(0.0);
            let reproductive_effort = maturity.min(1.0) * reserve_ratio;
            let mut budget =
                surplus * (0.10 + candidate.genome.reserve_allocation * 0.18) * reproductive_effort;
            if budget <= candidate.energy_max * 0.018 {
                continue;
            }
            let parent_patch_crowding = self
                .grid
                .neighbors(candidate.x, candidate.y, 5.0)
                .into_iter()
                .filter(|&entity| self.world.get::<&genome::ProducerGenome>(entity).is_ok())
                .count()
                .saturating_sub(1) as f32;
            let dispersal_pressure = (1.0 + parent_patch_crowding * 0.08).clamp(1.0, 1.45);
            let clonal_pressure = 1.0 / (1.0 + parent_patch_crowding * 0.18);

            if candidate.state.seed_cooldown <= 0.0
                && self.rng.random_bool(
                    (0.20
                        + reproductive_effort * 0.30
                        + candidate.genome.seed_count.min(2.0) * 0.12
                        + parent_patch_crowding * 0.02)
                        .min(0.92) as f64,
                )
            {
                let num_seeds = ((candidate.genome.seed_count
                    * (0.7 + candidate.genome.reserve_allocation * 0.6))
                    .round() as usize)
                    .clamp(1, 3);
                let per_seed_cost =
                    (candidate.energy_max * 0.035 * candidate.genome.seed_size.max(0.5)).max(0.55);

                for _ in 0..num_seeds {
                    if active_producer_count + spawns.len() >= max_producers
                        || raw_producer_count + spawns.len() >= hard_max_producers
                        || budget < per_seed_cost
                    {
                        break;
                    }

                    let mut child_genome = candidate.genome.clone();
                    child_genome.generation += 1;
                    let mutation_rate = candidate.genome.mutation_rate_factor
                        * 0.12
                        * self
                            .calibration
                            .evolution
                            .producer_mutation_multiplier
                        * self.diversity_coefficient;
                    genetics::mutate_producer(&mut child_genome, mutation_rate, &mut self.rng);

                    // Research note: seed dispersal in submerged plants produces
                    // broader patch structure than clonal spread, so we use a fat-tailed
                    // kernel with occasional long-distance jumps (Li et al., 2015).
                    let base_seed_range =
                        (3.0 + candidate.genome.seed_range * 7.0) * dispersal_pressure;
                    let long_jump = self
                        .rng
                        .random_bool((0.08 + candidate.genome.seed_range * 0.08).min(0.24) as f64);
                    let distance = if long_jump {
                        self.rng
                            .random_range(base_seed_range..(base_seed_range * 2.4))
                    } else {
                        self.rng.random_range(0.0..base_seed_range)
                    };
                    let angle = self.rng.random_range(0.0..std::f32::consts::TAU);
                    let new_x = (candidate.x + angle.cos() * distance).clamp(2.0, tw - 2.0);
                    let new_y =
                        (candidate.y + angle.sin() * distance * 0.45).clamp(th * 0.22, th - 3.0);

                    let establishment =
                        self.producer_establishment_chance(new_x, new_y, ProducerOrigin::Broadcast, candidate.genome.hardiness);
                    if self.rng.random_bool(establishment as f64) {
                        spawns.push(SpawnRequest {
                            pos: Position { x: new_x, y: new_y },
                            genome: child_genome,
                            kind: ProducerOrigin::Broadcast,
                        });
                    }
                    budget -= per_seed_cost;
                    reserve_cost += per_seed_cost;
                }
                seed_cd = 7.5 - candidate.genome.seed_size.min(1.5) * 1.5;
            }

            if candidate.genome.clonal_spread > 0.25
                && candidate.state.clonal_cooldown <= 0.0
                && candidate.state.meristem_bank > 0.05
                && budget > candidate.energy_max * 0.02
                && self.rng.random_bool(
                    ((0.18 + reproductive_effort * 0.25 + candidate.genome.clonal_spread * 0.30)
                        * clonal_pressure)
                        .min(0.85) as f64,
                )
            {
                let clone_cost =
                    candidate.energy_max * (0.04 + candidate.genome.clonal_spread * 0.03);
                if budget >= clone_cost
                    && active_producer_count + spawns.len() < max_producers
                    && raw_producer_count + spawns.len() < hard_max_producers
                {
                    let mut child_genome = candidate.genome.clone();
                    child_genome.generation += 1;
                    let mutation_rate = candidate.genome.mutation_rate_factor
                        * 0.04
                        * self
                            .calibration
                            .evolution
                            .producer_mutation_multiplier
                        * self.diversity_coefficient;
                    genetics::mutate_producer(&mut child_genome, mutation_rate, &mut self.rng);

                    let spread = (1.5 + candidate.genome.clonal_spread * 4.0)
                        * self.substrate.clonal_bonus(candidate.x);
                    let new_x =
                        (candidate.x + self.rng.random_range(-spread..spread)).clamp(2.0, tw - 2.0);
                    let new_y =
                        (candidate.y + self.rng.random_range(-1.5..1.5)).clamp(th * 0.25, th - 3.0);

                    let establishment =
                        self.producer_establishment_chance(new_x, new_y, ProducerOrigin::Fragment, candidate.genome.hardiness);
                    if self.rng.random_bool(establishment as f64) {
                        spawns.push(SpawnRequest {
                            pos: Position { x: new_x, y: new_y },
                            genome: child_genome,
                            kind: ProducerOrigin::Fragment,
                        });
                    }
                    reserve_cost += clone_cost;
                    clonal_cd = 10.0 - candidate.genome.clonal_spread * 4.0;
                }
            }

            if reserve_cost > 0.0 {
                parent_updates.push(ParentUpdate {
                    entity: candidate.entity,
                    reserve_cost,
                    seed_cd,
                    clonal_cd,
                });
            }
        }

        for update in parent_updates {
            if let Ok(mut energy) = self.world.get::<&mut Energy>(update.entity) {
                energy.current = (energy.current - update.reserve_cost).max(0.0);
            }
            if let Ok(mut state) = self.world.get::<&mut ProducerState>(update.entity) {
                state.seed_cooldown = update.seed_cd.max(state.seed_cooldown);
                state.clonal_cooldown = update.clonal_cd.max(state.clonal_cooldown);
                state.meristem_bank = (state.meristem_bank - update.reserve_cost * 0.02).max(0.0);
            }
        }

        let mut producer_births = 0u64;
        for spawn in spawns {
            self.spawn_producer_with_propagule(spawn.pos, spawn.genome, Some(spawn.kind));
            raw_producer_count += 1;
            active_producer_count += 1;
            producer_births += 1;
            if active_producer_count >= max_producers || raw_producer_count >= hard_max_producers {
                break;
            }
        }
        producer_births
    }
}
