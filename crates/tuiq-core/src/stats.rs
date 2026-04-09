//! Simulation statistics and ecology diagnostics.

use crate::components::{ConsumerState, ProducerState, RootedMacrophyte};
use crate::ecosystem::{
    consumer_reproduction_reserve_threshold, consumer_reproductive_threshold, Detritus, Energy,
};
use crate::genetics::{genomic_distance, CREATURE_SPECIES_THRESHOLD};
use crate::genome::CreatureGenome;
use crate::needs::Needs;
use crate::phenotype::DerivedPhysics;
use serde::{Deserialize, Serialize};
/// Statistics about the current simulation state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SimStats {
    pub entity_count: usize,
    pub creature_count: usize,
    pub tick_count: u64,
    pub creature_births: u64,
    pub creature_deaths: u64,
    pub producer_births: u64,
    pub producer_deaths: u64,
    /// Elapsed in-game days (derived from environment time progression).
    pub elapsed_days: u64,
    /// Maximum generation number among living creatures.
    pub max_generation: u32,
    /// Average complexity of living creatures.
    pub avg_complexity: f32,
    /// Maximum complexity among living creatures.
    pub max_creature_complexity: f32,
    /// Estimated number of species (clusters by genomic distance).
    pub species_count: usize,
    pub producer_leaf_biomass: f32,
    pub producer_structural_biomass: f32,
    pub producer_belowground_reserve: f32,
    pub consumer_biomass: f32,
    pub rolling_producer_npp: f32,
    pub rolling_consumer_intake: f32,
    pub rolling_consumer_maintenance: f32,
    pub juvenile_count: usize,
    pub adult_count: usize,
}

pub(crate) const ECOLOGY_HISTORY_DAYS: usize = 32;
const SUBADULT_MATURITY_THRESHOLD: f32 = 0.85;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EcologyInstant {
    pub producer_total_biomass: f32,
    pub producer_active_biomass: f32,
    pub consumer_biomass: f32,
    pub rolling_producer_npp: f32,
    pub rolling_consumer_intake: f32,
    pub rolling_consumer_maintenance: f32,
    pub dissolved_n: f32,
    pub dissolved_p: f32,
    pub sediment_n: f32,
    pub sediment_p: f32,
    pub phytoplankton_load: f32,
    pub creature_births: u64,
    pub creature_deaths: u64,
    pub producer_births: u64,
    pub producer_deaths: u64,
    pub consumer_to_producer_biomass_ratio: f32,
    pub intake_to_npp_ratio: f32,
    pub maintenance_to_intake_ratio: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DailyEcologySample {
    pub day: u64,
    pub producer_total_biomass: f32,
    pub consumer_biomass: f32,
    #[serde(default)]
    pub creature_count: usize,
    #[serde(default)]
    pub juvenile_count: usize,
    #[serde(default)]
    pub subadult_count: usize,
    #[serde(default)]
    pub species_count: usize,
    #[serde(default)]
    pub rooted_producer_count: usize,
    #[serde(default)]
    pub rooted_producer_biomass: f32,
    #[serde(default)]
    pub detritus_energy: f32,
    #[serde(default)]
    pub mean_maturity_progress: f32,
    #[serde(default)]
    pub mean_reserve_buffer: f32,
    #[serde(default)]
    pub mean_reproductive_buffer: f32,
    #[serde(default)]
    pub max_reproductive_buffer: f32,
    #[serde(default)]
    pub mean_recent_assimilation: f32,
    #[serde(default)]
    pub mean_brood_cooldown: f32,
    #[serde(default)]
    pub reproduction_ready_count: usize,
    #[serde(default)]
    pub rolling_pelagic_consumer_intake: f32,
    pub rolling_producer_npp: f32,
    pub rolling_consumer_intake: f32,
    pub rolling_consumer_maintenance: f32,
    pub dissolved_n: f32,
    pub dissolved_p: f32,
    pub phytoplankton_load: f32,
    pub creature_births_delta: u64,
    pub creature_deaths_delta: u64,
    pub producer_births_delta: u64,
    pub producer_deaths_delta: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EcologyDiagnostics {
    pub instant: EcologyInstant,
    pub daily_history: Vec<DailyEcologySample>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ConsumerLifeHistorySummary {
    mean_maturity_progress: f32,
    mean_reserve_buffer: f32,
    mean_reproductive_buffer: f32,
    max_reproductive_buffer: f32,
    mean_recent_assimilation: f32,
    mean_brood_cooldown: f32,
    reproduction_ready_count: usize,
}

impl super::AquariumSim {
    /// Recompute cached stats (creature count, generation, complexity, species).
    /// Called periodically instead of every frame.
    pub(crate) fn recompute_cached_stats(&mut self) {
        let mut creature_count = 0;
        let mut max_generation: u32 = 0;
        let mut total_complexity: f32 = 0.0;
        let mut max_creature_complexity: f32 = 0.0;
        let mut consumer_biomass = 0.0;
        let mut juvenile_count = 0usize;
        let mut adult_count = 0usize;
        let mut producer_leaf_biomass = 0.0;
        let mut producer_structural_biomass = 0.0;
        let mut producer_belowground_reserve = 0.0;

        for genome in &mut self.world.query::<&CreatureGenome>() {
            creature_count += 1;
            max_generation = max_generation.max(genome.generation);
            total_complexity += genome.complexity;
            max_creature_complexity = max_creature_complexity.max(genome.complexity);
        }

        for (physics, state) in &mut self.world.query::<(&DerivedPhysics, &ConsumerState)>() {
            consumer_biomass += physics.body_mass;
            if state.is_adult() {
                adult_count += 1;
            } else {
                juvenile_count += 1;
            }
        }

        for state in &mut self.world.query::<&ProducerState>() {
            producer_leaf_biomass += state.leaf_biomass;
            producer_structural_biomass += state.structural_biomass;
            producer_belowground_reserve += state.belowground_reserve;
        }

        self.cached_creature_count = creature_count;
        self.cached_max_generation = max_generation;
        self.cached_avg_complexity = if creature_count > 0 {
            total_complexity / creature_count as f32
        } else {
            0.0
        };
        self.cached_max_creature_complexity = max_creature_complexity;
        self.cached_consumer_biomass = consumer_biomass;
        self.cached_juvenile_count = juvenile_count;
        self.cached_adult_count = adult_count;
        self.cached_producer_leaf_biomass = producer_leaf_biomass;
        self.cached_producer_structural_biomass = producer_structural_biomass;
        self.cached_producer_belowground_reserve = producer_belowground_reserve;

        // Estimate species without cloning — sample up to 100 genomes as centroids
        let mut centroid_indices: Vec<usize> = Vec::new();
        let mut sampled: Vec<CreatureGenome> = Vec::new();

        // Collect a sample (all if <200, else first 200)
        let mut count = 0;
        for genome in &mut self.world.query::<&CreatureGenome>() {
            if count >= 200 {
                break;
            }
            sampled.push((*genome).clone());
            count += 1;
        }

        for (i, g) in sampled.iter().enumerate() {
            let fits = centroid_indices
                .iter()
                .any(|&ci| genomic_distance(&sampled[ci], g) < CREATURE_SPECIES_THRESHOLD);
            if !fits {
                centroid_indices.push(i);
            }
        }

        self.cached_species_count = centroid_indices.len();
        self.stats_cache_tick = self.tick_count;
    }

    pub(crate) fn producer_total_biomass(&self) -> f32 {
        self.cached_producer_leaf_biomass
            + self.cached_producer_structural_biomass
            + self.cached_producer_belowground_reserve
    }

    pub(crate) fn ratio_or_zero(numerator: f32, denominator: f32) -> f32 {
        if denominator.abs() <= 1e-6 {
            0.0
        } else {
            numerator / denominator
        }
    }

    pub(crate) fn build_ecology_instant(&self) -> EcologyInstant {
        let producer_total_biomass = self.producer_total_biomass();
        let producer_active_biomass = self.cached_producer_leaf_biomass;
        let consumer_biomass = self.cached_consumer_biomass;

        EcologyInstant {
            producer_total_biomass,
            producer_active_biomass,
            consumer_biomass,
            rolling_producer_npp: self.rolling_producer_npp,
            rolling_consumer_intake: self.rolling_consumer_intake,
            rolling_consumer_maintenance: self.rolling_consumer_maintenance,
            dissolved_n: self.nutrients.dissolved_n,
            dissolved_p: self.nutrients.dissolved_p,
            sediment_n: self.nutrients.sediment_n,
            sediment_p: self.nutrients.sediment_p,
            phytoplankton_load: self.nutrients.phytoplankton_load,
            creature_births: self.total_creature_births,
            creature_deaths: self.total_creature_deaths,
            producer_births: self.total_producer_births,
            producer_deaths: self.total_producer_deaths,
            consumer_to_producer_biomass_ratio: Self::ratio_or_zero(
                consumer_biomass,
                producer_total_biomass,
            ),
            intake_to_npp_ratio: Self::ratio_or_zero(
                self.rolling_consumer_intake,
                self.rolling_producer_npp,
            ),
            maintenance_to_intake_ratio: Self::ratio_or_zero(
                self.rolling_consumer_maintenance,
                self.rolling_consumer_intake,
            ),
        }
    }

    pub(crate) fn record_daily_diagnostics(&mut self) {
        let instant = self.build_ecology_instant();
        let life_history = self.sample_consumer_life_history();
        let (juvenile_count, subadult_count) = self.sample_consumer_stage_counts();
        let (rooted_producer_count, rooted_producer_biomass) = self.sample_rooted_producer_summary();
        let mut detritus_energy = self.pending_labile_detritus_energy.max(0.0);
        for (_detritus, energy) in &mut self.world.query::<(&Detritus, &Energy)>() {
            detritus_energy += energy.current.max(0.0);
        }
        let sample = DailyEcologySample {
            day: self.elapsed_days,
            producer_total_biomass: instant.producer_total_biomass,
            consumer_biomass: instant.consumer_biomass,
            creature_count: self.cached_creature_count,
            juvenile_count,
            subadult_count,
            species_count: self.cached_species_count,
            rooted_producer_count,
            rooted_producer_biomass,
            detritus_energy,
            mean_maturity_progress: life_history.mean_maturity_progress,
            mean_reserve_buffer: life_history.mean_reserve_buffer,
            mean_reproductive_buffer: life_history.mean_reproductive_buffer,
            max_reproductive_buffer: life_history.max_reproductive_buffer,
            mean_recent_assimilation: life_history.mean_recent_assimilation,
            mean_brood_cooldown: life_history.mean_brood_cooldown,
            reproduction_ready_count: life_history.reproduction_ready_count,
            rolling_pelagic_consumer_intake: self.rolling_pelagic_consumer_intake,
            rolling_producer_npp: instant.rolling_producer_npp,
            rolling_consumer_intake: instant.rolling_consumer_intake,
            rolling_consumer_maintenance: instant.rolling_consumer_maintenance,
            dissolved_n: instant.dissolved_n,
            dissolved_p: instant.dissolved_p,
            phytoplankton_load: instant.phytoplankton_load,
            creature_births_delta: self
                .total_creature_births
                .saturating_sub(self.last_daily_creature_births),
            creature_deaths_delta: self
                .total_creature_deaths
                .saturating_sub(self.last_daily_creature_deaths),
            producer_births_delta: self
                .total_producer_births
                .saturating_sub(self.last_daily_producer_births),
            producer_deaths_delta: self
                .total_producer_deaths
                .saturating_sub(self.last_daily_producer_deaths),
        };

        self.last_daily_creature_births = self.total_creature_births;
        self.last_daily_creature_deaths = self.total_creature_deaths;
        self.last_daily_producer_births = self.total_producer_births;
        self.last_daily_producer_deaths = self.total_producer_deaths;

        if self.daily_ecology_history.len() == ECOLOGY_HISTORY_DAYS {
            self.daily_ecology_history.pop_front();
        }
        self.daily_ecology_history.push_back(sample.clone());
        self.archived_daily_history.push(sample);
    }

    pub(crate) fn reset_runtime_counters(&mut self) {
        self.tick_count = 0;
        self.total_creature_births = 0;
        self.total_creature_deaths = 0;
        self.total_producer_births = 0;
        self.total_producer_deaths = 0;
        self.elapsed_days = 0;
        self.prev_time_of_day = self.env.time_of_day;
        self.pending_births.clear();
        self.rolling_producer_npp = 0.0;
        self.rolling_pelagic_consumer_intake = 0.0;
        self.rolling_consumer_intake = 0.0;
        self.rolling_consumer_maintenance = 0.0;
        self.pending_labile_detritus_energy = 0.0;
        self.daily_ecology_history.clear();
        self.archived_daily_history.clear();
        self.last_daily_creature_births = 0;
        self.last_daily_creature_deaths = 0;
        self.last_daily_producer_births = 0;
        self.last_daily_producer_deaths = 0;
        self.recompute_cached_stats();
        self.stats_cache_tick = self.tick_count;
    }

    pub fn archived_daily_history(&self) -> &[DailyEcologySample] {
        &self.archived_daily_history
    }

    fn sample_consumer_stage_counts(&mut self) -> (usize, usize) {
        let mut juvenile_count = 0usize;
        let mut subadult_count = 0usize;

        for state in &mut self.world.query::<&ConsumerState>() {
            if state.is_adult() {
                continue;
            }
            if state.maturity_progress >= SUBADULT_MATURITY_THRESHOLD {
                subadult_count += 1;
            } else {
                juvenile_count += 1;
            }
        }

        (juvenile_count, subadult_count)
    }

    fn sample_rooted_producer_summary(&mut self) -> (usize, f32) {
        let mut rooted_producer_count = 0usize;
        let mut rooted_producer_biomass = 0.0;

        for state in &mut self.world.query::<(&ProducerState, &RootedMacrophyte)>() {
            rooted_producer_count += 1;
            rooted_producer_biomass += state.0.total_biomass();
        }

        (rooted_producer_count, rooted_producer_biomass)
    }

    fn sample_consumer_life_history(&mut self) -> ConsumerLifeHistorySummary {
        let mut count = 0usize;
        let mut maturity_total = 0.0;
        let mut reserve_total = 0.0;
        let mut reproductive_total = 0.0;
        let mut max_reproductive_buffer: f32 = 0.0;
        let mut recent_assimilation_total = 0.0;
        let mut brood_cooldown_total = 0.0;
        let mut reproduction_ready_count = 0usize;

        for (energy, needs, genome, physics, state) in &mut self.world.query::<(
            &Energy,
            &Needs,
            &CreatureGenome,
            &DerivedPhysics,
            &ConsumerState,
        )>() {
            count += 1;
            maturity_total += state.maturity_progress;
            reserve_total += state.reserve_buffer;
            reproductive_total += state.reproductive_buffer;
            max_reproductive_buffer = max_reproductive_buffer.max(state.reproductive_buffer);
            recent_assimilation_total += state.recent_assimilation;
            brood_cooldown_total += state.brood_cooldown;

            let reserve_threshold = consumer_reproduction_reserve_threshold(physics, genome);
            let reproductive_threshold = consumer_reproductive_threshold(physics, genome);
            if state.is_adult()
                && state.brood_cooldown <= 0.0
                && state.reproductive_buffer >= reproductive_threshold
                && needs.reproduction > 0.22
                && energy.fraction() > reserve_threshold
            {
                reproduction_ready_count += 1;
            }
        }

        if count == 0 {
            return ConsumerLifeHistorySummary::default();
        }

        let denom = count as f32;
        ConsumerLifeHistorySummary {
            mean_maturity_progress: maturity_total / denom,
            mean_reserve_buffer: reserve_total / denom,
            mean_reproductive_buffer: reproductive_total / denom,
            max_reproductive_buffer,
            mean_recent_assimilation: recent_assimilation_total / denom,
            mean_brood_cooldown: brood_cooldown_total / denom,
            reproduction_ready_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    #[test]
    fn test_daily_diagnostics_capture_consumer_life_history_state() {
        let mut sim = crate::AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();
        sim.record_daily_diagnostics();

        let sample = sim
            .archived_daily_history()
            .last()
            .expect("daily diagnostics should append an archived sample");

        assert!(sample.creature_count > 0);
        assert!(sample.mean_maturity_progress > 0.0);
        assert!(sample.mean_reserve_buffer > 0.0);
        assert!(sample.mean_reproductive_buffer >= 0.0);
        assert!(sample.max_reproductive_buffer >= sample.mean_reproductive_buffer);
        assert!(sample.mean_recent_assimilation >= 0.0);
        assert!(sample.mean_brood_cooldown >= 0.0);
        assert!(sample.reproduction_ready_count <= sample.creature_count);
        assert!(sample.subadult_count <= sample.creature_count);
    }

    #[test]
    fn test_daily_diagnostics_distinguish_juveniles_from_subadults() {
        let mut sim = crate::AquariumSim::new(32, 12);
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);

        for (maturity_progress, matured_once) in [(0.40, false), (0.92, false), (1.0, true)] {
            let genome = crate::genome::CreatureGenome::minimal_cell(&mut rng);
            let physics = crate::phenotype::derive_physics(&genome);
            sim.world.spawn((
                crate::ecosystem::Energy::new(physics.max_energy),
                crate::needs::Needs::default(),
                genome,
                physics,
                crate::components::ConsumerState {
                    reserve_buffer: 0.35,
                    maturity_progress,
                    matured_once,
                    reproductive_buffer: 0.0,
                    brood_cooldown: 0.0,
                    recent_assimilation: 0.0,
                },
            ));
        }

        sim.recompute_cached_stats();
        sim.record_daily_diagnostics();

        let sample = sim
            .archived_daily_history()
            .last()
            .expect("daily diagnostics should append an archived sample");

        assert_eq!(sample.creature_count, 3);
        assert_eq!(sample.juvenile_count, 1);
        assert_eq!(sample.subadult_count, 1);
    }
}
