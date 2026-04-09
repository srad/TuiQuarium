use crate::behavior::BehaviorState;
use crate::boids::{Boid, BoidParams};
use crate::brain::{Brain, InnovationTracker};
use crate::calibration::RuntimeCalibration;
use crate::components::{
    AnimationState, Appearance, BoundingBox, ConsumerState, Position, ProducerAge, ProducerStage,
    ProducerState, RootedMacrophyte, Velocity,
};
use crate::ecosystem::{Age, Detritus, Energy, NutrientPool, Producer};
use crate::environment::{Environment, SubstrateGrid};
use crate::genome::{CreatureGenome, ProducerGenome};
use crate::needs::{NeedWeights, Needs};
use crate::phenotype::{DerivedPhysics, FeedingCapability};
use crate::pheromone::PheromoneGrid;
use crate::stats::DailyEcologySample;
use crate::AquariumSim;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    pub tank_width: u16,
    pub tank_height: u16,
    pub calibration: RuntimeCalibration,
    pub tick_count: u64,
    pub total_creature_births: u64,
    pub total_creature_deaths: u64,
    pub total_producer_births: u64,
    pub total_producer_deaths: u64,
    pub elapsed_days: u64,
    pub prev_time_of_day: f32,
    pub environment: Environment,
    pub nutrients: NutrientPool,
    pub substrate: SubstrateGrid,
    pub pheromone_grid: PheromoneGrid,
    pub innovation_tracker: InnovationTracker,
    pub rolling_producer_npp: f32,
    #[serde(default)]
    pub rolling_pelagic_consumer_intake: f32,
    pub rolling_consumer_intake: f32,
    pub rolling_consumer_maintenance: f32,
    pub pending_labile_detritus_energy: f32,
    pub feeding_frenzy_food_budget: f32,
    pub feeding_frenzy_detritus_budget: f32,
    pub daily_ecology_history: Vec<DailyEcologySample>,
    pub archived_daily_history: Vec<DailyEcologySample>,
    pub last_daily_creature_births: u64,
    pub last_daily_creature_deaths: u64,
    pub last_daily_producer_births: u64,
    pub last_daily_producer_deaths: u64,
    pub diversity_coefficient: f32,
    pub entities: Vec<EntitySnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntitySnapshot {
    Creature(CreatureEntitySnapshot),
    Producer(ProducerEntitySnapshot),
    Food(ResourceEntitySnapshot),
    Detritus(ResourceEntitySnapshot),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatureEntitySnapshot {
    pub position: Position,
    pub velocity: Velocity,
    pub bounding_box: BoundingBox,
    pub appearance: Appearance,
    pub animation: AnimationState,
    pub genome: CreatureGenome,
    pub physics: DerivedPhysics,
    pub feeding: FeedingCapability,
    pub energy: Energy,
    pub age: Age,
    pub needs: Needs,
    pub need_weights: NeedWeights,
    pub behavior: BehaviorState,
    pub consumer_state: ConsumerState,
    pub brain: Brain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerEntitySnapshot {
    pub position: Position,
    pub velocity: Velocity,
    pub bounding_box: BoundingBox,
    pub appearance: Appearance,
    pub animation: AnimationState,
    pub energy: Energy,
    pub physics: DerivedPhysics,
    pub feeding: FeedingCapability,
    pub genome: ProducerGenome,
    pub producer_state: ProducerState,
    pub stage: ProducerStage,
    pub producer_age: ProducerAge,
    pub age: Age,
    pub rooted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEntitySnapshot {
    pub position: Position,
    pub velocity: Velocity,
    pub bounding_box: BoundingBox,
    pub appearance: Appearance,
    pub animation: AnimationState,
    pub energy: Energy,
    pub physics: DerivedPhysics,
    pub feeding: FeedingCapability,
}

impl AquariumSim {
    pub fn snapshot(&self) -> SimulationSnapshot {
        let entities = self.snapshot_entities();
        SimulationSnapshot {
            tank_width: self.tank_width,
            tank_height: self.tank_height,
            calibration: self.calibration,
            tick_count: self.tick_count,
            total_creature_births: self.total_creature_births,
            total_creature_deaths: self.total_creature_deaths,
            total_producer_births: self.total_producer_births,
            total_producer_deaths: self.total_producer_deaths,
            elapsed_days: self.elapsed_days,
            prev_time_of_day: self.prev_time_of_day,
            environment: self.env.clone(),
            nutrients: self.nutrients.clone(),
            substrate: self.substrate.clone(),
            pheromone_grid: self.pheromone_grid.clone(),
            innovation_tracker: self.innovation_tracker.clone(),
            rolling_producer_npp: self.rolling_producer_npp,
            rolling_pelagic_consumer_intake: self.rolling_pelagic_consumer_intake,
            rolling_consumer_intake: self.rolling_consumer_intake,
            rolling_consumer_maintenance: self.rolling_consumer_maintenance,
            pending_labile_detritus_energy: self.pending_labile_detritus_energy,
            feeding_frenzy_food_budget: self.feeding_frenzy_food_budget,
            feeding_frenzy_detritus_budget: self.feeding_frenzy_detritus_budget,
            daily_ecology_history: self.daily_ecology_history.iter().cloned().collect(),
            archived_daily_history: self.archived_daily_history.clone(),
            last_daily_creature_births: self.last_daily_creature_births,
            last_daily_creature_deaths: self.last_daily_creature_deaths,
            last_daily_producer_births: self.last_daily_producer_births,
            last_daily_producer_deaths: self.last_daily_producer_deaths,
            diversity_coefficient: self.diversity_coefficient,
            entities,
        }
    }

    pub fn from_snapshot(snapshot: &SimulationSnapshot) -> Result<Self, String> {
        let mut sim = Self {
            world: hecs::World::new(),
            env: snapshot.environment.clone(),
            calibration: snapshot.calibration,
            tick_count: snapshot.tick_count,
            total_creature_births: snapshot.total_creature_births,
            total_creature_deaths: snapshot.total_creature_deaths,
            total_producer_births: snapshot.total_producer_births,
            total_producer_deaths: snapshot.total_producer_deaths,
            elapsed_days: snapshot.elapsed_days,
            prev_time_of_day: snapshot.prev_time_of_day,
            tank_width: snapshot.tank_width,
            tank_height: snapshot.tank_height,
            grid: crate::spatial::SpatialGrid::new(6.0),
            boid_params: BoidParams::default(),
            rng: rand::SeedableRng::seed_from_u64(rand::random()),
            innovation_tracker: snapshot.innovation_tracker.clone(),
            pheromone_grid: snapshot.pheromone_grid.clone(),
            light_field: crate::ecosystem::LightField::new(
                snapshot.tank_width,
                snapshot.tank_height,
            ),
            nutrients: snapshot.nutrients.clone(),
            substrate: snapshot.substrate.clone(),
            pending_births: Vec::new(),
            brain_system: crate::systems::NeatBrainSystem,
            ecosystem_system: crate::systems::AllometricEcosystem,
            hunting_system: crate::systems::BodySizeHunting,
            reproduction_system: crate::systems::DefaultReproduction,
            producer_lifecycle_system: crate::systems::DefaultProducerLifecycle,
            cached_max_generation: 0,
            cached_avg_complexity: 0.0,
            cached_max_creature_complexity: 0.0,
            cached_species_count: 0,
            cached_creature_count: 0,
            cached_producer_leaf_biomass: 0.0,
            cached_producer_structural_biomass: 0.0,
            cached_producer_belowground_reserve: 0.0,
            cached_consumer_biomass: 0.0,
            cached_juvenile_count: 0,
            cached_adult_count: 0,
            rolling_producer_npp: snapshot.rolling_producer_npp,
            rolling_pelagic_consumer_intake: snapshot.rolling_pelagic_consumer_intake,
            rolling_consumer_intake: snapshot.rolling_consumer_intake,
            rolling_consumer_maintenance: snapshot.rolling_consumer_maintenance,
            pending_labile_detritus_energy: snapshot.pending_labile_detritus_energy,
            feeding_frenzy_food_budget: snapshot.feeding_frenzy_food_budget,
            feeding_frenzy_detritus_budget: snapshot.feeding_frenzy_detritus_budget,
            stats_cache_tick: snapshot.tick_count,
            daily_ecology_history: VecDeque::from(snapshot.daily_ecology_history.clone()),
            archived_daily_history: snapshot.archived_daily_history.clone(),
            last_daily_creature_births: snapshot.last_daily_creature_births,
            last_daily_creature_deaths: snapshot.last_daily_creature_deaths,
            last_daily_producer_births: snapshot.last_daily_producer_births,
            last_daily_producer_deaths: snapshot.last_daily_producer_deaths,
            diversity_coefficient: snapshot.diversity_coefficient,
        };

        for entity in &snapshot.entities {
            sim.restore_entity(entity);
        }

        sim.grid.rebuild(&sim.world);
        sim.light_field.rebuild(&sim.world);
        sim.recompute_cached_stats();
        sim.stats_cache_tick = sim.tick_count;
        Ok(sim)
    }

    fn snapshot_entities(&self) -> Vec<EntitySnapshot> {
        let mut entities = Vec::with_capacity(self.world.len() as usize);
        for (entity, pos, vel, bbox, appearance, anim, energy, physics, feeding) in
            &mut self.world.query::<(
                hecs::Entity,
                &Position,
                &Velocity,
                &BoundingBox,
                &Appearance,
                &AnimationState,
                &Energy,
                &DerivedPhysics,
                &FeedingCapability,
            )>()
        {
            if let Ok(genome) = self.world.get::<&CreatureGenome>(entity) {
                let needs = (*self.world.get::<&Needs>(entity).unwrap()).clone();
                let need_weights = (*self.world.get::<&NeedWeights>(entity).unwrap()).clone();
                let behavior = (*self.world.get::<&BehaviorState>(entity).unwrap()).clone();
                let consumer_state = (*self.world.get::<&ConsumerState>(entity).unwrap()).clone();
                let brain = (*self.world.get::<&Brain>(entity).unwrap()).clone();
                let age = (*self.world.get::<&Age>(entity).unwrap()).clone();
                entities.push(EntitySnapshot::Creature(CreatureEntitySnapshot {
                    position: pos.clone(),
                    velocity: vel.clone(),
                    bounding_box: bbox.clone(),
                    appearance: appearance.clone(),
                    animation: anim.clone(),
                    genome: (*genome).clone(),
                    physics: physics.clone(),
                    feeding: feeding.clone(),
                    energy: energy.clone(),
                    age,
                    needs,
                    need_weights,
                    behavior,
                    consumer_state,
                    brain,
                }));
                continue;
            }

            if let Ok(genome) = self.world.get::<&ProducerGenome>(entity) {
                let producer_state = (*self.world.get::<&ProducerState>(entity).unwrap()).clone();
                let stage = *self.world.get::<&ProducerStage>(entity).unwrap();
                let producer_age = (*self.world.get::<&ProducerAge>(entity).unwrap()).clone();
                let age = (*self.world.get::<&Age>(entity).unwrap()).clone();
                let rooted = self.world.get::<&RootedMacrophyte>(entity).is_ok();
                entities.push(EntitySnapshot::Producer(ProducerEntitySnapshot {
                    position: pos.clone(),
                    velocity: vel.clone(),
                    bounding_box: bbox.clone(),
                    appearance: appearance.clone(),
                    animation: anim.clone(),
                    energy: energy.clone(),
                    physics: physics.clone(),
                    feeding: feeding.clone(),
                    genome: (*genome).clone(),
                    producer_state,
                    stage,
                    producer_age,
                    age,
                    rooted,
                }));
                continue;
            }

            let resource = ResourceEntitySnapshot {
                position: pos.clone(),
                velocity: vel.clone(),
                bounding_box: bbox.clone(),
                appearance: appearance.clone(),
                animation: anim.clone(),
                energy: energy.clone(),
                physics: physics.clone(),
                feeding: feeding.clone(),
            };
            if self.world.get::<&Detritus>(entity).is_ok() {
                entities.push(EntitySnapshot::Detritus(resource));
            } else if self.world.get::<&Producer>(entity).is_ok() {
                entities.push(EntitySnapshot::Food(resource));
            }
        }
        entities
    }

    fn restore_entity(&mut self, entity: &EntitySnapshot) {
        match entity {
            EntitySnapshot::Creature(creature) => {
                let entity = self.world.spawn((
                    creature.position.clone(),
                    creature.velocity.clone(),
                    creature.bounding_box.clone(),
                    creature.appearance.clone(),
                    creature.animation.clone(),
                    creature.genome.clone(),
                    creature.physics.clone(),
                    creature.feeding.clone(),
                    creature.energy.clone(),
                    creature.age.clone(),
                    creature.needs.clone(),
                    creature.need_weights.clone(),
                    creature.behavior.clone(),
                    Boid,
                    creature.brain.clone(),
                ));
                let _ = self
                    .world
                    .insert(entity, (creature.consumer_state.clone(),));
            }
            EntitySnapshot::Producer(producer) => {
                let entity = self.world.spawn((
                    producer.position.clone(),
                    producer.velocity.clone(),
                    producer.bounding_box.clone(),
                    producer.appearance.clone(),
                    producer.animation.clone(),
                    Producer,
                    producer.energy.clone(),
                    producer.physics.clone(),
                    producer.feeding.clone(),
                    producer.genome.clone(),
                    producer.producer_state.clone(),
                    producer.stage,
                    producer.producer_age.clone(),
                    producer.age.clone(),
                ));
                if producer.rooted {
                    let _ = self.world.insert(entity, (RootedMacrophyte,));
                }
            }
            EntitySnapshot::Food(resource) => {
                self.world.spawn((
                    resource.position.clone(),
                    resource.velocity.clone(),
                    resource.bounding_box.clone(),
                    resource.appearance.clone(),
                    resource.animation.clone(),
                    Producer,
                    resource.energy.clone(),
                    resource.physics.clone(),
                    resource.feeding.clone(),
                ));
            }
            EntitySnapshot::Detritus(resource) => {
                self.world.spawn((
                    resource.position.clone(),
                    resource.velocity.clone(),
                    resource.bounding_box.clone(),
                    resource.appearance.clone(),
                    resource.animation.clone(),
                    Producer,
                    Detritus,
                    resource.energy.clone(),
                    resource.physics.clone(),
                    resource.feeding.clone(),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Simulation;

    #[test]
    fn simulation_snapshot_roundtrip_restores_stats_and_history() {
        let mut sim = AquariumSim::new_seeded(48, 16, 42);
        sim.bootstrap_founder_web();
        for _ in 0..400 {
            sim.tick(0.5);
        }
        let snapshot = sim.snapshot();
        let restored = AquariumSim::from_snapshot(&snapshot).unwrap();

        assert_eq!(restored.tank_size(), sim.tank_size());
        assert_eq!(restored.stats().tick_count, sim.stats().tick_count);
        assert_eq!(
            restored.stats().creature_births,
            sim.stats().creature_births
        );
        assert_eq!(
            restored.stats().producer_births,
            sim.stats().producer_births
        );
        assert_eq!(
            restored.archived_daily_history(),
            sim.archived_daily_history()
        );
        assert_eq!(
            restored.ecology_diagnostics().daily_history,
            sim.ecology_diagnostics().daily_history
        );
        assert_eq!(restored.world().len(), sim.world().len());
    }
}
