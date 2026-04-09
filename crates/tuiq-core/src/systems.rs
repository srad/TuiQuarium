//! Trait abstractions for the major ECS systems.
//!
//! Each trait defines the interface contract for a simulation subsystem.
//! Default implementations wrap the existing free functions in their
//! respective modules, providing zero-cost abstraction via monomorphization.

use crate::brain;
use crate::calibration::{EcologyCalibration, EvolutionCalibration};
use crate::ecosystem::{
    self, DeathResult, FeedingSummary, LightField, MetabolismFlux, NutrientPool,
    ProducerEcologyFlux,
};
use crate::environment::Environment;
use crate::pheromone::PheromoneGrid;
use crate::spatial::SpatialGrid;
use crate::spawner;
use crate::{producer_lifecycle, EntityInfoMap};

use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Brain
// ---------------------------------------------------------------------------

/// Sensory processing and motor decision for all creatures.
pub trait BrainSystem {
    /// Evaluate all creature brains. Returns pheromone deposits (x, y, amount).
    fn evaluate(
        &self,
        world: &mut hecs::World,
        grid: &SpatialGrid,
        entity_map: &EntityInfoMap,
        env: &Environment,
        pheromone_grid: &PheromoneGrid,
        dt: f32,
        tank_w: f32,
        tank_h: f32,
    ) -> Vec<(f32, f32, f32)>;
}

/// NEAT-based brain using feedforward networks with Hebbian learning.
pub struct NeatBrainSystem;

impl BrainSystem for NeatBrainSystem {
    fn evaluate(
        &self,
        world: &mut hecs::World,
        grid: &SpatialGrid,
        entity_map: &EntityInfoMap,
        env: &Environment,
        pheromone_grid: &PheromoneGrid,
        dt: f32,
        tank_w: f32,
        tank_h: f32,
    ) -> Vec<(f32, f32, f32)> {
        brain::brain_system(
            world,
            grid,
            entity_map,
            env,
            pheromone_grid,
            dt,
            tank_w,
            tank_h,
        )
    }
}

// ---------------------------------------------------------------------------
// Ecosystem
// ---------------------------------------------------------------------------

/// Metabolic, growth, aging, and death processes.
pub trait EcosystemSystem {
    /// Consumer + producer metabolism; returns nutrient flux for recycling.
    fn tick_metabolism(
        &self,
        world: &mut hecs::World,
        dt: f32,
        env: &Environment,
        tank_height: f32,
        calibration: &EcologyCalibration,
    ) -> MetabolismFlux;

    /// Producer photosynthesis, nutrient uptake, stress, and turnover.
    fn tick_producer_ecology(
        &self,
        world: &mut hecs::World,
        dt: f32,
        env: &Environment,
        tank_height: f32,
        light_field: &LightField,
        nutrients: &mut NutrientPool,
        calibration: &EcologyCalibration,
    ) -> ProducerEcologyFlux;

    /// Increment entity ages.
    fn tick_aging(&self, world: &mut hecs::World);

    /// Update maturation, reproductive buffer, and consumer needs.
    fn tick_consumer_lifecycle(
        &self,
        world: &mut hecs::World,
        dt: f32,
        tank_width: f32,
        tank_height: f32,
    );

    /// Remove dead entities and recycle nutrients.
    fn tick_death(&self, world: &mut hecs::World) -> DeathResult;
}

/// Allometric ecosystem model using Kleiber's law and Beer-Lambert light.
pub struct AllometricEcosystem;

impl EcosystemSystem for AllometricEcosystem {
    fn tick_metabolism(
        &self,
        world: &mut hecs::World,
        dt: f32,
        env: &Environment,
        tank_height: f32,
        calibration: &EcologyCalibration,
    ) -> MetabolismFlux {
        ecosystem::metabolism_system(world, dt, env, tank_height, calibration)
    }

    fn tick_producer_ecology(
        &self,
        world: &mut hecs::World,
        dt: f32,
        env: &Environment,
        tank_height: f32,
        light_field: &LightField,
        nutrients: &mut NutrientPool,
        calibration: &EcologyCalibration,
    ) -> ProducerEcologyFlux {
        ecosystem::producer_ecology_system(
            world,
            dt,
            env,
            tank_height,
            light_field,
            nutrients,
            calibration,
        )
    }

    fn tick_aging(&self, world: &mut hecs::World) {
        ecosystem::age_system(world);
    }

    fn tick_consumer_lifecycle(
        &self,
        world: &mut hecs::World,
        dt: f32,
        tank_width: f32,
        tank_height: f32,
    ) {
        ecosystem::consumer_life_history_system(world, dt, tank_width, tank_height);
    }

    fn tick_death(&self, world: &mut hecs::World) -> DeathResult {
        ecosystem::death_system(world)
    }
}

// ---------------------------------------------------------------------------
// Hunting
// ---------------------------------------------------------------------------

/// Predation: prey detection and kill application.
pub trait HuntingSystem {
    /// Find predator-prey pairs and apply kills. Returns feeding summary.
    fn find_and_apply(
        &self,
        world: &mut hecs::World,
        grid: &SpatialGrid,
        entity_map: &EntityInfoMap,
        dt: f32,
    ) -> FeedingSummary;
}

/// Body-size-ratio hunting with emergent feeding capabilities.
pub struct BodySizeHunting;

impl HuntingSystem for BodySizeHunting {
    fn find_and_apply(
        &self,
        world: &mut hecs::World,
        grid: &SpatialGrid,
        entity_map: &EntityInfoMap,
        dt: f32,
    ) -> FeedingSummary {
        let kills = ecosystem::hunting_check(world, grid, entity_map);
        ecosystem::apply_kills(world, &kills, dt)
    }
}

// ---------------------------------------------------------------------------
// Creature Reproduction
// ---------------------------------------------------------------------------

/// Creature sexual/asexual reproduction with speciation.
pub trait ReproductionSystem {
    /// Process creature reproduction. Returns newly spawned entities.
    fn reproduce_creatures(
        &self,
        world: &mut hecs::World,
        grid: &SpatialGrid,
        rng: &mut StdRng,
        tank_w: f32,
        tank_h: f32,
        creature_count: usize,
        tracker: &mut brain::InnovationTracker,
        evolution: &EvolutionCalibration,
        diversity_coefficient: f32,
    ) -> Vec<hecs::Entity>;
}

/// Default reproduction: fitness sharing, asexual fission, sexual crossover.
pub struct DefaultReproduction;

impl ReproductionSystem for DefaultReproduction {
    fn reproduce_creatures(
        &self,
        world: &mut hecs::World,
        grid: &SpatialGrid,
        rng: &mut StdRng,
        tank_w: f32,
        tank_h: f32,
        creature_count: usize,
        tracker: &mut brain::InnovationTracker,
        evolution: &EvolutionCalibration,
        diversity_coefficient: f32,
    ) -> Vec<hecs::Entity> {
        spawner::reproduction_system(
            world,
            grid,
            rng,
            tank_w,
            tank_h,
            creature_count,
            tracker,
            evolution,
            diversity_coefficient,
        )
    }
}

// ---------------------------------------------------------------------------
// Producer Lifecycle
// ---------------------------------------------------------------------------

/// Plant growth stages and appearance generation.
pub trait ProducerLifecycleSystem {
    /// Advance producer ages, recompute growth stages, update appearances.
    fn tick(&self, world: &mut hecs::World, dt: f32);
}

/// Default lifecycle: Cell → Patch → Mature → Broadcasting/Collapsing.
pub struct DefaultProducerLifecycle;

impl ProducerLifecycleSystem for DefaultProducerLifecycle {
    fn tick(&self, world: &mut hecs::World, dt: f32) {
        producer_lifecycle::producer_lifecycle_system(world, dt);
    }
}
