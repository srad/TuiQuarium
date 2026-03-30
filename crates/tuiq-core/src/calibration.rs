//! Centralized startup and ecology calibration parameters.

/// Default startup and ecology calibration for the unified aquatic model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcologyCalibration {
    pub producers_per_1000_cells: f32,
    pub min_consumer_founders: usize,
    pub max_consumer_founders: usize,
    pub founder_spacing: f32,
    pub consumer_biomass_share: f32,
    pub edible_biomass_bonus: f32,
    pub producer_growth_multiplier: f32,
    pub producer_maintenance_multiplier: f32,
    pub producer_turnover_multiplier: f32,
    pub producer_nutrient_demand_multiplier: f32,
    pub phytoplankton_shading_multiplier: f32,
    pub consumer_metabolism_multiplier: f32,
}

/// Evolutionary calibration layered on top of the ecological baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvolutionCalibration {
    pub creature_mutation_multiplier: f32,
    pub producer_mutation_multiplier: f32,
    pub fitness_sharing_strength: f32,
}

/// Full runtime calibration bundle used by the simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeCalibration {
    pub ecology: EcologyCalibration,
    pub evolution: EvolutionCalibration,
}

impl Default for EcologyCalibration {
    fn default() -> Self {
        Self {
            producers_per_1000_cells: 14.0,
            min_consumer_founders: 3,
            max_consumer_founders: 5,
            founder_spacing: 2.6,
            consumer_biomass_share: 0.035,
            edible_biomass_bonus: 0.05,
            producer_growth_multiplier: 1.0,
            producer_maintenance_multiplier: 1.0,
            producer_turnover_multiplier: 1.0,
            producer_nutrient_demand_multiplier: 0.50,
            phytoplankton_shading_multiplier: 1.0,
            consumer_metabolism_multiplier: 0.50,
        }
    }
}

impl EcologyCalibration {
    pub fn target_producer_count(
        self,
        tank_width: u16,
        tank_height: u16,
        max_producers: usize,
    ) -> usize {
        let area = tank_width as f32 * tank_height as f32;
        ((area / 1000.0) * self.producers_per_1000_cells)
            .round()
            .clamp(12.0, max_producers.max(12) as f32) as usize
    }

    pub fn target_consumer_biomass(self, producer_biomass: f32, active_biomass: f32) -> f32 {
        producer_biomass.max(0.0) * self.consumer_biomass_share
            + active_biomass.max(0.0) * self.edible_biomass_bonus
    }
}

impl Default for EvolutionCalibration {
    fn default() -> Self {
        Self {
            creature_mutation_multiplier: 1.0,
            producer_mutation_multiplier: 1.0,
            fitness_sharing_strength: 1.0,
        }
    }
}

impl Default for RuntimeCalibration {
    fn default() -> Self {
        Self {
            ecology: EcologyCalibration::default(),
            evolution: EvolutionCalibration::default(),
        }
    }
}

impl From<EcologyCalibration> for RuntimeCalibration {
    fn from(ecology: EcologyCalibration) -> Self {
        Self {
            ecology,
            evolution: EvolutionCalibration::default(),
        }
    }
}
