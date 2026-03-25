//! Realistic animal needs system — drives behavior through a priority hierarchy.

/// Each creature's needs, ranging from 0.0 (fully satisfied) to 1.0 (critical).
#[derive(Debug, Clone)]
pub struct Needs {
    // Survival needs (highest priority)
    pub hunger: f32,
    pub safety: f32,
    pub oxygen: f32,

    // Wellbeing needs
    pub rest: f32,
    pub comfort: f32,
    pub social: f32,

    // Growth needs (lowest priority)
    pub reproduction: f32,
    pub territory: f32,
    pub curiosity: f32,
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            hunger: 0.3,
            safety: 0.0,
            oxygen: 0.0,
            rest: 0.2,
            comfort: 0.1,
            social: 0.3,
            reproduction: 0.0,
            territory: 0.0,
            curiosity: 0.3,
        }
    }
}

impl Needs {
    /// Clamp all needs to [0, 1].
    pub fn clamp(&mut self) {
        self.hunger = self.hunger.clamp(0.0, 1.0);
        self.safety = self.safety.clamp(0.0, 1.0);
        self.oxygen = self.oxygen.clamp(0.0, 1.0);
        self.rest = self.rest.clamp(0.0, 1.0);
        self.comfort = self.comfort.clamp(0.0, 1.0);
        self.social = self.social.clamp(0.0, 1.0);
        self.reproduction = self.reproduction.clamp(0.0, 1.0);
        self.territory = self.territory.clamp(0.0, 1.0);
        self.curiosity = self.curiosity.clamp(0.0, 1.0);
    }
}

/// Weights that control how quickly each need accumulates — species personality.
#[derive(Debug, Clone)]
pub struct NeedWeights {
    pub hunger_rate: f32,
    pub safety_sensitivity: f32,
    pub rest_rate: f32,
    pub comfort_sensitivity: f32,
    pub social_need: f32,
    pub reproduction_rate: f32,
    pub territory_need: f32,
    pub curiosity_rate: f32,
}

impl Default for NeedWeights {
    fn default() -> Self {
        Self {
            hunger_rate: 0.02,
            safety_sensitivity: 0.5,
            rest_rate: 0.01,
            comfort_sensitivity: 0.3,
            social_need: 0.5,
            reproduction_rate: 0.02,
            territory_need: 0.0,
            curiosity_rate: 0.01,
        }
    }
}

/// Update all creature needs based on time passing and environment.
/// This is a simple per-tick update — environmental reactions (safety from predators)
/// are handled by the behavior system which has access to the spatial grid.
pub fn needs_tick(needs: &mut Needs, weights: &NeedWeights, dt: f32) {
    // Hunger rises steadily (faster with higher metabolism)
    needs.hunger += weights.hunger_rate * dt;

    // Safety decays toward 0 when no threat (relaxation)
    needs.safety *= (1.0 - 0.5 * dt).max(0.0);

    // Oxygen stays low in normal conditions
    needs.oxygen *= (1.0 - 1.0 * dt).max(0.0);

    // Rest accumulates slowly
    needs.rest += weights.rest_rate * dt;

    // Comfort decays toward 0
    needs.comfort *= (1.0 - 0.3 * dt).max(0.0);

    // Social need drifts toward the species baseline
    let social_target = weights.social_need;
    needs.social += (social_target - needs.social) * 0.1 * dt;

    // Reproduction builds over time
    needs.reproduction += weights.reproduction_rate * dt;

    // Territory need drifts toward species baseline
    let territory_target = weights.territory_need;
    needs.territory += (territory_target - needs.territory) * 0.05 * dt;

    // Curiosity fluctuates
    needs.curiosity += weights.curiosity_rate * dt;

    needs.clamp();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hunger_rises() {
        let mut needs = Needs::default();
        let weights = NeedWeights::default();
        let initial = needs.hunger;
        for _ in 0..100 {
            needs_tick(&mut needs, &weights, 0.05);
        }
        assert!(needs.hunger > initial, "Hunger should rise over time");
    }

    #[test]
    fn test_safety_decays() {
        let mut needs = Needs::default();
        needs.safety = 0.9;
        let weights = NeedWeights::default();
        for _ in 0..100 {
            needs_tick(&mut needs, &weights, 0.05);
        }
        assert!(needs.safety < 0.5, "Safety concern should decay without threat");
    }

    #[test]
    fn test_needs_stay_clamped() {
        let mut needs = Needs::default();
        needs.hunger = 0.99;
        let weights = NeedWeights {
            hunger_rate: 10.0,
            ..Default::default()
        };
        for _ in 0..100 {
            needs_tick(&mut needs, &weights, 0.05);
        }
        assert!(needs.hunger <= 1.0, "Hunger should not exceed 1.0");
    }
}
