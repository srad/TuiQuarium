//! Need-driven behavior system — picks actions based on highest unmet need.

use crate::components::AnimAction;
use crate::needs::Needs;

/// The current behavioral action being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorAction {
    Flee,
    Surface,
    Hunt,
    Forage,
    Rest,
    SeekComfort,
    MateSeek,
    Patrol,
    School,
    Explore,
    Idle,
}

impl BehaviorAction {
    /// Map behavioral action to animation action.
    pub fn to_anim_action(self) -> AnimAction {
        match self {
            BehaviorAction::Flee => AnimAction::Swim, // fast swim
            BehaviorAction::Surface => AnimAction::Swim,
            BehaviorAction::Hunt => AnimAction::Swim,
            BehaviorAction::Forage => AnimAction::Swim,
            BehaviorAction::Rest => AnimAction::Idle,
            BehaviorAction::SeekComfort => AnimAction::Swim,
            BehaviorAction::MateSeek => AnimAction::Swim,
            BehaviorAction::Patrol => AnimAction::Swim,
            BehaviorAction::School => AnimAction::Swim,
            BehaviorAction::Explore => AnimAction::Swim,
            BehaviorAction::Idle => AnimAction::Idle,
        }
    }

    /// Speed multiplier: how fast the creature should move for this action.
    pub fn speed_multiplier(self) -> f32 {
        match self {
            BehaviorAction::Flee => 1.5,
            BehaviorAction::Hunt => 1.2,
            BehaviorAction::Rest => 0.1,
            BehaviorAction::Idle => 0.2,
            BehaviorAction::Forage => 0.8,
            BehaviorAction::School => 1.0,
            BehaviorAction::Explore => 0.6,
            BehaviorAction::Patrol => 0.7,
            BehaviorAction::SeekComfort => 0.5,
            BehaviorAction::MateSeek => 0.9,
            BehaviorAction::Surface => 1.0,
        }
    }
}

/// The behavior state of a creature.
#[derive(Debug, Clone)]
pub struct BehaviorState {
    pub action: BehaviorAction,
    pub action_timer: f32,
}

impl Default for BehaviorState {
    fn default() -> Self {
        Self {
            action: BehaviorAction::Idle,
            action_timer: 0.0,
        }
    }
}

/// Select the highest-priority action based on current needs.
pub fn select_action(needs: &Needs, is_predator: bool) -> BehaviorAction {
    // Priority cascade — highest unmet need wins
    if needs.safety > 0.8 {
        return BehaviorAction::Flee;
    }
    if needs.oxygen > 0.7 {
        return BehaviorAction::Surface;
    }
    if needs.hunger > 0.7 {
        return if is_predator {
            BehaviorAction::Hunt
        } else {
            BehaviorAction::Forage
        };
    }
    if needs.rest > 0.8 {
        return BehaviorAction::Rest;
    }
    if needs.comfort > 0.7 {
        return BehaviorAction::SeekComfort;
    }
    if needs.reproduction > 0.8 {
        return BehaviorAction::MateSeek;
    }
    if needs.territory > 0.7 {
        return BehaviorAction::Patrol;
    }
    if needs.social > 0.6 {
        return BehaviorAction::School;
    }
    if needs.curiosity > 0.5 {
        return BehaviorAction::Explore;
    }
    BehaviorAction::Idle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flee_highest_priority() {
        let needs = Needs {
            safety: 0.9,
            hunger: 0.95,
            ..Default::default()
        };
        assert_eq!(select_action(&needs, false), BehaviorAction::Flee);
    }

    #[test]
    fn test_hunger_triggers_forage() {
        let needs = Needs {
            hunger: 0.8,
            ..Default::default()
        };
        assert_eq!(select_action(&needs, false), BehaviorAction::Forage);
    }

    #[test]
    fn test_hunger_triggers_hunt_for_predator() {
        let needs = Needs {
            hunger: 0.8,
            ..Default::default()
        };
        assert_eq!(select_action(&needs, true), BehaviorAction::Hunt);
    }

    #[test]
    fn test_default_is_idle() {
        let needs = Needs {
            hunger: 0.1,
            safety: 0.0,
            rest: 0.1,
            social: 0.1,
            curiosity: 0.1,
            ..Default::default()
        };
        assert_eq!(select_action(&needs, false), BehaviorAction::Idle);
    }

    #[test]
    fn test_reproduction_after_survival() {
        let needs = Needs {
            hunger: 0.3,
            safety: 0.1,
            rest: 0.2,
            reproduction: 0.9,
            ..Default::default()
        };
        assert_eq!(select_action(&needs, false), BehaviorAction::MateSeek);
    }
}
