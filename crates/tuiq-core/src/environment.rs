//! Environment simulation: day/night cycle, temperature, currents, random events.

use rand::Rng;
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// Global environment state for the aquarium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    /// Time of day in hours (0.0–24.0).
    pub time_of_day: f32,
    /// Ambient light level (0.0 = pitch dark, 1.0 = full brightness).
    pub light_level: f32,
    /// Water temperature in degrees Celsius.
    pub temperature: f32,
    /// Global water current vector (dx, dy) in cells/sec.
    pub current: (f32, f32),
    /// Currently active event, if any.
    pub active_event: Option<EnvironmentEvent>,
    /// Accumulated sim time in seconds (for current rotation).
    sim_time: f32,
    /// Time until next random event roll (seconds).
    event_cooldown: f32,
    /// Whether stochastic events are enabled.
    random_events_enabled: bool,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            time_of_day: 8.0,
            light_level: 0.8,
            temperature: 25.0,
            current: (0.0, 0.0),
            active_event: None,
            sim_time: 0.0,
            event_cooldown: 60.0, // first event check after 60s
            random_events_enabled: true,
        }
    }
}

/// A random environmental event that temporarily changes conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentEvent {
    pub kind: EventKind,
    /// Remaining duration in seconds.
    pub remaining: f32,
}

/// Types of random events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    /// Plants grow faster, visibility drops.
    AlgaeBloom,
    /// Food rains from above.
    FeedingFrenzy,
    /// Temperature drops, metabolism slows.
    ColdSnap,
    /// Fish scatter — triggers flee behavior.
    Earthquake,
}

impl EventKind {
    /// Duration of this event type in seconds.
    pub fn duration(self) -> f32 {
        match self {
            EventKind::AlgaeBloom => 30.0,
            EventKind::FeedingFrenzy => 15.0,
            EventKind::ColdSnap => 25.0,
            EventKind::Earthquake => 5.0,
        }
    }

    /// Pick a random event type.
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..4) {
            0 => EventKind::AlgaeBloom,
            1 => EventKind::FeedingFrenzy,
            2 => EventKind::ColdSnap,
            _ => EventKind::Earthquake,
        }
    }
}

/// Speed of time: 1 in-game day = 300 real seconds (5 minutes).
const HOURS_PER_SECOND: f32 = 24.0 / 300.0;

/// Average interval between random events (seconds).
const EVENT_INTERVAL: f32 = 60.0;

impl Environment {
    pub fn set_random_events_enabled(&mut self, enabled: bool) {
        self.random_events_enabled = enabled;
        if !enabled {
            self.active_event = None;
        }
    }

    pub fn random_events_enabled(&self) -> bool {
        self.random_events_enabled
    }

    /// Advance the environment by dt seconds.
    pub fn tick(&mut self, dt: f32, rng: &mut impl Rng) {
        self.sim_time += dt;

        // Advance time of day
        self.time_of_day += HOURS_PER_SECOND * dt;
        if self.time_of_day >= 24.0 {
            self.time_of_day -= 24.0;
        }

        // Light level follows a sine curve: bright during day, dark at night
        // Peak at noon (12:00), trough at midnight (0:00)
        let hour_angle = (self.time_of_day - 6.0) * std::f32::consts::PI / 12.0;
        let base_light = (hour_angle.sin() * 0.5 + 0.5).clamp(0.0, 1.0);

        // Events can modify light
        let light_modifier = match &self.active_event {
            Some(e) if e.kind == EventKind::AlgaeBloom => 0.72, // reduced visibility
            _ => 1.0,
        };
        self.light_level = (base_light * light_modifier).clamp(0.05, 1.0);

        // Temperature: base 25°C with slight daily variation
        let temp_variation = hour_angle.sin() * 1.5; // ±1.5°C
        let base_temp = 25.0 + temp_variation;
        self.temperature = match &self.active_event {
            Some(e) if e.kind == EventKind::ColdSnap => base_temp - 7.0,
            _ => base_temp,
        };

        // Water current: slowly rotating vector
        let current_angle = self.sim_time * 0.05; // full rotation every ~125s
        let (current_strength, vertical_strength) = match &self.active_event {
            Some(e) if e.kind == EventKind::Earthquake => (1.2, 0.65),
            _ => (0.3, 0.3),
        };
        self.current = (
            current_angle.cos() * current_strength,
            current_angle.sin() * current_strength * vertical_strength,
        );

        // Tick active event
        if let Some(ref mut event) = self.active_event {
            event.remaining -= dt;
            if event.remaining <= 0.0 {
                self.active_event = None;
            }
        }

        // Roll for new event
        if self.random_events_enabled && self.active_event.is_none() {
            self.event_cooldown -= dt;
            if self.event_cooldown <= 0.0 {
                let kind = EventKind::random(rng);
                self.active_event = Some(EnvironmentEvent {
                    remaining: kind.duration(),
                    kind,
                });
                self.event_cooldown = EVENT_INTERVAL + rng.random_range(-20.0..20.0);
            }
        }
    }

    /// Whether it's currently nighttime (roughly 20:00–06:00).
    pub fn is_night(&self) -> bool {
        self.time_of_day >= 20.0 || self.time_of_day < 6.0
    }

    /// Temperature modifier for metabolism (higher temp = faster metabolism).
    pub fn temperature_modifier(&self) -> f32 {
        // Q10 rule simplified: 10% per degree from baseline 25°C
        let delta = self.temperature - 25.0;
        (1.0 + delta * 0.02).clamp(0.5, 2.0)
    }

    /// Light level at a given depth (y position, tank_height).
    /// Surface (y < 30% of height) gets full light.
    /// Mid-water (30-70%) gets moderate light.
    /// Deep (bottom 30%) gets reduced light.
    pub fn light_at_depth(&self, y: f32, tank_height: f32) -> f32 {
        let depth_fraction = y / tank_height; // 0.0 = top, 1.0 = bottom
        let depth_factor = if depth_fraction < 0.3 {
            1.0 // Surface zone: full light
        } else if depth_fraction < 0.7 {
            0.7 // Mid-water: moderate
        } else {
            0.4 // Deep zone: dim
        };
        self.light_level * depth_factor
    }

    /// Metabolism modifier at a given depth. Deep water is cooler = slower metabolism.
    pub fn metabolism_at_depth(&self, y: f32, tank_height: f32) -> f32 {
        let depth_fraction = y / tank_height;
        if depth_fraction > 0.7 {
            0.85 // Deep zone: 15% slower metabolism
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Substrate zones
// ---------------------------------------------------------------------------

/// Substrate type at a bottom cell. Affects producer establishment and nutrient release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Substrate {
    /// Default substrate — neutral modifiers.
    Sandy,
    /// Higher establishment for stress-tolerant producers, slower nutrient release.
    Rocky,
    /// Bonus for clonal spread and higher nutrient release from organic matter.
    Planted,
}

/// A 1-D substrate map across the tank bottom (one value per column).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateGrid {
    cols: Vec<Substrate>,
}

impl SubstrateGrid {
    /// Procedurally generate substrate zones using a seeded hash.
    /// Creates coherent patches of 5–15 columns wide.
    pub fn generate(tank_width: u16, seed: u64) -> Self {
        let w = tank_width as usize;
        let mut cols = vec![Substrate::Sandy; w];
        if w == 0 {
            return Self { cols };
        }

        // Simple deterministic zone generation using the seed.
        // Walk across columns placing zones of varying width.
        let mut x = 0usize;
        // Mix the seed to avoid poor initial state for small seeds
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        while x < w {
            // xorshift64 for lightweight deterministic randomness
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let zone_width = 5 + (state % 11) as usize; // 5–15 columns
            let substrate = match (state >> 8) % 3 {
                0 => Substrate::Sandy,
                1 => Substrate::Rocky,
                _ => Substrate::Planted,
            };

            let end = (x + zone_width).min(w);
            for col in &mut cols[x..end] {
                *col = substrate;
            }
            x = end;
        }

        Self { cols }
    }

    /// Substrate type at a given x position.
    pub fn at(&self, x: f32) -> Substrate {
        let idx = (x as usize).min(self.cols.len().saturating_sub(1));
        self.cols[idx]
    }

    /// Establishment modifier for a producer at position (x, y).
    /// Rocky boosts stress-tolerant (high hardiness) producers.
    /// Planted boosts clonal spread.
    /// Sandy is neutral.
    pub fn establishment_modifier(&self, x: f32, hardiness: f32) -> f32 {
        match self.at(x) {
            Substrate::Sandy => 1.0,
            Substrate::Rocky => 0.8 + 0.4 * hardiness, // 0.8–1.2 based on hardiness
            Substrate::Planted => 1.15,                // general bonus for established zones
        }
    }

    /// Clonal spread bonus at this substrate.
    pub fn clonal_bonus(&self, x: f32) -> f32 {
        match self.at(x) {
            Substrate::Planted => 1.3,
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_time_advances() {
        let mut env = Environment::default();
        let mut rng = StdRng::seed_from_u64(42);
        let initial = env.time_of_day;
        env.tick(10.0, &mut rng);
        assert!(env.time_of_day > initial, "Time should advance");
    }

    #[test]
    fn test_time_wraps_at_24() {
        let mut env = Environment::default();
        let mut rng = StdRng::seed_from_u64(42);
        env.time_of_day = 23.9;
        env.tick(2.0, &mut rng); // should push past 24
        assert!(
            env.time_of_day < 24.0,
            "Time should wrap: {}",
            env.time_of_day
        );
    }

    #[test]
    fn test_light_bright_at_noon() {
        let mut env = Environment::default();
        let mut rng = StdRng::seed_from_u64(42);
        env.time_of_day = 11.9;
        env.tick(0.01, &mut rng);
        assert!(
            env.light_level > 0.8,
            "Noon should be bright: {}",
            env.light_level
        );
    }

    #[test]
    fn test_light_dark_at_midnight() {
        let mut env = Environment::default();
        let mut rng = StdRng::seed_from_u64(42);
        env.time_of_day = 23.9;
        // Advance just a tiny bit so it processes around midnight
        env.tick(0.01, &mut rng);
        assert!(
            env.light_level < 0.3,
            "Midnight should be dark: {}",
            env.light_level
        );
    }

    #[test]
    fn test_is_night() {
        let mut env = Environment::default();
        env.time_of_day = 22.0;
        assert!(env.is_night());
        env.time_of_day = 3.0;
        assert!(env.is_night());
        env.time_of_day = 12.0;
        assert!(!env.is_night());
    }

    #[test]
    fn test_temperature_modifier() {
        let mut env = Environment::default();
        env.temperature = 25.0;
        assert!((env.temperature_modifier() - 1.0).abs() < 0.01);
        env.temperature = 20.0;
        assert!(env.temperature_modifier() < 1.0);
        env.temperature = 30.0;
        assert!(env.temperature_modifier() > 1.0);
    }

    #[test]
    fn test_cold_snap_lowers_temperature() {
        let mut env = Environment::default();
        env.active_event = Some(EnvironmentEvent {
            kind: EventKind::ColdSnap,
            remaining: 10.0,
        });
        let mut rng = StdRng::seed_from_u64(42);
        env.tick(0.01, &mut rng);
        assert!(
            env.temperature < 22.0,
            "Cold snap should lower temp: {}",
            env.temperature
        );
    }

    #[test]
    fn test_event_expires() {
        let mut env = Environment::default();
        env.active_event = Some(EnvironmentEvent {
            kind: EventKind::Earthquake,
            remaining: 0.5,
        });
        let mut rng = StdRng::seed_from_u64(42);
        env.tick(1.0, &mut rng);
        assert!(env.active_event.is_none(), "Event should have expired");
    }

    #[test]
    fn test_current_changes_over_time() {
        let mut env = Environment::default();
        let mut rng = StdRng::seed_from_u64(42);
        let c1 = env.current;
        env.tick(30.0, &mut rng);
        let c2 = env.current;
        assert!(
            (c1.0 - c2.0).abs() > 0.01 || (c1.1 - c2.1).abs() > 0.01,
            "Current should change over time"
        );
    }

    #[test]
    fn test_light_at_depth_surface() {
        let mut env = Environment::default();
        env.light_level = 1.0;
        // y=10 in a 100-height tank = 10% depth (surface zone)
        let light = env.light_at_depth(10.0, 100.0);
        assert!(
            (light - 1.0).abs() < 0.01,
            "Surface zone should have full light: {}",
            light
        );
    }

    #[test]
    fn test_light_at_depth_mid() {
        let mut env = Environment::default();
        env.light_level = 1.0;
        // y=50 in a 100-height tank = 50% depth (mid-water zone)
        let light = env.light_at_depth(50.0, 100.0);
        assert!(
            (light - 0.7).abs() < 0.01,
            "Mid-water zone should have 0.7 light: {}",
            light
        );
    }

    #[test]
    fn test_light_at_depth_deep() {
        let mut env = Environment::default();
        env.light_level = 1.0;
        // y=80 in a 100-height tank = 80% depth (deep zone)
        let light = env.light_at_depth(80.0, 100.0);
        assert!(
            (light - 0.4).abs() < 0.01,
            "Deep zone should have 0.4 light: {}",
            light
        );
    }

    #[test]
    fn test_metabolism_at_depth_surface_normal() {
        let env = Environment::default();
        let meta = env.metabolism_at_depth(20.0, 100.0);
        assert!(
            (meta - 1.0).abs() < 0.01,
            "Surface metabolism should be 1.0: {}",
            meta
        );
    }

    #[test]
    fn test_metabolism_at_depth_deep_reduced() {
        let env = Environment::default();
        let meta = env.metabolism_at_depth(80.0, 100.0);
        assert!(
            (meta - 0.85).abs() < 0.01,
            "Deep metabolism should be 0.85: {}",
            meta
        );
    }

    #[test]
    fn test_substrate_grid_generates_all_types() {
        let grid = SubstrateGrid::generate(100, 42);
        let mut has_sandy = false;
        let mut has_rocky = false;
        let mut has_planted = false;
        for x in 0..100 {
            match grid.at(x as f32) {
                Substrate::Sandy => has_sandy = true,
                Substrate::Rocky => has_rocky = true,
                Substrate::Planted => has_planted = true,
            }
        }
        assert!(has_sandy, "Should generate Sandy substrate");
        assert!(has_rocky, "Should generate Rocky substrate");
        assert!(has_planted, "Should generate Planted substrate");
    }

    #[test]
    fn test_substrate_establishment_modifier_rocky_high_hardiness() {
        let grid = SubstrateGrid::generate(100, 1);
        // Rocky with max hardiness should give > 1.0 modifier
        let rocky_x = (0..100)
            .map(|x| x as f32)
            .find(|&x| grid.at(x) == Substrate::Rocky)
            .expect("Should have rocky substrate");
        let mod_high = grid.establishment_modifier(rocky_x, 1.0);
        assert!(
            mod_high > 1.0,
            "Rocky + high hardiness should boost: {}",
            mod_high
        );
        let mod_low = grid.establishment_modifier(rocky_x, 0.0);
        assert!(
            mod_low < 1.0,
            "Rocky + low hardiness should penalize: {}",
            mod_low
        );
    }

    #[test]
    fn test_substrate_clonal_bonus_planted() {
        let grid = SubstrateGrid::generate(100, 1);
        let planted_x = (0..100)
            .map(|x| x as f32)
            .find(|&x| grid.at(x) == Substrate::Planted)
            .expect("Should have planted substrate");
        assert!(
            grid.clonal_bonus(planted_x) > 1.0,
            "Planted should give clonal bonus"
        );
        let sandy_x = (0..100)
            .map(|x| x as f32)
            .find(|&x| grid.at(x) == Substrate::Sandy)
            .expect("Should have sandy substrate");
        assert!(
            (grid.clonal_bonus(sandy_x) - 1.0).abs() < 0.01,
            "Sandy should give no bonus"
        );
    }

    #[test]
    fn test_substrate_deterministic_with_seed() {
        let g1 = SubstrateGrid::generate(50, 123);
        let g2 = SubstrateGrid::generate(50, 123);
        for x in 0..50 {
            assert_eq!(
                g1.at(x as f32),
                g2.at(x as f32),
                "Same seed should produce same substrate at x={}",
                x
            );
        }
    }
}
