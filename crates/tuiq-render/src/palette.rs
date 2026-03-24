//! Color palette that shifts with day/night cycle and environment events.

use ratatui::style::Color;
use tuiq_core::environment::{Environment, EventKind};

/// Palette derived from the current environment state.
pub struct Palette {
    pub water_fg: Color,
    pub water_bg: Color,
    pub border: Color,
    pub sand: Color,
    pub gravel: Color,
    pub title: &'static str,
}

/// Compute the rendering palette from environment state.
pub fn palette_for(env: &Environment) -> Palette {
    // Check for active event overrides first
    if let Some(ref event) = env.active_event {
        match event.kind {
            EventKind::AlgaeBloom => {
                return Palette {
                    water_fg: Color::Green,
                    water_bg: Color::Black,
                    border: Color::Green,
                    sand: Color::Yellow,
                    gravel: Color::DarkGray,
                    title: " tuiquarium [ALGAE BLOOM] ",
                };
            }
            EventKind::ColdSnap => {
                return Palette {
                    water_fg: Color::LightCyan,
                    water_bg: Color::Black,
                    border: Color::LightCyan,
                    sand: Color::White,
                    gravel: Color::Gray,
                    title: " tuiquarium [COLD SNAP] ",
                };
            }
            EventKind::FeedingFrenzy => {
                return Palette {
                    water_fg: Color::LightYellow,
                    water_bg: Color::Black,
                    border: Color::Yellow,
                    sand: Color::Yellow,
                    gravel: Color::DarkGray,
                    title: " tuiquarium [FEEDING FRENZY] ",
                };
            }
            EventKind::Earthquake => {
                return Palette {
                    water_fg: Color::Red,
                    water_bg: Color::Black,
                    border: Color::LightRed,
                    sand: Color::Yellow,
                    gravel: Color::DarkGray,
                    title: " tuiquarium [EARTHQUAKE!] ",
                };
            }
        }
    }

    // Day/night based on light_level
    if env.light_level > 0.7 {
        // Bright day
        Palette {
            water_fg: Color::Blue,
            water_bg: Color::Black,
            border: Color::Cyan,
            sand: Color::Yellow,
            gravel: Color::DarkGray,
            title: " tuiquarium ",
        }
    } else if env.light_level > 0.4 {
        // Dusk/dawn
        Palette {
            water_fg: Color::Rgb(80, 80, 180),
            water_bg: Color::Black,
            border: Color::Rgb(200, 130, 50),
            sand: Color::Rgb(180, 150, 50),
            gravel: Color::DarkGray,
            title: " tuiquarium ",
        }
    } else if env.light_level > 0.15 {
        // Twilight
        Palette {
            water_fg: Color::DarkGray,
            water_bg: Color::Black,
            border: Color::Rgb(60, 60, 120),
            sand: Color::Rgb(100, 90, 40),
            gravel: Color::DarkGray,
            title: " tuiquarium ",
        }
    } else {
        // Night
        Palette {
            water_fg: Color::Rgb(20, 20, 60),
            water_bg: Color::Black,
            border: Color::Rgb(30, 30, 80),
            sand: Color::Rgb(60, 55, 30),
            gravel: Color::Rgb(30, 30, 30),
            title: " tuiquarium ",
        }
    }
}
