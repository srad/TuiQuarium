//! Color palette that shifts with day/night cycle and environment events.

use crate::RenderTheme;
use ratatui::style::Color;
use tuiq_core::environment::{Environment, EventKind};

/// Palette derived from the current environment state.
pub struct Palette {
    pub water_fg: Color,
    pub water_bg: Color,
    pub border: Color,
    pub sand: Color,
    pub sand_bg: Color,
    pub gravel: Color,
    pub gravel_bg: Color,
    pub title: &'static str,
}

/// Compute the rendering palette from environment state and active theme.
pub fn palette_for(env: &Environment, theme: RenderTheme) -> Palette {
    let mut pal = base_palette(env);
    if theme == RenderTheme::Classic {
        // Classic: all backgrounds black
        pal.water_bg = Color::Black;
        pal.sand_bg = Color::Black;
        pal.gravel_bg = Color::Black;
    }
    pal
}

/// Build the base palette (Ocean-style bg colors). Classic overrides to black.
fn base_palette(env: &Environment) -> Palette {
    // Check for active event overrides first
    if let Some(ref event) = env.active_event {
        match event.kind {
            EventKind::AlgaeBloom => {
                return Palette {
                    water_fg: Color::Green,
                    water_bg: Color::Rgb(0, 30, 10),
                    border: Color::Green,
                    sand: Color::Yellow,
                    sand_bg: Color::Rgb(60, 50, 10),
                    gravel: Color::DarkGray,
                    gravel_bg: Color::Rgb(30, 30, 15),
                    title: " tuiquarium [ALGAE BLOOM] ",
                };
            }
            EventKind::ColdSnap => {
                return Palette {
                    water_fg: Color::LightCyan,
                    water_bg: Color::Rgb(10, 20, 40),
                    border: Color::LightCyan,
                    sand: Color::White,
                    sand_bg: Color::Rgb(50, 50, 55),
                    gravel: Color::Gray,
                    gravel_bg: Color::Rgb(35, 35, 40),
                    title: " tuiquarium [COLD SNAP] ",
                };
            }
            EventKind::FeedingFrenzy => {
                return Palette {
                    water_fg: Color::LightYellow,
                    water_bg: Color::Rgb(10, 15, 40),
                    border: Color::Yellow,
                    sand: Color::Yellow,
                    sand_bg: Color::Rgb(60, 45, 10),
                    gravel: Color::DarkGray,
                    gravel_bg: Color::Rgb(35, 30, 15),
                    title: " tuiquarium [FEEDING FRENZY] ",
                };
            }
            EventKind::Earthquake => {
                return Palette {
                    water_fg: Color::Red,
                    water_bg: Color::Rgb(25, 10, 10),
                    border: Color::LightRed,
                    sand: Color::Yellow,
                    sand_bg: Color::Rgb(55, 40, 10),
                    gravel: Color::DarkGray,
                    gravel_bg: Color::Rgb(35, 25, 10),
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
            water_bg: Color::Rgb(0, 20, 50),
            border: Color::Cyan,
            sand: Color::Yellow,
            sand_bg: Color::Rgb(60, 45, 15),
            gravel: Color::DarkGray,
            gravel_bg: Color::Rgb(35, 30, 20),
            title: " tuiquarium ",
        }
    } else if env.light_level > 0.4 {
        // Dusk/dawn
        Palette {
            water_fg: Color::Rgb(80, 80, 180),
            water_bg: Color::Rgb(0, 12, 35),
            border: Color::Rgb(200, 130, 50),
            sand: Color::Rgb(180, 150, 50),
            sand_bg: Color::Rgb(45, 35, 10),
            gravel: Color::DarkGray,
            gravel_bg: Color::Rgb(28, 25, 15),
            title: " tuiquarium ",
        }
    } else if env.light_level > 0.15 {
        // Twilight
        Palette {
            water_fg: Color::DarkGray,
            water_bg: Color::Rgb(0, 8, 22),
            border: Color::Rgb(60, 60, 120),
            sand: Color::Rgb(100, 90, 40),
            sand_bg: Color::Rgb(30, 25, 8),
            gravel: Color::DarkGray,
            gravel_bg: Color::Rgb(20, 18, 10),
            title: " tuiquarium ",
        }
    } else {
        // Night
        Palette {
            water_fg: Color::Rgb(20, 20, 60),
            water_bg: Color::Rgb(0, 4, 12),
            border: Color::Rgb(30, 30, 80),
            sand: Color::Rgb(60, 55, 30),
            sand_bg: Color::Rgb(18, 15, 5),
            gravel: Color::Rgb(30, 30, 30),
            gravel_bg: Color::Rgb(12, 10, 5),
            title: " tuiquarium ",
        }
    }
}
