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
    match theme {
        RenderTheme::Classic => {
            let mut pal = ocean_palette(env);
            pal.water_bg = Color::Black;
            pal.sand_bg = Color::Black;
            pal.gravel_bg = Color::Black;
            pal
        }
        RenderTheme::Ocean => ocean_palette(env),
        RenderTheme::DeepSea => deep_sea_palette(env),
        RenderTheme::CoralReef => coral_reef_palette(env),
        RenderTheme::Brackish => brackish_palette(env),
        RenderTheme::RetroCrt => retro_crt_palette(env),
        RenderTheme::Blueprint => blueprint_palette(env),
        RenderTheme::Frozen => frozen_palette(env),
    }
}

/// Ocean palette — the default colored-background look.
fn ocean_palette(env: &Environment) -> Palette {
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

/// Deep Sea — very dark, bioluminescent feel. Minimal day/night variation.
fn deep_sea_palette(env: &Environment) -> Palette {
    let title = event_title(env);
    let l = env.light_level;
    let dim = (l * 0.3).max(0.05);
    Palette {
        water_fg: Color::Rgb((10.0 * dim) as u8, (15.0 * dim) as u8, (50.0 + 30.0 * dim) as u8),
        water_bg: Color::Rgb(0, (2.0 * dim) as u8, (8.0 + 4.0 * dim) as u8),
        border: Color::Rgb(20, 30, 80),
        sand: Color::Rgb((40.0 * dim) as u8, (35.0 * dim) as u8, (20.0 * dim) as u8),
        sand_bg: Color::Rgb(5, 4, 2),
        gravel: Color::Rgb(20, 20, 25),
        gravel_bg: Color::Rgb(3, 3, 5),
        title,
    }
}

/// Coral Reef — warm turquoise water, pinkish coral-sand substrate.
fn coral_reef_palette(env: &Environment) -> Palette {
    let title = event_title(env);
    let l = env.light_level.max(0.1);
    Palette {
        water_fg: Color::Rgb(0, (140.0 * l) as u8, (160.0 * l) as u8),
        water_bg: Color::Rgb(0, (25.0 * l) as u8, (35.0 * l) as u8),
        border: Color::Rgb((180.0 * l) as u8, (100.0 * l) as u8, (80.0 * l) as u8),
        sand: Color::Rgb((220.0 * l) as u8, (180.0 * l) as u8, (150.0 * l) as u8),
        sand_bg: Color::Rgb((50.0 * l) as u8, (35.0 * l) as u8, (28.0 * l) as u8),
        gravel: Color::Rgb((160.0 * l) as u8, (120.0 * l) as u8, (100.0 * l) as u8),
        gravel_bg: Color::Rgb((35.0 * l) as u8, (25.0 * l) as u8, (20.0 * l) as u8),
        title,
    }
}

/// Brackish — murky green-brown water, dark mud.
fn brackish_palette(env: &Environment) -> Palette {
    let title = event_title(env);
    let l = env.light_level.max(0.1);
    Palette {
        water_fg: Color::Rgb((40.0 * l) as u8, (60.0 * l) as u8, (20.0 * l) as u8),
        water_bg: Color::Rgb((8.0 * l) as u8, (12.0 * l) as u8, (4.0 * l) as u8),
        border: Color::Rgb((80.0 * l) as u8, (70.0 * l) as u8, (30.0 * l) as u8),
        sand: Color::Rgb((100.0 * l) as u8, (80.0 * l) as u8, (40.0 * l) as u8),
        sand_bg: Color::Rgb((25.0 * l) as u8, (20.0 * l) as u8, (8.0 * l) as u8),
        gravel: Color::Rgb((60.0 * l) as u8, (50.0 * l) as u8, (30.0 * l) as u8),
        gravel_bg: Color::Rgb((15.0 * l) as u8, (12.0 * l) as u8, (5.0 * l) as u8),
        title,
    }
}

/// Retro CRT — green phosphor on black.
fn retro_crt_palette(env: &Environment) -> Palette {
    let title = event_title(env);
    let l = env.light_level.max(0.15);
    Palette {
        water_fg: Color::Rgb(0, (80.0 * l) as u8, 0),
        water_bg: Color::Rgb(0, (6.0 * l) as u8, 0),
        border: Color::Rgb(0, (180.0 * l) as u8, 0),
        sand: Color::Rgb(0, (120.0 * l) as u8, 0),
        sand_bg: Color::Rgb(0, (12.0 * l) as u8, 0),
        gravel: Color::Rgb(0, (90.0 * l) as u8, 0),
        gravel_bg: Color::Rgb(0, (8.0 * l) as u8, 0),
        title,
    }
}

/// Blueprint — white/cyan on dark blue.
fn blueprint_palette(env: &Environment) -> Palette {
    let title = event_title(env);
    let l = env.light_level.max(0.15);
    Palette {
        water_fg: Color::Rgb((60.0 * l) as u8, (100.0 * l) as u8, (200.0 * l.min(1.0)) as u8),
        water_bg: Color::Rgb((5.0 * l) as u8, (10.0 * l) as u8, (30.0 * l) as u8),
        border: Color::Rgb((120.0 * l) as u8, (160.0 * l) as u8, (220.0 * l.min(1.0)) as u8),
        sand: Color::Rgb((100.0 * l) as u8, (140.0 * l) as u8, (200.0 * l.min(1.0)) as u8),
        sand_bg: Color::Rgb((8.0 * l) as u8, (15.0 * l) as u8, (40.0 * l) as u8),
        gravel: Color::Rgb((80.0 * l) as u8, (120.0 * l) as u8, (180.0 * l) as u8),
        gravel_bg: Color::Rgb((6.0 * l) as u8, (12.0 * l) as u8, (35.0 * l) as u8),
        title,
    }
}

/// Frozen — pale icy blue water, white substrate.
fn frozen_palette(env: &Environment) -> Palette {
    let title = event_title(env);
    let l = env.light_level.max(0.15);
    Palette {
        water_fg: Color::Rgb((120.0 * l) as u8, (160.0 * l) as u8, (200.0 * l.min(1.0)) as u8),
        water_bg: Color::Rgb((12.0 * l) as u8, (18.0 * l) as u8, (30.0 * l) as u8),
        border: Color::Rgb((180.0 * l) as u8, (200.0 * l.min(1.0)) as u8, (220.0 * l.min(1.0)) as u8),
        sand: Color::Rgb((200.0 * l.min(1.0)) as u8, (210.0 * l.min(1.0)) as u8, (220.0 * l.min(1.0)) as u8),
        sand_bg: Color::Rgb((40.0 * l) as u8, (45.0 * l) as u8, (50.0 * l) as u8),
        gravel: Color::Rgb((160.0 * l) as u8, (175.0 * l) as u8, (190.0 * l) as u8),
        gravel_bg: Color::Rgb((30.0 * l) as u8, (35.0 * l) as u8, (40.0 * l) as u8),
        title,
    }
}

fn event_title(env: &Environment) -> &'static str {
    if let Some(ref event) = env.active_event {
        match event.kind {
            EventKind::AlgaeBloom => " tuiquarium [ALGAE BLOOM] ",
            EventKind::ColdSnap => " tuiquarium [COLD SNAP] ",
            EventKind::FeedingFrenzy => " tuiquarium [FEEDING FRENZY] ",
            EventKind::Earthquake => " tuiquarium [EARTHQUAKE!] ",
        }
    } else {
        " tuiquarium "
    }
}
