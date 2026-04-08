use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders},
    Frame,
};
use tuiq_core::{
    components::{AnimationState, Appearance, Direction, Position},
    Simulation,
};

use crate::effects::BubbleSystem;
use crate::palette;
use crate::screenshot::color_to_rgb;
use crate::RenderTheme;

/// Render the aquarium tank: border, water fill, sand substrate, creatures, and effects.
pub fn render_tank(
    frame: &mut Frame,
    area: Rect,
    sim: &dyn Simulation,
    bubbles: &BubbleSystem,
    theme: RenderTheme,
) {
    let env = sim.environment();
    let pal = palette::palette_for(env, theme);
    let phytoplankton_load = sim.ecology_diagnostics().instant.phytoplankton_load;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(pal.border))
        .title(pal.title);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let inner_width = inner.width as usize;
    let inner_height = inner.height as usize;

    if inner_width == 0 || inner_height == 0 {
        return;
    }

    // Fill water and substrate into the buffer directly
    let substrate_height = 2.min(inner_height);
    let water_height = inner_height.saturating_sub(substrate_height);
    let water_style = Style::default().fg(pal.water_fg).bg(pal.water_bg);
    let sand_style = Style::default().fg(pal.sand).bg(pal.sand_bg);
    let gravel_style = Style::default().fg(pal.gravel).bg(pal.gravel_bg);

    let buf = frame.buffer_mut();

    // Water rows
    for row in 0..water_height {
        let y = inner.y + row as u16;
        for col in 0..inner_width {
            let x = inner.x + col as u16;
            let ch = if (row + col) % 4 == 0 || (row + col) % 4 == 2 {
                '~'
            } else {
                ' '
            };
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(ch).set_style(water_style);
            }
        }
    }

    // Substrate rows
    for row in 0..substrate_height {
        let y = inner.y + (water_height + row) as u16;
        let style = if row == 0 { gravel_style } else { sand_style };
        for col in 0..inner_width {
            let x = inner.x + col as u16;
            let ch = if row == 0 {
                match (col * 7 + row * 3) % 5 {
                    0 | 2 => ':',
                    1 | 3 => '.',
                    _ => ',',
                }
            } else {
                match (col * 3 + row * 7) % 3 {
                    0 | 2 => ':',
                    _ => '.',
                }
            };
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(ch).set_style(style);
            }
        }
    }

    render_phytoplankton(buf, inner, water_height, env, phytoplankton_load);

    // Render bubbles
    bubbles.render(buf, inner, env.light_level);

    // Render creatures from the ECS world
    let color_fn: fn(u8) -> Color = theme.creature_color_override().unwrap_or(creature_color);
    let world = sim.world();
    for (pos, appearance, anim) in &mut world.query::<(&Position, &Appearance, &AnimationState)>() {
        render_creature(buf, inner, pos, appearance, anim, color_fn);
    }
}

fn render_phytoplankton(
    buf: &mut Buffer,
    area: Rect,
    water_height: usize,
    env: &tuiq_core::environment::Environment,
    phytoplankton_load: f32,
) {
    let visible_load = phytoplankton_visibility(phytoplankton_load, env.light_level);
    if visible_load <= 0.01 || area.width == 0 || water_height == 0 {
        return;
    }

    let bloom_boost = if matches!(
        env.active_event.as_ref().map(|event| event.kind),
        Some(tuiq_core::environment::EventKind::AlgaeBloom)
    ) {
        0.10
    } else {
        0.0
    };
    let time_seed = (env.time_of_day * 5.0).floor() as i32;

    for row in 0..water_height {
        let y = area.y + row as u16;
        for col in 0..area.width as usize {
            let x = area.x + col as u16;
            let broad_a = noise01(col as i32 / 5, row as i32 / 3, time_seed + 17);
            let broad_b = noise01((col as i32 + 13) / 9, (row as i32 + 7) / 4, time_seed + 61);
            let broad = broad_a * 0.58 + broad_b * 0.42;
            let patch_strength =
                (broad - (0.76 - visible_load * 0.38 - bloom_boost)).clamp(0.0, 1.0);
            if patch_strength <= 0.0 {
                continue;
            }

            if let Some(cell) = buf.cell_mut((x, y)) {
                let base_bg = color_to_rgb(cell.bg);
                let base_fg = color_to_rgb(cell.fg);
                let bg_alpha = (0.08 + patch_strength * 0.24 + visible_load * 0.10).min(0.42);
                let fg_alpha = (0.06 + patch_strength * 0.18 + visible_load * 0.08).min(0.32);
                let bg = blend_rgb(base_bg, [16, 54, 24], bg_alpha);
                let fg = blend_rgb(base_fg, [110, 170, 88], fg_alpha);

                let existing = cell.symbol().chars().next().unwrap_or(' ');
                let fine = noise01(col as i32, row as i32, time_seed + 113);
                let clustered = ((col as i32 + row as i32 * 3 + time_seed) % 11) == 0;
                let mut ch = existing;
                if (existing == ' ' || existing == '~')
                    && (fine > 0.88 - visible_load * 0.22 - patch_strength * 0.18
                        || (visible_load > 0.55 && clustered && existing == ' '))
                {
                    ch = match ((fine * 1000.0) as i32 + row as i32 + col as i32 + time_seed) & 3 {
                        0 => '.',
                        1 => ',',
                        2 => ':',
                        _ => ';',
                    };
                }

                cell.set_char(ch).set_style(
                    Style::default()
                        .fg(Color::Rgb(fg[0], fg[1], fg[2]))
                        .bg(Color::Rgb(bg[0], bg[1], bg[2])),
                );
            }
        }
    }
}

fn phytoplankton_visibility(load: f32, light_level: f32) -> f32 {
    let density = ((load - 0.07) / 0.78).clamp(0.0, 1.0);
    density * (0.35 + light_level.clamp(0.0, 1.0) * 0.65)
}

fn noise01(x: i32, y: i32, seed: i32) -> f32 {
    let mut n =
        x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263) ^ seed.wrapping_mul(69_069);
    n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
    ((n ^ (n >> 16)) as u32) as f32 / u32::MAX as f32
}

fn blend_rgb(base: [u8; 3], tint: [u8; 3], alpha: f32) -> [u8; 3] {
    let a = alpha.clamp(0.0, 1.0);
    [
        (base[0] as f32 * (1.0 - a) + tint[0] as f32 * a).round() as u8,
        (base[1] as f32 * (1.0 - a) + tint[1] as f32 * a).round() as u8,
        (base[2] as f32 * (1.0 - a) + tint[2] as f32 * a).round() as u8,
    ]
}

/// Map color_index to a terminal color.
fn creature_color(color_index: u8) -> Color {
    match color_index {
        0 => Color::LightCyan,          // Small fish
        1 => Color::LightYellow,        // Tropical fish
        2 => Color::White,              // Angelfish
        3 => Color::LightMagenta,       // Jellyfish
        4 => Color::LightRed,           // Crab
        5 => Color::Green,              // Seaweed / plant green
        6 => Color::Rgb(255, 165, 0),   // Orange / plant brown
        7 => Color::Rgb(200, 100, 200), // Purple
        8 => Color::Rgb(0, 120, 60),    // Dark green (plants)
        9 => Color::Rgb(180, 220, 40),  // Yellow-green / lime (plants)
        10 => Color::Rgb(0, 180, 140),  // Teal / sea green (plants)
        11 => Color::Rgb(160, 82, 45),  // Sienna / rust brown (plants)
        _ => Color::White,
    }
}

fn render_creature(
    buf: &mut Buffer,
    area: Rect,
    pos: &Position,
    appearance: &Appearance,
    anim: &AnimationState,
    color_fn: fn(u8) -> Color,
) {
    let action_idx = anim.current_action as usize;
    let frame_set = match appearance.frame_sets.get(action_idx) {
        Some(fs) if !fs.is_empty() => fs,
        _ => return,
    };
    let frame_idx = anim.frame_index.min(frame_set.len() - 1);
    let art_frame = &frame_set[frame_idx];

    let flip = appearance.facing == Direction::Left;

    for (row_idx, row_str) in art_frame.rows.iter().enumerate() {
        let screen_y = pos.y as i32 + row_idx as i32;
        if screen_y < 0 || screen_y >= area.height as i32 {
            continue;
        }
        let y = area.y + screen_y as u16;

        let row_len = row_str.chars().count();
        for (col_idx, ch) in row_str.chars().enumerate() {
            if ch == ' ' {
                continue; // Transparent
            }
            // When facing left, mirror the column index (no allocation)
            let draw_col = if flip { row_len - 1 - col_idx } else { col_idx };
            let screen_x = pos.x as i32 + draw_col as i32;
            if screen_x < 0 || screen_x >= area.width as i32 {
                continue;
            }
            let x = area.x + screen_x as u16;

            if let Some(cell) = buf.cell_mut((x, y)) {
                // Inherit the background from the water/substrate already painted
                let bg = cell.bg;
                cell.set_char(ch)
                    .set_style(Style::default().fg(color_fn(appearance.color_index)).bg(bg));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use tuiq_core::environment::Environment;

    fn seed_water_buffer(area: Rect) -> Buffer {
        let mut buf = Buffer::empty(area);
        let water_style = Style::default().fg(Color::Blue).bg(Color::Rgb(0, 20, 50));
        for row in 0..area.height as usize {
            for col in 0..area.width as usize {
                let ch = if (row + col) % 4 == 0 || (row + col) % 4 == 2 {
                    '~'
                } else {
                    ' '
                };
                buf.cell_mut((area.x + col as u16, area.y + row as u16))
                    .unwrap()
                    .set_char(ch)
                    .set_style(water_style);
            }
        }
        buf
    }

    #[test]
    fn phytoplankton_overlay_is_invisible_near_floor() {
        let area = Rect::new(0, 0, 32, 10);
        let mut buf = seed_water_buffer(area);
        let before = buf.clone();
        let env = Environment::default();

        render_phytoplankton(&mut buf, area, area.height as usize, &env, 0.05);

        let changed = (0..area.height)
            .flat_map(|row| (0..area.width).map(move |col| (row, col)))
            .filter(|(row, col)| {
                let a = before.cell((area.x + *col, area.y + *row)).unwrap();
                let b = buf.cell((area.x + *col, area.y + *row)).unwrap();
                a.symbol() != b.symbol() || a.bg != b.bg || a.fg != b.fg
            })
            .count();

        assert_eq!(
            changed, 0,
            "Low phytoplankton should not visibly alter water"
        );
    }

    #[test]
    fn phytoplankton_overlay_adds_haze_and_specks_at_high_load() {
        let area = Rect::new(0, 0, 48, 14);
        let mut buf = seed_water_buffer(area);
        let before = buf.clone();
        let mut env = Environment::default();
        env.light_level = 0.85;

        render_phytoplankton(&mut buf, area, area.height as usize, &env, 0.78);

        let changed = (0..area.height)
            .flat_map(|row| (0..area.width).map(move |col| (row, col)))
            .filter(|(row, col)| {
                let a = before.cell((area.x + *col, area.y + *row)).unwrap();
                let b = buf.cell((area.x + *col, area.y + *row)).unwrap();
                a.symbol() != b.symbol() || a.bg != b.bg || a.fg != b.fg
            })
            .count();
        let specks = (0..area.height)
            .flat_map(|row| (0..area.width).map(move |col| (row, col)))
            .filter(|(row, col)| {
                let ch = buf
                    .cell((area.x + *col, area.y + *row))
                    .unwrap()
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' ');
                matches!(ch, '.' | ',' | ':' | ';')
            })
            .count();

        assert!(
            changed > 20,
            "High phytoplankton should tint a visible haze, got {changed} changed cells"
        );
        assert!(
            specks > 2,
            "High phytoplankton should add suspended specks, got {specks}"
        );
    }
}
