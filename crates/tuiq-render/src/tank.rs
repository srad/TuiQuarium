use ratatui::{
    Frame,
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders},
};
use tuiq_core::{
    Simulation,
    components::{AnimationState, Appearance, Direction, Position},
};

use crate::effects::BubbleSystem;
use crate::palette;

/// Render the aquarium tank: border, water fill, sand substrate, creatures, and effects.
pub fn render_tank(frame: &mut Frame, area: Rect, sim: &dyn Simulation, bubbles: &BubbleSystem) {
    let env = sim.environment();
    let pal = palette::palette_for(env);

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
    let water_style = Style::default().fg(pal.water_fg);
    let sand_style = Style::default().fg(pal.sand);
    let gravel_style = Style::default().fg(pal.gravel);

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

    // Render bubbles
    bubbles.render(buf, inner, env.light_level);

    // Render creatures from the ECS world
    let world = sim.world();
    for (pos, appearance, anim) in
        &mut world.query::<(&Position, &Appearance, &AnimationState)>()
    {
        render_creature(buf, inner, pos, appearance, anim);
    }
}

/// Map color_index to a terminal color.
fn creature_color(color_index: u8) -> Color {
    match color_index {
        0 => Color::LightCyan,    // Small fish
        1 => Color::LightYellow,  // Tropical fish
        2 => Color::White,        // Angelfish
        3 => Color::LightMagenta, // Jellyfish
        4 => Color::LightRed,     // Crab
        5 => Color::Green,        // Seaweed
        6 => Color::Rgb(255, 165, 0), // Orange
        7 => Color::Rgb(200, 100, 200), // Purple
        _ => Color::White,
    }
}

fn render_creature(
    buf: &mut Buffer,
    area: Rect,
    pos: &Position,
    appearance: &Appearance,
    anim: &AnimationState,
) {
    let action_idx = anim.current_action as usize;
    let frame_set = match appearance.frame_sets.get(action_idx) {
        Some(fs) if !fs.is_empty() => fs,
        _ => return,
    };
    let frame_idx = anim.frame_index.min(frame_set.len() - 1);
    let art_frame = &frame_set[frame_idx];

    let creature_style = Style::default().fg(creature_color(appearance.color_index));
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
                cell.set_char(ch).set_style(creature_style);
            }
        }
    }
}
