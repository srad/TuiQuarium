//! Visual effects: bubbles, light rays.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

/// A single bubble particle.
#[derive(Debug, Clone)]
pub struct Bubble {
    pub x: f32,
    pub y: f32,
    pub size: u8, // 0 = '.', 1 = 'o', 2 = 'O'
}

/// Manages cosmetic bubble particles.
pub struct BubbleSystem {
    bubbles: Vec<Bubble>,
    spawn_timer: f32,
}

impl BubbleSystem {
    pub fn new() -> Self {
        Self {
            bubbles: Vec::new(),
            spawn_timer: 0.0,
        }
    }

    /// Advance bubbles upward, remove those that left the screen, spawn new ones.
    pub fn tick(&mut self, dt: f32, tank_w: f32, tank_h: f32) {
        // Move bubbles up
        for b in &mut self.bubbles {
            b.y -= 2.0 * dt; // rise speed
            b.x += (b.y * 0.5).sin() * 0.3 * dt; // gentle wobble
        }

        // Remove bubbles that exited the top
        self.bubbles.retain(|b| b.y > 0.0);

        // Spawn new bubbles
        self.spawn_timer += dt;
        if self.spawn_timer > 1.5 {
            self.spawn_timer = 0.0;
            // Simple deterministic spawn based on tank size
            let spawn_x = ((self.bubbles.len() as f32 * 7.3) % tank_w).max(1.0);
            self.bubbles.push(Bubble {
                x: spawn_x,
                y: tank_h - 3.0,
                size: (self.bubbles.len() % 3) as u8,
            });
        }
    }

    /// Render bubbles into the buffer.
    pub fn render(&self, buf: &mut Buffer, area: Rect, light_level: f32) {
        let alpha = if light_level > 0.3 { 1.0 } else { 0.5 };
        let color = if alpha > 0.7 {
            Color::LightCyan
        } else {
            Color::DarkGray
        };
        let style = Style::default().fg(color);

        for b in &self.bubbles {
            let sx = area.x as i32 + b.x as i32;
            let sy = area.y as i32 + b.y as i32;

            if sx < area.x as i32
                || sx >= (area.x + area.width) as i32
                || sy < area.y as i32
                || sy >= (area.y + area.height) as i32
            {
                continue;
            }

            let ch = match b.size {
                0 => '.',
                1 => 'o',
                _ => 'O',
            };

            if let Some(cell) = buf.cell_mut((sx as u16, sy as u16)) {
                cell.set_char(ch).set_style(style);
            }
        }
    }
}
