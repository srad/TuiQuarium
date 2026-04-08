//! Visual effects: bubbles, light rays.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

/// A single bubble particle.
#[derive(Debug, Clone)]
pub struct Bubble {
    pub x: f32,
    pub y: f32,
    pub size: u8,          // 0 = '.', 1 = 'o', 2 = 'O', 3 = '°'
    pub rise_speed: f32,   // cells per second
    pub wobble_amp: f32,   // amplitude of horizontal wobble
    pub wobble_phase: f32, // phase offset for wobble
}

/// Manages cosmetic bubble particles.
pub struct BubbleSystem {
    bubbles: Vec<Bubble>,
    spawn_timer: f32,
    next_interval: f32,
    seed: u32, // simple PRNG state for deterministic randomness without rand dependency
}

impl BubbleSystem {
    pub fn new() -> Self {
        Self {
            bubbles: Vec::new(),
            spawn_timer: 0.0,
            next_interval: 1.2,
            seed: 42,
        }
    }

    /// Simple pseudo-random number in [0, 1).
    fn next_rand(&mut self) -> f32 {
        // xorshift32
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed as f32 / u32::MAX as f32).abs()
    }

    /// Advance bubbles upward, remove those that left the screen, spawn new ones.
    pub fn tick(&mut self, dt: f32, tank_w: f32, tank_h: f32) {
        // Move bubbles up with per-bubble physics
        for b in &mut self.bubbles {
            b.y -= b.rise_speed * dt;
            b.x += (b.y * 0.5 + b.wobble_phase).sin() * b.wobble_amp * dt;
        }

        // Remove bubbles that exited the top
        self.bubbles.retain(|b| b.y > 0.0);

        // Spawn new bubbles with varied timing
        self.spawn_timer += dt;
        if self.spawn_timer >= self.next_interval {
            self.spawn_timer = 0.0;

            // Randomize next spawn interval (0.6 – 2.5 seconds)
            self.next_interval = 0.6 + self.next_rand() * 1.9;

            // Random spawn position across the tank bottom
            let spawn_x = (self.next_rand() * (tank_w - 2.0) + 1.0).max(1.0);

            // Varied rise speed: smaller bubbles rise slower
            let size = (self.next_rand() * 4.0) as u8; // 0..3
            let rise_speed = match size {
                0 => 1.2 + self.next_rand() * 0.6, // tiny: slow
                1 => 1.8 + self.next_rand() * 0.8, // small: medium
                2 => 2.2 + self.next_rand() * 1.0, // medium: fast
                _ => 1.5 + self.next_rand() * 0.5, // degree: medium
            };

            let wobble_amp = 0.1 + self.next_rand() * 0.5;
            let wobble_phase = self.next_rand() * std::f32::consts::TAU;
            let spawn_y = tank_h - 2.0 - self.next_rand() * 2.0;

            self.bubbles.push(Bubble {
                x: spawn_x,
                y: spawn_y,
                size,
                rise_speed,
                wobble_amp,
                wobble_phase,
            });

            // Occasionally spawn a cluster of micro-bubbles
            let cluster_roll = self.next_rand();
            if cluster_roll > 0.6 {
                let cluster_x = spawn_x + self.next_rand() * 3.0 - 1.5;
                let cluster_count = 2 + (self.next_rand() * 2.0) as usize;
                for _ in 0..cluster_count {
                    let cx = (cluster_x + self.next_rand() * 2.0 - 1.0).clamp(1.0, tank_w - 1.0);
                    let cy = tank_h - 2.0 - self.next_rand() * 3.0;
                    let rs = 0.8 + self.next_rand() * 0.8;
                    let wa = 0.05 + self.next_rand() * 0.2;
                    let wp = self.next_rand() * std::f32::consts::TAU;
                    self.bubbles.push(Bubble {
                        x: cx,
                        y: cy,
                        size: 0,
                        rise_speed: rs,
                        wobble_amp: wa,
                        wobble_phase: wp,
                    });
                }
            }
        }
    }

    /// Spawn bubbles at a specific location (e.g., from creature movement).
    pub fn spawn_at(&mut self, x: f32, y: f32, count: usize) {
        for _ in 0..count {
            let offset_x = self.next_rand() * 2.0 - 1.0;
            let offset_y = self.next_rand() * 1.5;
            let bx = x + offset_x;
            let by = y - offset_y;
            let size = (self.next_rand() * 2.0) as u8;
            let rise_speed = 1.5 + self.next_rand() * 1.0;
            let wobble_amp = 0.1 + self.next_rand() * 0.3;
            let wobble_phase = self.next_rand() * std::f32::consts::TAU;
            self.bubbles.push(Bubble {
                x: bx,
                y: by,
                size,
                rise_speed,
                wobble_amp,
                wobble_phase,
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
                2 => 'O',
                _ => '°',
            };

            if let Some(cell) = buf.cell_mut((sx as u16, sy as u16)) {
                let bg = cell.bg;
                cell.set_char(ch)
                    .set_style(Style::default().fg(color).bg(bg));
            }
        }
    }
}
