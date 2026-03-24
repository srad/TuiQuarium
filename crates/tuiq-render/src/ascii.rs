//! Genome → multi-line ASCII art procedural generator.
//! Takes a CreatureGenome and produces animation frames for each action.

use tuiq_core::components::AsciiFrame;
use tuiq_core::genome::*;

/// Generate all animation frames from a genome.
/// Returns (swim_frames, idle_frames).
pub fn generate_frames(genome: &CreatureGenome) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let art = &genome.art;
    let base = build_base_body(art);
    let swim = generate_swim_frames(&base, art, &genome.anim);
    let idle = generate_idle_frames(&base, art, &genome.anim);
    (swim, idle)
}

/// A mutable 2D character grid for building ASCII art.
struct CharGrid {
    cells: Vec<Vec<char>>,
    width: usize,
    height: usize,
}

impl CharGrid {
    fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![vec![' '; width]; height],
            width,
            height,
        }
    }

    fn set(&mut self, x: usize, y: usize, ch: char) {
        if y < self.height && x < self.width {
            self.cells[y][x] = ch;
        }
    }

    fn set_str(&mut self, x: usize, y: usize, s: &str) {
        for (i, ch) in s.chars().enumerate() {
            self.set(x + i, y, ch);
        }
    }

    fn to_frame(&self) -> AsciiFrame {
        let rows: Vec<String> = self.cells.iter().map(|row| row.iter().collect()).collect();
        AsciiFrame::from_rows(rows.iter().map(|s| s.as_str()).collect())
    }
}

/// The base body without tail animation.
#[allow(dead_code)]
struct BaseBody {
    width: usize,
    height: usize,
    /// Outline and interior (no tail yet).
    body_grid: CharGrid,
    /// X position where the tail attaches.
    tail_x: usize,
    /// Y positions for tail attachment (mid-row range).
    tail_y_start: usize,
    tail_y_end: usize,
    /// Where dorsal fin sits.
    dorsal_x: usize,
    dorsal_y: usize,
    /// Where pectoral fins sit.
    pectoral_x: usize,
    pectoral_y_top: usize,
    pectoral_y_bot: usize,
}

fn build_base_body(art: &ArtGenome) -> BaseBody {
    // Scale dimensions from body_size
    let scale = art.body_size;

    let (body_w, body_h) = match art.body_plan {
        BodyPlan::Slim => ((10.0 * scale) as usize, (3.0 * scale).max(3.0) as usize),
        BodyPlan::Round => ((8.0 * scale) as usize, (5.0 * scale).max(4.0) as usize),
        BodyPlan::Flat => ((12.0 * scale) as usize, (3.0 * scale).max(3.0) as usize),
        BodyPlan::Tall => ((6.0 * scale) as usize, (7.0 * scale).max(5.0) as usize),
    };

    let body_w = body_w.max(5);
    let body_h = body_h.max(3);

    // Total width includes space for fins and tail
    let fin_space = if art.has_dorsal_fin || art.has_pectoral_fins { 2 } else { 0 };
    let tail_w = (3.0 * art.tail_length) as usize;
    let total_w = body_w + tail_w + fin_space + 2; // +2 for eye and outline
    let total_h = body_h + if art.has_dorsal_fin { 1 } else { 0 }
        + if art.has_pectoral_fins { 0 } else { 0 };
    let total_h = total_h.max(3);

    let mut grid = CharGrid::new(total_w, total_h);

    let eye_char = art.eye_style.char();
    let fill_char = match art.fill_pattern {
        FillPattern::Solid => '~',
        FillPattern::Striped => '=',
        FillPattern::Spotted => ':',
        FillPattern::Scales => '#',
    };

    // Draw body outline based on plan
    let body_start_y = if art.has_dorsal_fin { 1 } else { 0 };
    let mid_y = body_start_y + body_h / 2;

    match art.body_plan {
        BodyPlan::Slim | BodyPlan::Flat => {
            // Top outline
            grid.set_str(1, body_start_y, &format!(",{}.", "-".repeat(body_w - 2)));
            // Middle rows with fill
            for y in (body_start_y + 1)..(body_start_y + body_h - 1) {
                grid.set(0, y, if y == mid_y { '(' } else { '|' });
                // Eye on the first interior row
                if y == body_start_y + 1 {
                    grid.set(1, y, ' ');
                    grid.set(2, y, eye_char);
                    for x in 3..body_w {
                        grid.set(x, y, if (x + y) % 3 == 0 { fill_char } else { ' ' });
                    }
                } else {
                    for x in 1..body_w {
                        grid.set(x, y, if (x + y) % 3 == 0 { fill_char } else { ' ' });
                    }
                }
                grid.set(body_w, y, if y == mid_y { ')' } else { '|' });
            }
            // Bottom outline
            grid.set_str(1, body_start_y + body_h - 1, &format!("'{}´", "-".repeat(body_w - 2)));
        }
        BodyPlan::Round => {
            // Rounded body with / \ outlines
            let half = body_h / 2;
            for y in 0..body_h {
                let row_y = body_start_y + y;
                let indent = if y <= half {
                    half - y
                } else {
                    y - half
                };
                let row_w = body_w.saturating_sub(indent * 2);
                if row_w < 2 {
                    continue;
                }
                let start_x = indent + 1;
                if y == 0 || y == body_h - 1 {
                    // Top/bottom edge
                    grid.set(start_x, row_y, if y == 0 { ',' } else { '\'' });
                    for x in 1..row_w - 1 {
                        grid.set(start_x + x, row_y, if y == 0 { '-' } else { '-' });
                    }
                    grid.set(start_x + row_w - 1, row_y, if y == 0 { '.' } else { '´' });
                } else {
                    grid.set(start_x, row_y, if y < half { '/' } else { '\\' });
                    // Interior
                    if y == 1 + (half.saturating_sub(1)).min(1) {
                        grid.set(start_x + 1, row_y, ' ');
                        grid.set(start_x + 2, row_y, eye_char);
                        for x in 3..row_w - 1 {
                            grid.set(start_x + x, row_y, if (x + y) % 3 == 0 { fill_char } else { ' ' });
                        }
                    } else {
                        for x in 1..row_w - 1 {
                            grid.set(start_x + x, row_y, if (x + y) % 3 == 0 { fill_char } else { ' ' });
                        }
                    }
                    grid.set(start_x + row_w - 1, row_y, if y < half { '\\' } else { '/' });
                }
            }
        }
        BodyPlan::Tall => {
            // Tall angelfish-like body
            let half = body_h / 2;
            for y in 0..body_h {
                let row_y = body_start_y + y;
                let dy = if y <= half { y } else { body_h - 1 - y };
                let row_w = (2 + dy * 2).min(body_w);
                let start_x = (body_w / 2).saturating_sub(row_w / 2) + 1;
                if dy == 0 {
                    grid.set(start_x + row_w / 2, row_y, if y == 0 { '/' } else { '\\' });
                    if row_w > 1 {
                        grid.set(start_x + row_w / 2 + 1, row_y, if y == 0 { '\\' } else { '/' });
                    }
                } else {
                    grid.set(start_x, row_y, if y <= half { '/' } else { '\\' });
                    if y == half.min(2) {
                        grid.set(start_x + 1, row_y, ' ');
                        grid.set(start_x + 2, row_y, eye_char);
                        for x in 3..row_w - 1 {
                            grid.set(start_x + x, row_y, if (x + y) % 2 == 0 { fill_char } else { ' ' });
                        }
                    } else {
                        for x in 1..row_w - 1 {
                            grid.set(start_x + x, row_y, if (x + y) % 2 == 0 { fill_char } else { ' ' });
                        }
                    }
                    grid.set(start_x + row_w - 1, row_y, if y <= half { '\\' } else { '/' });
                }
            }
        }
    }

    // Dorsal fin
    let dorsal_x = body_w / 3 + 1;
    let dorsal_y = 0;
    if art.has_dorsal_fin && body_start_y > 0 {
        grid.set(dorsal_x, dorsal_y, '/');
        grid.set(dorsal_x + 1, dorsal_y, '\\');
    }

    BaseBody {
        width: total_w,
        height: total_h,
        body_grid: grid,
        tail_x: body_w + 1,
        tail_y_start: body_start_y + body_h / 2 - 1,
        tail_y_end: body_start_y + body_h / 2 + 1,
        dorsal_x,
        dorsal_y,
        pectoral_x: body_w / 4,
        pectoral_y_top: body_start_y + 1,
        pectoral_y_bot: body_start_y + body_h - 2,
    }
}

fn generate_swim_frames(base: &BaseBody, art: &ArtGenome, anim: &AnimGenome) -> Vec<AsciiFrame> {
    let num_frames = 3;
    let tail_w = (3.0 * art.tail_length) as usize;

    (0..num_frames)
        .map(|i| {
            let mut grid = CharGrid::new(base.width, base.height);
            // Copy base body
            for y in 0..base.body_grid.height.min(grid.height) {
                for x in 0..base.body_grid.width.min(grid.width) {
                    grid.cells[y][x] = base.body_grid.cells[y][x];
                }
            }

            // Draw tail with animation offset
            let mid_y = (base.tail_y_start + base.tail_y_end) / 2;
            let tail_chars = match art.tail_style {
                TailStyle::Forked => vec!['=', '=', '<'],
                TailStyle::Fan => vec!['=', '>', '>'],
                TailStyle::Pointed => vec!['-', '-', '>'],
                TailStyle::Flowing => vec!['~', '~', '>'],
            };

            // Animate tail vertical position
            let y_offset: i32 = match i {
                0 => 0,
                1 => if anim.tail_amplitude > 0.5 { -1 } else { 0 },
                2 => if anim.tail_amplitude > 0.5 { 1 } else { 0 },
                _ => 0,
            };

            for t in 0..tail_w.min(tail_chars.len()) {
                let tx = base.tail_x + t;
                let ty = (mid_y as i32 + y_offset).max(0) as usize;
                if tx < grid.width && ty < grid.height {
                    grid.set(tx, ty, tail_chars[t % tail_chars.len()]);
                }
            }

            // Extra tail lines for forked/fan
            if matches!(art.tail_style, TailStyle::Forked | TailStyle::Fan) && tail_w > 1 {
                let ty_up = ((mid_y as i32 - 1 + y_offset).max(0)) as usize;
                let ty_dn = ((mid_y as i32 + 1 + y_offset).max(0)) as usize;
                if base.tail_x < grid.width {
                    if ty_up < grid.height {
                        grid.set(base.tail_x + tail_w.saturating_sub(1), ty_up, '/');
                    }
                    if ty_dn < grid.height {
                        grid.set(base.tail_x + tail_w.saturating_sub(1), ty_dn, '\\');
                    }
                }
            }

            grid.to_frame()
        })
        .collect()
}

fn generate_idle_frames(base: &BaseBody, art: &ArtGenome, anim: &AnimGenome) -> Vec<AsciiFrame> {
    let num_frames = 2;
    let tail_w = (3.0 * art.tail_length) as usize;

    (0..num_frames)
        .map(|i| {
            let x_shift = if i == 1 && anim.idle_sway > 0.3 { 1 } else { 0 };
            let new_w = base.width + x_shift;
            let mut grid = CharGrid::new(new_w, base.height);

            for y in 0..base.body_grid.height.min(grid.height) {
                for x in 0..base.body_grid.width.min(grid.width) {
                    let nx = x + x_shift;
                    if nx < grid.width {
                        grid.cells[y][nx] = base.body_grid.cells[y][x];
                    }
                }
            }

            // Static tail for idle
            let mid_y = (base.tail_y_start + base.tail_y_end) / 2;
            let tail_char = match art.tail_style {
                TailStyle::Forked => '=',
                TailStyle::Fan => '>',
                TailStyle::Pointed => '>',
                TailStyle::Flowing => '~',
            };
            for t in 0..tail_w {
                let tx = base.tail_x + x_shift + t;
                if tx < grid.width && mid_y < grid.height {
                    grid.set(tx, mid_y, if t == tail_w - 1 { '>' } else { tail_char });
                }
            }

            grid.to_frame()
        })
        .collect()
}

/// Map a genome's primary_color to a color index for the renderer.
pub fn genome_color_index(genome: &CreatureGenome) -> u8 {
    genome.art.primary_color % 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_produces_non_empty_frames() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let genome = CreatureGenome::random(&mut rng);
            let (swim, idle) = generate_frames(&genome);
            assert!(!swim.is_empty(), "Should have swim frames");
            assert!(!idle.is_empty(), "Should have idle frames");
            for f in &swim {
                assert!(f.width > 0 && f.height > 0, "Swim frame has zero dimensions");
                assert!(
                    f.rows.iter().any(|r| r.chars().any(|c| c != ' ')),
                    "Swim frame is all spaces"
                );
            }
            for f in &idle {
                assert!(f.width > 0 && f.height > 0, "Idle frame has zero dimensions");
            }
        }
    }

    #[test]
    fn test_different_body_plans_produce_different_art() {
        let mut rng = rand::rng();
        let mut base = CreatureGenome::random(&mut rng);
        base.art.body_size = 1.0;

        let mut results = Vec::new();
        for plan in BodyPlan::ALL {
            base.art.body_plan = plan;
            let (swim, _) = generate_frames(&base);
            let signature: String = swim[0].rows.concat();
            results.push((plan, signature));
        }

        // At least some body plans should produce different art
        let unique: std::collections::HashSet<&String> =
            results.iter().map(|(_, s)| s).collect();
        assert!(
            unique.len() >= 2,
            "Different body plans should produce at least some different art"
        );
    }
}
