//! Genome → ASCII art procedural generator.
//! Complexity-gated: simple cells are 1-2 chars, elaborate organisms are multi-line.
//! No animal assumptions — body shapes, appendages, patterns all emerge from continuous genome values.

use tuiq_core::components::AsciiFrame;
use tuiq_core::genome::*;

/// Generate all animation frames from a genome.
/// Returns (swim_frames, idle_frames).
pub fn generate_frames(genome: &CreatureGenome) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let complexity = genome.complexity;

    if complexity < 0.15 {
        // Single character cell
        generate_cell_frames(genome)
    } else if complexity < 0.35 {
        // Simple 2-4 char body
        generate_simple_frames(genome)
    } else if complexity < 0.6 {
        // Multi-cell body with optional protrusions
        generate_medium_frames(genome)
    } else {
        // Full multi-line body with appendages and detail
        generate_complex_frames(genome)
    }
}

/// Map a genome's primary_hue to a color index for the renderer.
pub fn genome_color_index(genome: &CreatureGenome) -> u8 {
    genome.art.color_index()
}

// ── Cell: complexity 0.0–0.15 ────────────────────────────────

fn generate_cell_frames(genome: &CreatureGenome) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let chars = ['.', 'o', '*', 'O', '+', '@'];
    let idx = (genome.art.primary_hue * chars.len() as f32).floor() as usize;
    let ch = chars[idx.min(chars.len() - 1)];
    let s = String::from(ch);

    let swim = vec![AsciiFrame::from_rows(vec![&s])];
    let idle = vec![AsciiFrame::from_rows(vec![&s])];
    (swim, idle)
}

// ── Simple: complexity 0.15–0.35 ─────────────────────────────

fn generate_simple_frames(genome: &CreatureGenome) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let art = &genome.art;
    let eye = if art.eye_size > 0.2 { art.eye_char() } else { ' ' };

    // Body width from elongation: round=2, elongated=4
    let body_w = (2.0 + 2.0 * art.body_elongation) as usize;

    let body: String = match body_w {
        0..=2 => {
            if eye != ' ' {
                format!("({eye})")
            } else {
                "()".to_string()
            }
        }
        3 => {
            if eye != ' ' {
                format!("({eye} )")
            } else {
                "(~ )".to_string()
            }
        }
        _ => {
            let inner_w = body_w - 2;
            let mut inner = String::with_capacity(inner_w);
            if eye != ' ' {
                inner.push(eye);
                for i in 1..inner_w {
                    inner.push(if art.pattern_density > 0.3 && i % 2 == 0 { '~' } else { ' ' });
                }
            } else {
                for i in 0..inner_w {
                    inner.push(if art.pattern_density > 0.3 && i % 2 == 0 { '~' } else { ' ' });
                }
            }
            format!("({inner})")
        }
    };

    // Optional rear protrusion
    let tail = if art.tail_length > 0.1 && genome.complexity > 0.2 {
        if art.tail_fork > 0.5 { "=" } else { "-" }
    } else {
        ""
    };

    let frame1 = format!("{body}{tail}");

    // Animation: slight body variation
    let frame2 = if art.tail_length > 0.1 && genome.complexity > 0.2 {
        let tail2 = if art.tail_fork > 0.5 { ">" } else { "~" };
        format!("{body}{tail2}")
    } else {
        frame1.clone()
    };

    let swim = vec![
        AsciiFrame::from_rows(vec![&frame1]),
        AsciiFrame::from_rows(vec![&frame2]),
    ];
    let idle = vec![AsciiFrame::from_rows(vec![&frame1])];
    (swim, idle)
}

// ── Medium: complexity 0.35–0.6 ──────────────────────────────

fn generate_medium_frames(genome: &CreatureGenome) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let art = &genome.art;
    let scale = art.body_size.max(0.3);

    let body_w = ((4.0 + 4.0 * art.body_elongation) * scale).max(3.0) as usize;
    let body_h = ((2.0 + 2.0 * art.body_height_ratio) * scale).max(2.0) as usize;
    let body_h = body_h.min(4);
    let body_w = body_w.min(10);

    let eye = art.eye_char();
    let fill = if art.pattern_density > 0.4 { '~' }
        else if art.pattern_density > 0.2 { ':' }
        else { ' ' };

    let has_top = genome.complexity > 0.3 && art.top_appendage > 0.2;
    let has_sides = genome.complexity > 0.3 && art.side_appendages > 0.2;
    let tail_len = if genome.complexity > 0.2 && art.tail_length > 0.1 {
        (art.tail_length * 2.0).ceil() as usize
    } else {
        0
    };

    let total_w = body_w + tail_len + if has_sides { 2 } else { 0 };

    let side_offset = if has_sides { 1 } else { 0 };

    let build_frame = |tail_variant: usize| -> AsciiFrame {
        let mut rows: Vec<String> = Vec::new();

        // Top appendage row
        if has_top {
            let mut row = vec![' '; total_w];
            let apex = side_offset + body_w / 3;
            if apex < total_w {
                row[apex] = if art.top_appendage > 0.6 { '^' } else { '/' };
                if apex + 1 < total_w {
                    row[apex + 1] = if art.top_appendage > 0.6 { '^' } else { '\\' };
                }
            }
            rows.push(row.into_iter().collect());
        }

        // Body rows
        for y in 0..body_h {
            let mut row = vec![' '; total_w];
            let is_top = y == 0;
            let is_bot = y == body_h - 1;
            let is_mid = !is_top && !is_bot;

            // Side appendages
            if has_sides && is_mid {
                row[0] = if art.side_appendages > 0.6 { '{' } else { '<' };
                let _right = side_offset + body_w + tail_len;
            }

            // Body outline
            if is_top {
                row[side_offset] = ',';
                for x in 1..body_w - 1 {
                    row[side_offset + x] = '-';
                }
                row[side_offset + body_w - 1] = '.';
            } else if is_bot {
                row[side_offset] = '\'';
                for x in 1..body_w - 1 {
                    row[side_offset + x] = '-';
                }
                row[side_offset + body_w - 1] = '\'';
            } else {
                // Interior
                row[side_offset] = '|';
                if y == 1 && art.eye_size > 0.1 {
                    row[side_offset + 1] = eye;
                    for x in 2..body_w - 1 {
                        row[side_offset + x] = if (x + y) % 3 == 0 && art.pattern_density > 0.2 { fill } else { ' ' };
                    }
                } else {
                    for x in 1..body_w - 1 {
                        row[side_offset + x] = if (x + y) % 3 == 0 && art.pattern_density > 0.2 { fill } else { ' ' };
                    }
                }
                row[side_offset + body_w - 1] = '|';
            }

            // Tail
            if tail_len > 0 && is_mid {
                let tail_start = side_offset + body_w;
                for t in 0..tail_len {
                    if tail_start + t < total_w {
                        let ch = match tail_variant {
                            0 => if art.tail_fork > 0.5 { '=' } else { '-' },
                            1 => '~',
                            _ => if t == tail_len - 1 { '>' } else { '-' },
                        };
                        row[tail_start + t] = ch;
                    }
                }
            }

            rows.push(row.into_iter().collect());
        }

        AsciiFrame::from_rows(rows.iter().map(|s| s.as_str()).collect())
    };

    let swim = vec![build_frame(0), build_frame(1), build_frame(2)];
    let idle = vec![build_frame(0)];
    (swim, idle)
}

// ── Complex: complexity 0.6+ ─────────────────────────────────

fn generate_complex_frames(genome: &CreatureGenome) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let art = &genome.art;
    let scale = art.body_size.max(0.4);

    let body_w = ((5.0 + 6.0 * art.body_elongation) * scale).max(4.0) as usize;
    let body_h = ((3.0 + 4.0 * art.body_height_ratio) * scale).max(3.0) as usize;
    let body_h = body_h.min(7);
    let body_w = body_w.min(14);

    let eye = art.eye_char();
    let fill = if art.pattern_density > 0.6 { '#' }
        else if art.pattern_density > 0.3 { '~' }
        else if art.pattern_density > 0.1 { ':' }
        else { ' ' };

    let has_top = art.top_appendage > 0.2;
    let has_sides = art.side_appendages > 0.2;
    let tail_len = if art.tail_length > 0.1 {
        (art.tail_length * 3.0).ceil() as usize
    } else {
        0
    };

    let top_h = if has_top { (art.top_appendage * 2.0).ceil() as usize } else { 0 };
    let total_w = body_w + tail_len + if has_sides { 2 } else { 0 } + 1;

    let side_offset = if has_sides { 1 } else { 0 };

    let build_frame = |tail_variant: usize| -> AsciiFrame {
        let mut rows: Vec<String> = Vec::new();

        // Top appendage rows
        for ty in 0..top_h {
            let mut row = vec![' '; total_w];
            let apex = side_offset + body_w / 3;
            let spread = (top_h - ty) as usize;
            if ty == 0 {
                // Tip
                if apex < total_w { row[apex] = '/'; }
                if apex + 1 < total_w { row[apex + 1] = '\\'; }
            } else {
                // Wider base
                let left = apex.saturating_sub(spread);
                let right = (apex + 1 + spread).min(total_w - 1);
                if left < total_w { row[left] = '/'; }
                for x in (left + 1)..right {
                    if x < total_w {
                        row[x] = if art.pattern_density > 0.4 { '|' } else { ' ' };
                    }
                }
                if right < total_w { row[right] = '\\'; }
            }
            rows.push(row.into_iter().collect());
        }

        // Body rows
        let mid_y = body_h / 2;
        for y in 0..body_h {
            let mut row = vec![' '; total_w];
            let is_top = y == 0;
            let is_bot = y == body_h - 1;

            // Rounded body shape
            let indent = if art.body_elongation < 0.4 {
                // Rounder bodies taper at top/bottom
                let dy = if y <= mid_y { mid_y - y } else { y - mid_y };
                dy / 2
            } else {
                0
            };

            let row_start = side_offset + indent;
            let row_w = body_w.saturating_sub(indent * 2);
            if row_w < 2 { continue; }

            // Side appendages
            if has_sides && y > 0 && y < body_h - 1 && y == mid_y {
                if row_start > 0 {
                    row[row_start - 1] = if art.side_appendages > 0.6 { '{' } else { '<' };
                }
            }

            if is_top {
                row[row_start] = ',';
                for x in 1..row_w - 1 {
                    row[row_start + x] = '-';
                }
                if row_start + row_w - 1 < total_w {
                    row[row_start + row_w - 1] = '.';
                }
            } else if is_bot {
                row[row_start] = '\'';
                for x in 1..row_w - 1 {
                    row[row_start + x] = '-';
                }
                if row_start + row_w - 1 < total_w {
                    row[row_start + row_w - 1] = '\'';
                }
            } else {
                // Interior with eye and fill
                let left_ch = if art.body_elongation < 0.4 {
                    if y < mid_y { '/' } else { '\\' }
                } else {
                    '|'
                };
                let right_ch = if art.body_elongation < 0.4 {
                    if y < mid_y { '\\' } else { '/' }
                } else {
                    '|'
                };

                row[row_start] = left_ch;
                if y == 1 && art.eye_size > 0.1 {
                    // Eye row
                    if row_start + 1 < total_w { row[row_start + 1] = ' '; }
                    if row_start + 2 < total_w { row[row_start + 2] = eye; }
                    for x in 3..row_w - 1 {
                        if row_start + x < total_w {
                            row[row_start + x] = if (x + y) % 3 == 0 { fill } else { ' ' };
                        }
                    }
                } else {
                    for x in 1..row_w - 1 {
                        if row_start + x < total_w {
                            row[row_start + x] = if (x + y) % 3 == 0 { fill } else { ' ' };
                        }
                    }
                }
                if row_start + row_w - 1 < total_w {
                    row[row_start + row_w - 1] = right_ch;
                }
            }

            // Tail
            if tail_len > 0 && !is_top && !is_bot {
                let tail_y_center = mid_y;
                let tail_start = side_offset + body_w;

                if y == tail_y_center {
                    // Main tail line
                    for t in 0..tail_len {
                        if tail_start + t < total_w {
                            let ch = match tail_variant {
                                0 => if art.tail_fork > 0.5 { '=' } else { '-' },
                                1 => '~',
                                _ => if t == tail_len - 1 { '>' } else { '-' },
                            };
                            row[tail_start + t] = ch;
                        }
                    }
                } else if art.tail_fork > 0.5 && tail_len > 1 {
                    // Forked tail arms
                    let dy = (y as i32 - tail_y_center as i32).unsigned_abs() as usize;
                    if dy == 1 {
                        let fork_x = tail_start + tail_len - 1;
                        if fork_x < total_w {
                            row[fork_x] = if y < tail_y_center { '/' } else { '\\' };
                        }
                    }
                }
            }

            rows.push(row.into_iter().collect());
        }

        AsciiFrame::from_rows(rows.iter().map(|s| s.as_str()).collect())
    };

    let swim = vec![build_frame(0), build_frame(1), build_frame(2)];
    let idle = vec![build_frame(0), build_frame(2)];
    (swim, idle)
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
    fn test_cell_is_single_char() {
        let mut rng = rand::rng();
        for _ in 0..50 {
            let mut g = CreatureGenome::minimal_cell(&mut rng);
            g.complexity = 0.05; // Force cell tier
            let (swim, _) = generate_frames(&g);
            assert_eq!(swim[0].width, 1, "Cell should be 1 char wide");
            assert_eq!(swim[0].height, 1, "Cell should be 1 char tall");
        }
    }

    #[test]
    fn test_complex_creatures_are_larger() {
        let mut rng = rand::rng();
        for _ in 0..50 {
            let mut g = CreatureGenome::random(&mut rng);
            g.complexity = 0.8;
            g.art.body_size = 1.5;
            let (swim, _) = generate_frames(&g);
            let total_chars: usize = swim[0].rows.iter().map(|r| r.chars().filter(|c| *c != ' ').count()).sum();
            assert!(total_chars > 3, "Complex creature should have more than 3 visible chars, got {}", total_chars);
        }
    }

    #[test]
    fn test_different_complexities_produce_different_sizes() {
        let mut rng = rand::rng();
        let mut g = CreatureGenome::random(&mut rng);
        g.art.body_size = 1.0;

        g.complexity = 0.05;
        let (cell_frames, _) = generate_frames(&g);

        g.complexity = 0.5;
        let (medium_frames, _) = generate_frames(&g);

        g.complexity = 0.9;
        let (complex_frames, _) = generate_frames(&g);

        let cell_area = cell_frames[0].width * cell_frames[0].height;
        let medium_area = medium_frames[0].width * medium_frames[0].height;
        let complex_area = complex_frames[0].width * complex_frames[0].height;

        assert!(medium_area > cell_area, "Medium should be bigger than cell");
        assert!(complex_area > cell_area, "Complex should be bigger than cell");
    }
}
