use std::path::Path;

use fontdue::{Font, FontSettings};
use ratatui::buffer::Buffer;
use ratatui::style::Color;

use crate::constants::SCREENSHOT_FONT_SIZE;

const FONT_BYTES: &[u8] = include_bytes!("../fonts/JetBrainsMono-Regular.ttf");

/// Convert a ratatui [`Buffer`] to a PNG file at `path`.
///
/// Each terminal cell is rendered with JetBrains Mono at [`SCREENSHOT_FONT_SIZE`] px.
/// The cell dimensions are derived from the font metrics so the output
/// faithfully reflects what a monospace terminal would show.
pub fn save_buffer_as_png(buffer: &Buffer, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let font = Font::from_bytes(FONT_BYTES, FontSettings::default())
        .map_err(|e| format!("failed to load embedded font: {e}"))?;

    // Derive cell size from the font metrics of a reference character.
    let metrics = font.metrics('M', SCREENSHOT_FONT_SIZE);
    let cell_w = metrics.advance_width.ceil() as u32;
    let line_metrics = font
        .horizontal_line_metrics(SCREENSHOT_FONT_SIZE)
        .ok_or("font has no horizontal line metrics")?;
    let cell_h = (line_metrics.ascent - line_metrics.descent + line_metrics.line_gap)
        .ceil() as u32;
    let baseline = line_metrics.ascent;

    let area = buffer.area;
    let img_w = area.width as u32 * cell_w;
    let img_h = area.height as u32 * cell_h;

    let mut img = image::RgbImage::new(img_w, img_h);

    for row in 0..area.height {
        for col in 0..area.width {
            let cell = buffer.cell((area.x + col, area.y + row)).unwrap();

            let bg = color_to_rgb(cell.bg);
            let fg = color_to_rgb(cell.fg);

            let ch = cell.symbol().chars().next().unwrap_or(' ');
            let origin_x = col as u32 * cell_w;
            let origin_y = row as u32 * cell_h;

            // Fill background.
            for dy in 0..cell_h {
                for dx in 0..cell_w {
                    img.put_pixel(origin_x + dx, origin_y + dy, image::Rgb(bg));
                }
            }

            if ch == ' ' {
                continue;
            }

            // Rasterize the glyph and composite it onto the cell.
            let (gm, bitmap) = font.rasterize(ch, SCREENSHOT_FONT_SIZE);
            if bitmap.is_empty() {
                continue;
            }

            // Position the glyph relative to the cell's baseline.
            let gx = origin_x as i32 + gm.xmin;
            let gy = origin_y as i32 + (baseline as i32 - gm.height as i32 - gm.ymin);

            for py in 0..gm.height {
                for px in 0..gm.width {
                    let coverage = bitmap[py * gm.width + px];
                    if coverage == 0 {
                        continue;
                    }
                    let ix = gx + px as i32;
                    let iy = gy + py as i32;
                    if ix < 0 || iy < 0 || ix >= img_w as i32 || iy >= img_h as i32 {
                        continue;
                    }
                    let dst = img.get_pixel(ix as u32, iy as u32).0;
                    let blended = blend(fg, dst, coverage);
                    img.put_pixel(ix as u32, iy as u32, image::Rgb(blended));
                }
            }
        }
    }

    img.save(path)?;
    Ok(())
}

/// Alpha-blend `fg` over `bg` using `alpha` (0–255 coverage).
fn blend(fg: [u8; 3], bg: [u8; 3], alpha: u8) -> [u8; 3] {
    let a = alpha as u16;
    let inv = 255 - a;
    [
        ((fg[0] as u16 * a + bg[0] as u16 * inv) / 255) as u8,
        ((fg[1] as u16 * a + bg[1] as u16 * inv) / 255) as u8,
        ((fg[2] as u16 * a + bg[2] as u16 * inv) / 255) as u8,
    ]
}

/// Return the default screenshots directory (`~/.tuiquarium/screenshots/`),
/// creating it if necessary.
pub fn screenshots_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?
    } else {
        std::env::var("HOME")?
    };
    let dir = std::path::PathBuf::from(home)
        .join(".tuiquarium")
        .join("screenshots");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Convert a ratatui [`Color`] to an RGB byte triple.
pub(crate) fn color_to_rgb(color: Color) -> [u8; 3] {
    match color {
        Color::Reset => [12, 12, 12],
        Color::Black => [12, 12, 12],
        Color::Red => [197, 15, 31],
        Color::Green => [19, 161, 14],
        Color::Yellow => [193, 156, 0],
        Color::Blue => [0, 55, 218],
        Color::Magenta => [136, 23, 152],
        Color::Cyan => [58, 150, 221],
        Color::Gray => [204, 204, 204],
        Color::DarkGray => [118, 118, 118],
        Color::LightRed => [231, 72, 86],
        Color::LightGreen => [22, 198, 12],
        Color::LightYellow => [249, 241, 165],
        Color::LightBlue => [59, 120, 255],
        Color::LightMagenta => [180, 0, 158],
        Color::LightCyan => [97, 214, 214],
        Color::White => [242, 242, 242],
        Color::Rgb(r, g, b) => [r, g, b],
        Color::Indexed(i) => indexed_to_rgb(i),
    }
}

/// Map a 256-color palette index to RGB.
pub(crate) fn indexed_to_rgb(i: u8) -> [u8; 3] {
    match i {
        0 => [12, 12, 12],
        1 => [197, 15, 31],
        2 => [19, 161, 14],
        3 => [193, 156, 0],
        4 => [0, 55, 218],
        5 => [136, 23, 152],
        6 => [58, 150, 221],
        7 => [204, 204, 204],
        8 => [118, 118, 118],
        9 => [231, 72, 86],
        10 => [22, 198, 12],
        11 => [249, 241, 165],
        12 => [59, 120, 255],
        13 => [180, 0, 158],
        14 => [97, 214, 214],
        15 => [242, 242, 242],
        // 6×6×6 colour cube (indices 16–231).
        16..=231 => {
            let v = i - 16;
            let r = v / 36;
            let g = (v / 6) % 6;
            let b = v % 6;
            let to_byte = |c: u8| if c == 0 { 0 } else { 55 + c * 40 };
            [to_byte(r), to_byte(g), to_byte(b)]
        }
        // Grayscale ramp (indices 232–255).
        232..=255 => {
            let v = 8 + (i - 232) * 10;
            [v, v, v]
        }
    }
}
