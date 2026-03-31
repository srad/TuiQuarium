use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fontdue::{Font, FontSettings};
use gif::{Encoder, Frame as GifFrame, Repeat};
use ratatui::buffer::Buffer;

use crate::constants::{GIF_FONT_SIZE, GIF_FPS};
use crate::screenshot::{color_to_rgb, indexed_to_rgb};

const FONT_BYTES: &[u8] = include_bytes!("../fonts/JetBrainsMono-Regular.ttf");

/// GIF frame delay derived from [`GIF_FPS`] (in centiseconds).
const FRAME_DELAY_CS: u16 = (100 / GIF_FPS) as u16;

/// Minimum wall-clock interval between captured frames.
const FRAME_INTERVAL: Duration = Duration::from_millis(1000 / GIF_FPS as u64);

/// Streaming GIF recorder that captures ratatui [`Buffer`]s to an animated
/// GIF file. Each frame is rendered with fontdue at [`GIF_FONT_SIZE`] px
/// using the terminal 256-color palette as the GIF global color table.
pub struct GifRecorder {
    encoder: Encoder<BufWriter<File>>,
    path: PathBuf,
    start_time: Instant,
    last_capture: Instant,
    frame_count: u32,
    font: Font,
    cell_w: u32,
    cell_h: u32,
    baseline: f32,
    img_w: u16,
    img_h: u16,
    palette: [[u8; 3]; 256],
    color_cache: HashMap<[u8; 3], u8>,
}

impl GifRecorder {
    /// Begin recording. Derives image dimensions from the current `buffer`
    /// area. All subsequent frames must match these dimensions (mismatched
    /// frames are silently skipped).
    pub fn start(buffer: &Buffer, path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let font = Font::from_bytes(FONT_BYTES, FontSettings::default())
            .map_err(|e| format!("failed to load embedded font: {e}"))?;

        let metrics = font.metrics('M', GIF_FONT_SIZE);
        let cell_w = metrics.advance_width.ceil() as u32;
        let line_metrics = font
            .horizontal_line_metrics(GIF_FONT_SIZE)
            .ok_or("font has no horizontal line metrics")?;
        let cell_h =
            (line_metrics.ascent - line_metrics.descent + line_metrics.line_gap).ceil() as u32;
        let baseline = line_metrics.ascent;

        let area = buffer.area;
        let img_w = (area.width as u32 * cell_w) as u16;
        let img_h = (area.height as u32 * cell_h) as u16;

        // Build the terminal 256-color palette as flat RGB bytes for the GIF
        // global color table, and keep a structured copy for nearest-color lookup.
        let mut palette_flat = Vec::with_capacity(256 * 3);
        let mut palette = [[0u8; 3]; 256];
        for i in 0u16..256 {
            let rgb = indexed_to_rgb(i as u8);
            palette[i as usize] = rgb;
            palette_flat.extend_from_slice(&rgb);
        }

        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        let mut encoder = Encoder::new(writer, img_w, img_h, &palette_flat)?;
        encoder.set_repeat(Repeat::Infinite)?;

        let now = Instant::now();
        Ok(Self {
            encoder,
            path,
            start_time: now,
            // Set to the past so the first add_frame() captures immediately.
            last_capture: now - Duration::from_millis(200),
            frame_count: 0,
            font,
            cell_w,
            cell_h,
            baseline,
            img_w,
            img_h,
            palette,
            color_cache: HashMap::new(),
        })
    }

    /// Capture a frame from the current terminal `buffer`. Self-throttles to
    /// ~10 fps; returns `Ok(true)` when a frame was written and `Ok(false)`
    /// when the call was skipped (too soon since the last capture).
    pub fn add_frame(&mut self, buffer: &Buffer) -> Result<bool, Box<dyn std::error::Error>> {
        if self.last_capture.elapsed() < FRAME_INTERVAL {
            return Ok(false);
        }

        // Skip frames whose terminal area differs from the initial dimensions
        // (e.g. after a terminal resize).
        let area = buffer.area;
        let cur_w = (area.width as u32 * self.cell_w) as u16;
        let cur_h = (area.height as u32 * self.cell_h) as u16;
        if cur_w != self.img_w || cur_h != self.img_h {
            return Ok(false);
        }

        let pixels = self.render_indexed(buffer);

        let mut frame = GifFrame::default();
        frame.width = self.img_w;
        frame.height = self.img_h;
        frame.delay = FRAME_DELAY_CS;
        frame.buffer = Cow::Owned(pixels);
        self.encoder.write_frame(&frame)?;

        self.last_capture = Instant::now();
        self.frame_count += 1;
        Ok(true)
    }

    /// Finalize the recording. Returns the output path and total frame count.
    /// The GIF trailer is written when the encoder is dropped.
    pub fn stop(self) -> (PathBuf, u32) {
        let path = self.path;
        let count = self.frame_count;
        // Encoder::drop writes the GIF trailer and flushes the BufWriter.
        (path, count)
    }

    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Render a terminal buffer to palette-indexed pixel data.
    fn render_indexed(&mut self, buffer: &Buffer) -> Vec<u8> {
        let area = buffer.area;
        let total = self.img_w as usize * self.img_h as usize;
        let stride = self.img_w as usize;
        let mut pixels = vec![0u8; total];

        for row in 0..area.height {
            for col in 0..area.width {
                let cell = buffer.cell((area.x + col, area.y + row)).unwrap();

                let bg = color_to_rgb(cell.bg);
                let fg = color_to_rgb(cell.fg);

                let bg_idx = self.palette_index(bg);
                let ch = cell.symbol().chars().next().unwrap_or(' ');
                let origin_x = col as u32 * self.cell_w;
                let origin_y = row as u32 * self.cell_h;

                // Fill background.
                for dy in 0..self.cell_h {
                    for dx in 0..self.cell_w {
                        let px = (origin_x + dx) as usize;
                        let py = (origin_y + dy) as usize;
                        if px < stride && py < self.img_h as usize {
                            pixels[py * stride + px] = bg_idx;
                        }
                    }
                }

                if ch == ' ' {
                    continue;
                }

                // Rasterize the glyph and composite onto the cell.
                let (gm, bitmap) = self.font.rasterize(ch, GIF_FONT_SIZE);
                if bitmap.is_empty() {
                    continue;
                }

                let gx = origin_x as i32 + gm.xmin;
                let gy =
                    origin_y as i32 + (self.baseline as i32 - gm.height as i32 - gm.ymin);

                for py in 0..gm.height {
                    for px in 0..gm.width {
                        let coverage = bitmap[py * gm.width + px];
                        if coverage == 0 {
                            continue;
                        }
                        let ix = gx + px as i32;
                        let iy = gy + py as i32;
                        if ix < 0
                            || iy < 0
                            || ix >= self.img_w as i32
                            || iy >= self.img_h as i32
                        {
                            continue;
                        }
                        let blended = blend(fg, bg, coverage);
                        let idx = self.palette_index(blended);
                        pixels[iy as usize * stride + ix as usize] = idx;
                    }
                }
            }
        }

        pixels
    }

    /// Map an RGB triple to the nearest palette index, with caching.
    fn palette_index(&mut self, rgb: [u8; 3]) -> u8 {
        if let Some(&idx) = self.color_cache.get(&rgb) {
            return idx;
        }
        let idx = nearest_index(&self.palette, rgb);
        self.color_cache.insert(rgb, idx);
        idx
    }
}

/// Find the palette entry closest to `rgb` by Euclidean distance in RGB space.
fn nearest_index(palette: &[[u8; 3]; 256], rgb: [u8; 3]) -> u8 {
    let mut best = 0u8;
    let mut best_dist = u32::MAX;
    for (i, &[pr, pg, pb]) in palette.iter().enumerate() {
        let dr = rgb[0] as i32 - pr as i32;
        let dg = rgb[1] as i32 - pg as i32;
        let db = rgb[2] as i32 - pb as i32;
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best = i as u8;
            if dist == 0 {
                break;
            }
        }
    }
    best
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

/// Return the default recordings directory (`~/.tuiquarium/recordings/`),
/// creating it if necessary.
pub fn recordings_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))?
    } else {
        std::env::var("HOME")?
    };
    let dir = PathBuf::from(home)
        .join(".tuiquarium")
        .join("recordings");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    #[test]
    fn test_recordings_dir_is_created() {
        let dir = recordings_dir().unwrap();
        assert!(dir.exists());
        assert!(dir.ends_with("recordings"));
    }

    #[test]
    fn test_nearest_index_exact_match() {
        let mut palette = [[0u8; 3]; 256];
        palette[0] = [12, 12, 12]; // black
        palette[9] = [231, 72, 86]; // light red
        assert_eq!(nearest_index(&palette, [12, 12, 12]), 0);
        assert_eq!(nearest_index(&palette, [231, 72, 86]), 9);
    }

    #[test]
    fn test_nearest_index_approximate() {
        let mut palette = [[0u8; 3]; 256];
        palette[0] = [0, 0, 0];
        palette[15] = [255, 255, 255];
        // Very bright should be closer to white (index 15).
        assert_eq!(nearest_index(&palette, [250, 250, 250]), 15);
    }

    #[test]
    fn test_blend_fully_opaque() {
        assert_eq!(blend([255, 0, 0], [0, 0, 0], 255), [255, 0, 0]);
    }

    #[test]
    fn test_blend_fully_transparent() {
        assert_eq!(blend([255, 0, 0], [0, 255, 0], 0), [0, 255, 0]);
    }

    #[test]
    fn test_start_add_frame_stop_lifecycle() {
        let area = Rect::new(0, 0, 20, 5);
        let buffer = Buffer::empty(area);
        let path = std::env::temp_dir().join("tuiquarium_test_recording.gif");
        let _ = std::fs::remove_file(&path);

        let mut recorder = GifRecorder::start(&buffer, path.clone()).unwrap();
        // First frame should capture (throttle was pre-set to the past).
        let captured = recorder.add_frame(&buffer).unwrap();
        assert!(captured);

        // Immediate second call should be throttled.
        let captured = recorder.add_frame(&buffer).unwrap();
        assert!(!captured);

        assert_eq!(recorder.frame_count(), 1);

        let (out_path, count) = recorder.stop();
        assert_eq!(count, 1);
        assert!(out_path.exists());
        assert!(std::fs::metadata(&out_path).unwrap().len() > 0);
        let _ = std::fs::remove_file(&out_path);
    }
}
