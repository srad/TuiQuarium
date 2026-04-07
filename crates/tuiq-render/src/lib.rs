pub mod ascii;
pub mod constants;
pub mod effects;
pub mod gif_recorder;
pub mod hud;
pub mod palette;
pub mod screenshot;
pub mod tank;

use std::path::Path;
use std::time::Instant;

use constants::FLASH_DURATION_SECS;
use effects::BubbleSystem;
use gif_recorder::GifRecorder;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::Frame;
use tuiq_core::Simulation;

/// Abstraction over rendering. Could be swapped for a test renderer.
pub trait Renderer {
    /// Render one frame from the current simulation state.
    fn render(&mut self, frame: &mut Frame, sim: &dyn Simulation);

    /// Save the most recently rendered frame as a PNG file at `path`.
    ///
    /// Implementations that do not support screenshots should return an
    /// appropriate error rather than panic.
    fn save_screenshot(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>>;

    /// Show a transient message for a few seconds. The renderer decides
    /// how to display it (popup, toast, status bar, etc.) and when to
    /// expire it.
    fn flash_message(&mut self, message: String);
}

/// Display parameters for the HUD.
pub struct DisplayState {
    pub paused: bool,
    pub speed: f32,
    pub show_diagnostics: bool,
    pub show_help: bool,
    pub is_recording: bool,
    pub recording_secs: u32,
    pub theme: RenderTheme,
}

/// Rendering theme: controls background colors and overall look.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderTheme {
    /// Classic dark look — fg-only colors on the terminal default background.
    Classic,
    /// Ocean look — colored backgrounds (blue water, brown substrate).
    Ocean,
    /// Deep sea — near-black water, bioluminescent neon creatures.
    DeepSea,
    /// Coral reef — warm turquoise water, coral-sand substrate.
    CoralReef,
    /// Brackish — murky green-brown water, dark mud substrate.
    Brackish,
    /// Retro CRT — green phosphor on black, all entities in shades of green.
    RetroCrt,
    /// Blueprint — white/cyan on dark blue, technical schematic feel.
    Blueprint,
    /// Frozen — pale icy blue water, white substrate.
    Frozen,
}

impl RenderTheme {
    /// Cycle to the next theme.
    pub fn next(self) -> Self {
        match self {
            Self::Classic => Self::Ocean,
            Self::Ocean => Self::DeepSea,
            Self::DeepSea => Self::CoralReef,
            Self::CoralReef => Self::Brackish,
            Self::Brackish => Self::RetroCrt,
            Self::RetroCrt => Self::Blueprint,
            Self::Blueprint => Self::Frozen,
            Self::Frozen => Self::Classic,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Classic => "Classic",
            Self::Ocean => "Ocean",
            Self::DeepSea => "Deep Sea",
            Self::CoralReef => "Coral Reef",
            Self::Brackish => "Brackish",
            Self::RetroCrt => "Retro CRT",
            Self::Blueprint => "Blueprint",
            Self::Frozen => "Frozen",
        }
    }

    /// Whether this theme overrides creature colors (monochrome themes).
    pub fn creature_color_override(self) -> Option<fn(u8) -> ratatui::style::Color> {
        match self {
            Self::RetroCrt => Some(creature_color_retro),
            Self::Blueprint => Some(creature_color_blueprint),
            _ => None,
        }
    }
}

use ratatui::style::Color;

fn creature_color_retro(color_index: u8) -> Color {
    // Shades of green phosphor
    match color_index % 6 {
        0 => Color::Rgb(0, 255, 0),
        1 => Color::Rgb(0, 200, 0),
        2 => Color::Rgb(0, 160, 0),
        3 => Color::Rgb(80, 255, 80),
        4 => Color::Rgb(0, 220, 50),
        _ => Color::Rgb(40, 180, 40),
    }
}

fn creature_color_blueprint(color_index: u8) -> Color {
    // Shades of white/cyan on blue
    match color_index % 6 {
        0 => Color::White,
        1 => Color::Rgb(180, 220, 255),
        2 => Color::Rgb(140, 200, 255),
        3 => Color::LightCyan,
        4 => Color::Rgb(200, 230, 255),
        _ => Color::Rgb(160, 210, 255),
    }
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

/// The main TUI renderer using ratatui.
pub struct TuiRenderer {
    bubbles: BubbleSystem,
    last_buffer: Option<Buffer>,
    flash: Option<(String, Instant)>,
    recorder: Option<GifRecorder>,
}

impl TuiRenderer {
    pub fn new() -> Self {
        Self {
            bubbles: BubbleSystem::new(),
            last_buffer: None,
            flash: None,
            recorder: None,
        }
    }

    /// Render the full scene: tank + HUD.
    pub fn render_with_hud(
        &mut self,
        frame: &mut Frame,
        sim: &dyn Simulation,
        display: &DisplayState,
    ) {
        let area = frame.area();
        let (tw, th) = sim.tank_size();
        let stats = sim.stats();
        let diagnostics = if display.show_diagnostics {
            Some(sim.ecology_diagnostics())
        } else {
            None
        };

        // Tick bubbles
        self.bubbles.tick(0.016, tw as f32, th as f32);

        // Reserve 3 rows at bottom for the default HUD, or 9 rows when the
        // diagnostics overlay is active.
        let footer_rows = if display.show_diagnostics { 9 } else { 3 };
        let tank_width = tw.saturating_add(2);
        let tank_height = th.saturating_add(2);
        let content_area = centered_rect(area, tank_width, tank_height.saturating_add(footer_rows));
        let tank_area = Rect::new(
            content_area.x,
            content_area.y,
            content_area.width,
            tank_height.min(content_area.height.saturating_sub(footer_rows)),
        );
        let hud_area = Rect::new(
            content_area.x,
            content_area.y + tank_area.height,
            content_area.width,
            content_area.height.saturating_sub(tank_area.height),
        );

        tank::render_tank(frame, tank_area, sim, &self.bubbles, display.theme);
        hud::render_hud(
            frame,
            hud_area,
            &stats,
            diagnostics.as_ref(),
            sim.environment(),
            display.paused,
            display.speed,
            sim.diversity_coefficient(),
            display.is_recording,
            display.recording_secs,
        );

        if display.show_help {
            hud::render_help_popup(frame, area);
        }

        // Show the flash popup if it hasn't expired yet.
        if let Some((msg, at)) = &self.flash {
            if at.elapsed().as_secs_f32() < FLASH_DURATION_SECS {
                hud::render_flash_popup(frame, area, msg);
            } else {
                self.flash = None;
            }
        }

        // Cache the completed buffer so save_screenshot() can use it.
        self.last_buffer = Some(frame.buffer_mut().clone());

        // Capture a GIF frame if recording (self-throttles to ~10 fps).
        if let Some(recorder) = &mut self.recorder {
            if let Some(buf) = &self.last_buffer {
                if let Err(e) = recorder.add_frame(buf) {
                    let msg = format!("Recording error: {e}");
                    // Stop the broken recorder and flash an error.
                    self.recorder = None;
                    self.flash = Some((msg, Instant::now()));
                }
            }
        }
    }

    /// Start GIF recording to the given path. Returns an error if a
    /// recording is already in progress or the encoder cannot be created.
    pub fn start_recording(
        &mut self,
        path: std::path::PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.recorder.is_some() {
            return Err("Already recording".into());
        }
        let buffer = self
            .last_buffer
            .as_ref()
            .ok_or("No frame rendered yet — cannot determine dimensions")?;
        let recorder = GifRecorder::start(buffer, path)?;
        self.recorder = Some(recorder);
        Ok(())
    }

    /// Stop the current recording. Returns the output path and frame count,
    /// or `None` if no recording was in progress.
    pub fn stop_recording(&mut self) -> Option<(std::path::PathBuf, u32)> {
        self.recorder.take().map(|r| r.stop())
    }

    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    pub fn recording_elapsed_secs(&self) -> u32 {
        self.recorder
            .as_ref()
            .map(|r| r.elapsed().as_secs() as u32)
            .unwrap_or(0)
    }
}

impl Renderer for TuiRenderer {
    fn render(&mut self, frame: &mut Frame, sim: &dyn Simulation) {
        self.render_with_hud(
            frame,
            sim,
            &DisplayState {
                paused: false,
                speed: 1.0,
                show_diagnostics: false,
                show_help: false,
                is_recording: false,
                recording_secs: 0,
                theme: RenderTheme::Classic,
            },
        );
    }

    fn save_screenshot(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = self
            .last_buffer
            .as_ref()
            .ok_or("No frame has been rendered yet")?;
        screenshot::save_buffer_as_png(buffer, path)
    }

    fn flash_message(&mut self, message: String) {
        self.flash = Some((message, Instant::now()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    use tuiq_core::AquariumSim;

    #[test]
    fn test_centered_rect_uses_requested_size_when_space_allows() {
        let rect = centered_rect(Rect::new(0, 0, 120, 50), 98, 37);
        assert_eq!(rect, Rect::new(11, 6, 98, 37));
    }

    #[test]
    fn test_centered_rect_clamps_to_available_space() {
        let rect = centered_rect(Rect::new(4, 2, 40, 10), 98, 37);
        assert_eq!(rect, Rect::new(4, 2, 40, 10));
    }

    #[test]
    fn test_render_with_diagnostics_does_not_panic_on_small_viewport() {
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let sim = AquariumSim::new(32, 8);
        let mut renderer = TuiRenderer::new();

        terminal
            .draw(|frame| {
                renderer.render_with_hud(
                    frame,
                    &sim,
                    &DisplayState {
                        paused: false,
                        speed: 1.0,
                        show_diagnostics: true,
                        show_help: false,
                        is_recording: false,
                        recording_secs: 0,
                        theme: RenderTheme::Classic,
                    },
                );
            })
            .unwrap();
    }

    #[test]
    fn test_render_with_help_popup_does_not_panic() {
        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let sim = AquariumSim::new(80, 30);
        let mut renderer = TuiRenderer::new();

        terminal
            .draw(|frame| {
                renderer.render_with_hud(
                    frame,
                    &sim,
                    &DisplayState {
                        paused: false,
                        speed: 1.0,
                        show_diagnostics: false,
                        show_help: true,
                        is_recording: false,
                        recording_secs: 0,
                        theme: RenderTheme::Classic,
                    },
                );
            })
            .unwrap();
    }

    #[test]
    fn test_save_screenshot_errors_before_any_render() {
        let renderer = TuiRenderer::new();
        assert!(renderer
            .save_screenshot(std::path::Path::new("unreachable.png"))
            .is_err());
    }

    #[test]
    fn test_save_screenshot_succeeds_after_render() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let sim = AquariumSim::new(80, 24);
        let mut renderer = TuiRenderer::new();

        terminal
            .draw(|frame| {
                renderer.render_with_hud(
                    frame,
                    &sim,
                    &DisplayState {
                        paused: false,
                        speed: 1.0,
                        show_diagnostics: false,
                        show_help: false,
                        is_recording: false,
                        recording_secs: 0,
                        theme: RenderTheme::Classic,
                    },
                );
            })
            .unwrap();

        let path = std::path::Path::new("test_screenshot_output.png");
        assert!(renderer.save_screenshot(path).is_ok());
        assert!(path.exists());
        let _ = std::fs::remove_file(path);
    }
}
