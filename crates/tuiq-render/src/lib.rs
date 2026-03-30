pub mod ascii;
pub mod effects;
pub mod hud;
pub mod palette;
pub mod screenshot;
pub mod tank;

use std::path::Path;
use std::time::Instant;

use effects::BubbleSystem;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::Frame;
use tuiq_core::Simulation;

/// Duration a flash message stays visible.
const FLASH_DURATION_SECS: f32 = 3.0;

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
}

/// The main TUI renderer using ratatui.
pub struct TuiRenderer {
    bubbles: BubbleSystem,
    last_buffer: Option<Buffer>,
    flash: Option<(String, Instant)>,
}

impl TuiRenderer {
    pub fn new() -> Self {
        Self {
            bubbles: BubbleSystem::new(),
            last_buffer: None,
            flash: None,
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
        let tank_area = Rect::new(
            area.x,
            area.y,
            area.width,
            area.height.saturating_sub(footer_rows),
        );
        let hud_area = Rect::new(
            area.x,
            area.y + tank_area.height,
            area.width,
            footer_rows.min(area.height),
        );

        tank::render_tank(frame, tank_area, sim, &self.bubbles);
        hud::render_hud(
            frame,
            hud_area,
            &stats,
            diagnostics.as_ref(),
            sim.environment(),
            display.paused,
            display.speed,
            sim.diversity_coefficient(),
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
