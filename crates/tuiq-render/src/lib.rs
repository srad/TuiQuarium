pub mod ascii;
pub mod effects;
pub mod hud;
pub mod palette;
pub mod tank;

use effects::BubbleSystem;
use ratatui::layout::Rect;
use ratatui::Frame;
use tuiq_core::Simulation;

/// Abstraction over rendering. Could be swapped for a test renderer.
pub trait Renderer {
    /// Render one frame from the current simulation state.
    fn render(&mut self, frame: &mut Frame, sim: &dyn Simulation);
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
}

impl TuiRenderer {
    pub fn new() -> Self {
        Self {
            bubbles: BubbleSystem::new(),
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
}
