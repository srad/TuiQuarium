pub mod ascii;
pub mod effects;
pub mod hud;
pub mod palette;
pub mod tank;
pub mod templates;

use effects::BubbleSystem;
use ratatui::Frame;
use ratatui::layout::Rect;
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

        // Tick bubbles
        self.bubbles.tick(0.016, tw as f32, th as f32);

        // Reserve 2 rows at bottom for HUD
        let tank_area = Rect::new(area.x, area.y, area.width, area.height.saturating_sub(2));
        let hud_area = Rect::new(
            area.x,
            area.y + tank_area.height,
            area.width,
            2.min(area.height),
        );

        tank::render_tank(frame, tank_area, sim, &self.bubbles);
        hud::render_hud(
            frame,
            hud_area,
            &sim.stats(),
            sim.environment(),
            display.paused,
            display.speed,
        );
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
            },
        );
    }
}
