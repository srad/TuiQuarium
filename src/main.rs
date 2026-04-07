mod wgpu_frontend;

use std::error::Error;
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::Backend, backend::CrosstermBackend, Terminal};

use tuiq_core::components::*;
use tuiq_core::genome::CreatureGenome;
use tuiq_core::{AquariumSim, Simulation};
use tuiq_render::ascii::generate_frames;
use tuiq_render::{DisplayState, RenderTheme, Renderer, TuiRenderer};

const MIN_SIM_SPEED: f32 = 0.5;
const MAX_SIM_SPEED: f32 = 100.0;
const SIM_SPEED_STEP: f32 = 0.5;
const FIXED_TANK_WIDTH: u16 = 136;
const FIXED_TANK_HEIGHT: u16 = 44;
const MAX_DIAGNOSTIC_HUD_ROWS: u16 = 9;
const MIN_WINDOW_FONT_SIZE_PX: u32 = 8;
const DEFAULT_WINDOW_WIDTH_PX: f64 = 1440.0;
const DEFAULT_WINDOW_HEIGHT_PX: f64 = 960.0;
const MIN_WINDOW_WIDTH_PX: f64 = 1280.0;
const MIN_WINDOW_HEIGHT_PX: f64 = 840.0;

fn main() -> Result<(), Box<dyn Error>> {
    match parse_args()? {
        CliAction::Run(frontend) => match frontend {
            Frontend::Terminal => run_terminal().map_err(Into::into),
            Frontend::Wgpu => wgpu_frontend::run_wgpu(),
        },
        CliAction::PrintHelp => {
            print_help();
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frontend {
    Terminal,
    Wgpu,
}

enum CliAction {
    Run(Frontend),
    PrintHelp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AppAction {
    Quit,
    TogglePause,
    IncreaseSpeed,
    DecreaseSpeed,
    ToggleDiagnostics,
    DropFood,
    DecreaseDiversity,
    IncreaseDiversity,
    Reset,
    ToggleHelp,
    SaveScreenshot,
    ToggleRecording,
    NextTheme,
}

fn action_allows_repeat(action: AppAction) -> bool {
    matches!(
        action,
        AppAction::IncreaseSpeed
            | AppAction::DecreaseSpeed
            | AppAction::IncreaseDiversity
            | AppAction::DecreaseDiversity
    )
}

struct AppState {
    sim: AquariumSim,
    renderer: TuiRenderer,
    paused: bool,
    speed: f32,
    show_diagnostics: bool,
    show_help: bool,
    theme: RenderTheme,
    tick_rate: Duration,
    last_tick: Instant,
    accumulator: Duration,
}

impl AppState {
    fn new() -> Self {
        let mut sim = AquariumSim::new(FIXED_TANK_WIDTH, FIXED_TANK_HEIGHT);
        let mut renderer = TuiRenderer::new();

        let bootstrap_creatures = sim.bootstrap_founder_web();
        for entity in bootstrap_creatures {
            attach_creature_art(&mut sim, entity);
        }

        renderer.flash_message(format!(
            "Aquarium: {}x{}",
            FIXED_TANK_WIDTH, FIXED_TANK_HEIGHT
        ));

        Self {
            sim,
            renderer,
            paused: false,
            speed: 1.0,
            show_diagnostics: false,
            show_help: false,
            theme: RenderTheme::Ocean,
            tick_rate: Duration::from_millis(50),
            last_tick: Instant::now(),
            accumulator: Duration::ZERO,
        }
    }

    fn display_state(&self) -> DisplayState {
        DisplayState {
            paused: self.paused,
            speed: self.speed,
            show_diagnostics: self.show_diagnostics,
            show_help: self.show_help,
            is_recording: self.renderer.is_recording(),
            recording_secs: self.renderer.recording_elapsed_secs(),
            theme: self.theme,
        }
    }

    fn draw<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<(), B::Error> {
        let display = self.display_state();
        terminal.draw(|frame| {
            self.renderer.render_with_hud(frame, &self.sim, &display);
        })?;
        Ok(())
    }

    fn advance(&mut self) {
        if !self.paused {
            let now = Instant::now();
            self.accumulator += now - self.last_tick;
            self.last_tick = now;

            while self.accumulator >= self.tick_rate {
                self.sim.tick(self.tick_rate.as_secs_f32() * self.speed);
                self.accumulator -= self.tick_rate;
            }

            for entity in self.sim.drain_births() {
                attach_creature_art(&mut self.sim, entity);
            }
        } else {
            self.last_tick = Instant::now();
        }
    }

    fn apply_action(&mut self, action: AppAction) -> io::Result<bool> {
        match action {
            AppAction::Quit => return Ok(true),
            AppAction::TogglePause => self.paused = !self.paused,
            AppAction::IncreaseSpeed => {
                self.speed = increase_sim_speed(self.speed);
            }
            AppAction::DecreaseSpeed => {
                self.speed = decrease_sim_speed(self.speed);
            }
            AppAction::ToggleDiagnostics => {
                self.show_diagnostics = !self.show_diagnostics;
            }
            AppAction::DropFood => self.spawn_food(),
            AppAction::DecreaseDiversity => {
                let cur = self.sim.diversity_coefficient();
                self.sim.set_diversity_coefficient(cur - 0.1);
            }
            AppAction::IncreaseDiversity => {
                let cur = self.sim.diversity_coefficient();
                self.sim.set_diversity_coefficient(cur + 0.1);
            }
            AppAction::Reset => {
                self.speed = 1.0;
                self.sim.set_diversity_coefficient(1.0);
            }
            AppAction::ToggleHelp => {
                self.show_help = !self.show_help;
            }
            AppAction::SaveScreenshot => {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                match tuiq_render::screenshot::screenshots_dir() {
                    Ok(dir) => {
                        let path = dir.join(format!("tuiquarium_{ts}.png"));
                        match self.renderer.save_screenshot(&path) {
                            Ok(()) => self
                                .renderer
                                .flash_message(format!("Screenshot saved to {}", path.display())),
                            Err(e) => self
                                .renderer
                                .flash_message(format!("Screenshot failed: {e}")),
                        }
                    }
                    Err(e) => self
                        .renderer
                        .flash_message(format!("Cannot create screenshot dir: {e}")),
                }
            }
            AppAction::ToggleRecording => {
                if self.renderer.is_recording() {
                    if let Some((path, count)) = self.renderer.stop_recording() {
                        self.renderer.flash_message(format!(
                            "Recording saved: {} ({count} frames)",
                            path.display()
                        ));
                    }
                } else {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    match tuiq_render::gif_recorder::recordings_dir() {
                        Ok(dir) => {
                            let path = dir.join(format!("tuiquarium_{ts}.gif"));
                            match self.renderer.start_recording(path) {
                                Ok(()) => {}
                                Err(e) => self
                                    .renderer
                                    .flash_message(format!("Recording failed: {e}")),
                            }
                        }
                        Err(e) => self
                            .renderer
                            .flash_message(format!("Cannot create recordings dir: {e}")),
                    }
                }
            }
            AppAction::NextTheme => {
                self.theme = self.theme.next();
                self.renderer
                    .flash_message(format!("Theme: {}", self.theme.label()));
            }
        }

        Ok(false)
    }

    fn spawn_food(&mut self) {
        let (tank_width, _) = self.sim.tank_size();
        let food_x = (self.sim.stats().tick_count as f32 * 7.3) % tank_width as f32;
        let frame = AsciiFrame::from_rows(vec!["*"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 1,
        };
        self.sim.spawn_food(
            Position { x: food_x, y: 0.0 },
            Velocity { vx: 0.0, vy: 1.5 },
            BoundingBox { w: 1.0, h: 1.0 },
            appearance,
        );
    }
}

fn attach_creature_art(sim: &mut AquariumSim, entity: hecs::Entity) {
    let genome_copy: Option<CreatureGenome> = sim
        .world_mut()
        .get::<&CreatureGenome>(entity)
        .ok()
        .map(|g| (*g).clone());
    if let Some(genome) = genome_copy {
        let (swim_frames, idle_frames) = generate_frames(&genome);
        let w = swim_frames[0].width as f32;
        let h = swim_frames[0].height as f32;
        let color_index = genome.art.color_index();
        let appearance = Appearance {
            frame_sets: vec![swim_frames, idle_frames],
            facing: Direction::Right,
            color_index,
        };
        let _ = sim
            .world_mut()
            .insert(entity, (appearance, BoundingBox { w, h }));
    }
}

fn parse_args() -> Result<CliAction, String> {
    let mut frontend = Frontend::Terminal;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frontend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --frontend".to_string())?;
                frontend = parse_frontend(&value)?;
            }
            "--help" | "-h" => return Ok(CliAction::PrintHelp),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    Ok(CliAction::Run(frontend))
}

fn parse_frontend(value: &str) -> Result<Frontend, String> {
    match value {
        "terminal" | "tui" => Ok(Frontend::Terminal),
        "wgpu" | "gpu" => Ok(Frontend::Wgpu),
        _ => Err(format!(
            "unknown frontend '{value}', expected one of: terminal, gpu"
        )),
    }
}

fn print_help() {
    println!("tuiquarium");
    println!();
    println!("Usage:");
    println!("  tuiquarium [--frontend terminal|gpu]");
    println!();
    println!("Frontends:");
    println!("  terminal  Crossterm terminal backend");
    println!("  gpu       ratatui-wgpu window backend with fixed aquarium sizing");
    println!("            and automatic font scaling on resize");
}

fn increase_sim_speed(speed: f32) -> f32 {
    (speed + SIM_SPEED_STEP).min(MAX_SIM_SPEED)
}

fn decrease_sim_speed(speed: f32) -> f32 {
    (speed - SIM_SPEED_STEP).max(MIN_SIM_SPEED)
}

fn run_terminal() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut state = AppState::new();

    let result = run_terminal_app(&mut terminal, &mut state);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = &result {
        eprintln!("Error: {err}");
    }

    result
}

fn run_terminal_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut AppState,
) -> io::Result<()> {
    loop {
        state.draw(terminal)?;

        if let Some(action) = poll_terminal_action()? {
            if state.apply_action(action)? {
                return Ok(());
            }
        }

        state.advance();
    }
}

fn poll_terminal_action() -> io::Result<Option<AppAction>> {
    let timeout = Duration::from_millis(16);
    if !event::poll(timeout)? {
        return Ok(None);
    }

    let Event::Key(key) = event::read()? else {
        return Ok(None);
    };
    if key.kind == KeyEventKind::Release {
        return Ok(None);
    }

    let action = match key.code {
        KeyCode::Char('q') | KeyCode::Esc => Some(AppAction::Quit),
        KeyCode::Char(' ') => Some(AppAction::TogglePause),
        KeyCode::Right => Some(AppAction::IncreaseSpeed),
        KeyCode::Left => Some(AppAction::DecreaseSpeed),
        KeyCode::Char('d') => Some(AppAction::ToggleDiagnostics),
        KeyCode::Char('f') => Some(AppAction::DropFood),
        KeyCode::Down => Some(AppAction::DecreaseDiversity),
        KeyCode::Up => Some(AppAction::IncreaseDiversity),
        KeyCode::Char('r') => Some(AppAction::Reset),
        KeyCode::Char('?') | KeyCode::Char('h') => Some(AppAction::ToggleHelp),
        KeyCode::Char('p') => Some(AppAction::SaveScreenshot),
        KeyCode::Char('g') => Some(AppAction::ToggleRecording),
        KeyCode::Char('t') => Some(AppAction::NextTheme),
        _ => None,
    };

    Ok(match action {
        Some(action) if key.kind == KeyEventKind::Repeat && !action_allows_repeat(action) => None,
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontend_accepts_gpu_alias() {
        assert_eq!(parse_frontend("gpu").unwrap(), Frontend::Wgpu);
        assert_eq!(parse_frontend("wgpu").unwrap(), Frontend::Wgpu);
    }

    #[test]
    fn test_parse_frontend_rejects_unknown_value() {
        assert!(parse_frontend("web").is_err());
    }

    #[test]
    fn test_action_allows_repeat_only_for_continuous_adjustments() {
        assert!(action_allows_repeat(AppAction::IncreaseSpeed));
        assert!(action_allows_repeat(AppAction::DecreaseSpeed));
        assert!(action_allows_repeat(AppAction::IncreaseDiversity));
        assert!(action_allows_repeat(AppAction::DecreaseDiversity));
        assert!(!action_allows_repeat(AppAction::TogglePause));
        assert!(!action_allows_repeat(AppAction::NextTheme));
    }

    #[test]
    fn test_increase_sim_speed_clamps_at_maximum() {
        assert_eq!(increase_sim_speed(MAX_SIM_SPEED - 0.25), MAX_SIM_SPEED);
        assert_eq!(increase_sim_speed(MAX_SIM_SPEED), MAX_SIM_SPEED);
    }

    #[test]
    fn test_decrease_sim_speed_clamps_at_minimum() {
        assert_eq!(decrease_sim_speed(MIN_SIM_SPEED + 0.25), MIN_SIM_SPEED);
        assert_eq!(decrease_sim_speed(MIN_SIM_SPEED), MIN_SIM_SPEED);
    }
}
