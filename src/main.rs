mod session_persistence;
mod wgpu_frontend;

use std::error::Error;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::Backend,
    backend::CrosstermBackend,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Terminal,
};
use uuid::Uuid;

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
const MIN_WINDOW_FONT_SIZE_PX: u32 = 4;
const DEFAULT_WINDOW_WIDTH_PX: f64 = 1440.0;
const DEFAULT_WINDOW_HEIGHT_PX: f64 = 960.0;

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
    Confirm,
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
    SaveSession,
    ExportHistory,
    ToggleRecording,
    NextTheme,
    DeleteSelection,
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

struct SessionState {
    session_id: Uuid,
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

impl SessionState {
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
            session_id: Uuid::new_v4(),
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
            AppAction::Confirm => {}
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
            AppAction::SaveSession => {}
            AppAction::ExportHistory => {}
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
            AppAction::DeleteSelection => {}
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

struct AppState {
    mode: AppMode,
}

enum AppMode {
    Startup(StartupMenuState),
    SaveBrowser(SaveBrowserState),
    Running(SessionState),
}

struct StartupMenuState {
    selected: usize,
    has_valid_save: bool,
    has_any_save: bool,
    message: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SaveBrowserMode {
    Load,
    Delete,
}

struct SaveBrowserState {
    selected: usize,
    entries: Vec<session_persistence::SaveListEntry>,
    mode: SaveBrowserMode,
    confirm_delete: bool,
    message: Option<String>,
}

impl AppState {
    fn new() -> Self {
        Self {
            mode: AppMode::Startup(Self::startup_menu(None)),
        }
    }

    fn startup_menu(message: Option<String>) -> StartupMenuState {
        let entries = session_persistence::list_save_entries().unwrap_or_default();
        StartupMenuState {
            selected: 0,
            has_valid_save: session_persistence::latest_valid_save(&entries).is_some(),
            has_any_save: !entries.is_empty(),
            message,
        }
    }

    fn save_browser(mode: SaveBrowserMode, message: Option<String>) -> SaveBrowserState {
        SaveBrowserState {
            selected: 0,
            entries: session_persistence::list_save_entries().unwrap_or_default(),
            mode,
            confirm_delete: false,
            message,
        }
    }

    fn refresh_browser_entries(browser: &mut SaveBrowserState) {
        browser.entries = session_persistence::list_save_entries().unwrap_or_default();
        if browser.entries.is_empty() {
            browser.selected = 0;
        } else {
            browser.selected = browser
                .selected
                .min(browser.entries.len().saturating_sub(1));
        }
    }

    fn request_browser_delete(browser: &mut SaveBrowserState) {
        let Some(entry) = browser.entries.get(browser.selected) else {
            browser.message = Some("No saves available".to_string());
            browser.confirm_delete = false;
            return;
        };

        if !browser.confirm_delete {
            browser.confirm_delete = true;
            browser.message = Some(format!(
                "Press delete again to remove session {}\nJSON: {}\nCSV: matching session history file",
                entry.label,
                abbreviate_path_for_flash(&entry.path, 72),
            ));
            return;
        }

        match session_persistence::delete_save(&entry.path) {
            Ok(deleted) => {
                let mut message = format!(
                    "Deleted session {}\nJSON: {}",
                    entry.label,
                    abbreviate_path_for_flash(&deleted.session_path, 72),
                );
                if let Some(csv_path) = deleted.history_csv_path {
                    message.push_str(&format!(
                        "\nCSV: {}",
                        abbreviate_path_for_flash(&csv_path, 72)
                    ));
                }
                browser.message = Some(message);
                browser.confirm_delete = false;
                Self::refresh_browser_entries(browser);
            }
            Err(err) => {
                browser.message = Some(format!("Delete failed: {err}"));
                browser.confirm_delete = false;
            }
        }
    }

    fn display_state(&self) -> DisplayState {
        match &self.mode {
            AppMode::Running(session) => session.display_state(),
            _ => DisplayState {
                paused: false,
                speed: 1.0,
                show_diagnostics: false,
                show_help: false,
                is_recording: false,
                recording_secs: 0,
                theme: RenderTheme::Ocean,
            },
        }
    }

    fn draw<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<(), B::Error> {
        match &mut self.mode {
            AppMode::Running(session) => session.draw(terminal),
            AppMode::Startup(menu) => {
                terminal.draw(|frame| render_startup_menu(frame, menu))?;
                Ok(())
            }
            AppMode::SaveBrowser(browser) => {
                terminal.draw(|frame| render_save_browser(frame, browser))?;
                Ok(())
            }
        }
    }

    fn advance(&mut self) {
        if let AppMode::Running(session) = &mut self.mode {
            session.advance();
        }
    }

    fn apply_action(&mut self, action: AppAction) -> io::Result<bool> {
        let placeholder = AppMode::Startup(Self::startup_menu(None));
        let current_mode = std::mem::replace(&mut self.mode, placeholder);

        match current_mode {
            AppMode::Running(mut session) => {
                match action {
                    AppAction::SaveSession => match session_persistence::save_session(&session) {
                        Ok(paths) => session.renderer.flash_message(format!(
                            "Session saved\n{}\nCSV exported\n{}",
                            abbreviate_path_for_flash(&paths.session_path, 72),
                            abbreviate_path_for_flash(&paths.history_csv_path, 72),
                        )),
                        Err(err) => session
                            .renderer
                            .flash_message(format!("Session save failed: {err}")),
                    },
                    AppAction::ExportHistory => {
                        match session_persistence::export_history_csv(&session) {
                            Ok(_) => session
                                .renderer
                                .flash_message("History CSV exported".to_string()),
                            Err(err) => session
                                .renderer
                                .flash_message(format!("History export failed: {err}")),
                        }
                    }
                    other => {
                        let should_quit = session.apply_action(other)?;
                        self.mode = AppMode::Running(session);
                        return Ok(should_quit);
                    }
                }
                self.mode = AppMode::Running(session);
                Ok(false)
            }
            AppMode::Startup(mut menu) => match action {
                AppAction::IncreaseDiversity => {
                    menu.selected = menu.selected.saturating_sub(1);
                    self.mode = AppMode::Startup(menu);
                    Ok(false)
                }
                AppAction::DecreaseDiversity => {
                    menu.selected = (menu.selected + 1).min(4);
                    self.mode = AppMode::Startup(menu);
                    Ok(false)
                }
                AppAction::Confirm => match menu.selected {
                    0 => {
                        self.mode = AppMode::Running(SessionState::new());
                        Ok(false)
                    }
                    1 => {
                        let entries = session_persistence::list_save_entries().unwrap_or_default();
                        if let Some(entry) = session_persistence::latest_valid_save(&entries) {
                            match session_persistence::load_session(&entry.path) {
                                Ok(session) => self.mode = AppMode::Running(session),
                                Err(err) => {
                                    menu.message = Some(format!("Load failed: {err}"));
                                    self.mode = AppMode::Startup(menu);
                                }
                            }
                        } else {
                            menu.message = Some("No valid saves found".to_string());
                            self.mode = AppMode::Startup(menu);
                        }
                        Ok(false)
                    }
                    2 => {
                        self.mode =
                            AppMode::SaveBrowser(Self::save_browser(SaveBrowserMode::Load, None));
                        Ok(false)
                    }
                    3 => {
                        self.mode =
                            AppMode::SaveBrowser(Self::save_browser(SaveBrowserMode::Delete, None));
                        Ok(false)
                    }
                    _ => Ok(true),
                },
                AppAction::Quit => Ok(true),
                _ => {
                    self.mode = AppMode::Startup(menu);
                    Ok(false)
                }
            },
            AppMode::SaveBrowser(mut browser) => {
                match action {
                    AppAction::IncreaseDiversity => {
                        browser.selected = browser.selected.saturating_sub(1);
                        browser.confirm_delete = false;
                    }
                    AppAction::DecreaseDiversity => {
                        if !browser.entries.is_empty() {
                            browser.selected =
                                (browser.selected + 1).min(browser.entries.len().saturating_sub(1));
                        }
                        browser.confirm_delete = false;
                    }
                    AppAction::Confirm => {
                        if browser.mode == SaveBrowserMode::Delete {
                            Self::request_browser_delete(&mut browser);
                        } else {
                            let Some(entry) = browser.entries.get(browser.selected) else {
                                browser.message = Some("No saves available".to_string());
                                self.mode = AppMode::SaveBrowser(browser);
                                return Ok(false);
                            };
                            match &entry.preview {
                                Ok(_) => match session_persistence::load_session(&entry.path) {
                                    Ok(session) => {
                                        self.mode = AppMode::Running(session);
                                        return Ok(false);
                                    }
                                    Err(err) => {
                                        browser.message = Some(format!("Load failed: {err}"));
                                    }
                                },
                                Err(err) => {
                                    browser.message = Some(format!("Save is not loadable: {err}"));
                                }
                            }
                        }
                    }
                    AppAction::DeleteSelection => {
                        Self::request_browser_delete(&mut browser);
                    }
                    AppAction::Quit => {
                        browser.confirm_delete = false;
                        self.mode = AppMode::Startup(Self::startup_menu(browser.message.take()));
                        return Ok(false);
                    }
                    _ => {}
                }
                self.mode = AppMode::SaveBrowser(browser);
                Ok(false)
            }
        }
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
    println!();
    println!("Startup menu:");
    println!("  New Simulation | Continue Latest Save | Load Save... | Delete Save... | Quit");
    println!("In-run save/export:");
    println!("  s = save session, e = export daily history CSV");
}

fn increase_sim_speed(speed: f32) -> f32 {
    (speed + SIM_SPEED_STEP).min(MAX_SIM_SPEED)
}

fn decrease_sim_speed(speed: f32) -> f32 {
    (speed - SIM_SPEED_STEP).max(MIN_SIM_SPEED)
}

fn abbreviate_path_for_flash(path: &Path, max_chars: usize) -> String {
    let display = path.display().to_string();
    if display.chars().count() <= max_chars {
        return display;
    }

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if file_name.chars().count() + 4 >= max_chars {
        let tail_len = max_chars.saturating_sub(3);
        let tail: String = display
            .chars()
            .rev()
            .take(tail_len)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return format!("...{tail}");
    }

    let separator = std::path::MAIN_SEPARATOR;
    let remaining = max_chars.saturating_sub(file_name.chars().count() + 4);
    let prefix: String = display.chars().take(remaining).collect();
    format!("{prefix}...{separator}{file_name}")
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
        KeyCode::Enter => Some(AppAction::Confirm),
        KeyCode::Delete | KeyCode::Backspace => Some(AppAction::DeleteSelection),
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
        KeyCode::Char('s') => Some(AppAction::SaveSession),
        KeyCode::Char('e') => Some(AppAction::ExportHistory),
        KeyCode::Char('g') => Some(AppAction::ToggleRecording),
        KeyCode::Char('t') => Some(AppAction::NextTheme),
        _ => None,
    };

    Ok(match action {
        Some(action) if key.kind == KeyEventKind::Repeat && !action_allows_repeat(action) => None,
        other => other,
    })
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    if area.width == 0 || area.height == 0 {
        return area;
    }
    let width = width.min(area.width).max(1);
    let height = height.min(area.height).max(1);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

#[derive(Clone, Copy)]
struct StartupMenuItem {
    label: &'static str,
    enabled: bool,
    destructive: bool,
}

#[derive(Clone, Copy)]
struct MenuPalette {
    mid_water_bg: Color,
    deep_water_bg: Color,
    substrate_bg: Color,
    bubble_fg: Color,
    reed_fg: Color,
    substrate_fg: Color,
    panel_bg: Color,
    panel_border: Color,
    panel_shadow: Color,
    header_bg: Color,
    title_fg: Color,
    subtitle_fg: Color,
    text_fg: Color,
    muted_fg: Color,
    selected_bg: Color,
    selected_fg: Color,
    delete_bg: Color,
    delete_fg: Color,
    warning_bg: Color,
    warning_fg: Color,
    success_bg: Color,
    success_fg: Color,
    invalid_fg: Color,
}

fn menu_palette() -> MenuPalette {
    MenuPalette {
        mid_water_bg: Color::Rgb(0, 20, 50),
        deep_water_bg: Color::Rgb(35, 30, 20),
        substrate_bg: Color::Rgb(60, 45, 15),
        bubble_fg: Color::Blue,
        reed_fg: Color::DarkGray,
        substrate_fg: Color::Yellow,
        panel_bg: Color::Rgb(8, 18, 26),
        panel_border: Color::Rgb(126, 194, 206),
        panel_shadow: Color::Rgb(1, 6, 10),
        header_bg: Color::Rgb(14, 42, 58),
        title_fg: Color::Rgb(240, 250, 252),
        subtitle_fg: Color::Rgb(178, 214, 220),
        text_fg: Color::Rgb(224, 232, 236),
        muted_fg: Color::Rgb(134, 160, 170),
        selected_bg: Color::Rgb(26, 78, 92),
        selected_fg: Color::Rgb(250, 252, 252),
        delete_bg: Color::Rgb(84, 42, 28),
        delete_fg: Color::Rgb(252, 216, 188),
        warning_bg: Color::Rgb(84, 42, 28),
        warning_fg: Color::Rgb(255, 232, 196),
        success_bg: Color::Rgb(18, 62, 62),
        success_fg: Color::Rgb(204, 246, 240),
        invalid_fg: Color::Rgb(246, 156, 148),
    }
}

fn startup_menu_items(has_valid_save: bool, has_any_save: bool) -> [StartupMenuItem; 5] {
    [
        StartupMenuItem {
            label: "New Simulation",
            enabled: true,
            destructive: false,
        },
        StartupMenuItem {
            label: "Continue Latest Save",
            enabled: has_valid_save,
            destructive: false,
        },
        StartupMenuItem {
            label: "Load Save...",
            enabled: true,
            destructive: false,
        },
        StartupMenuItem {
            label: "Delete Save...",
            enabled: has_any_save,
            destructive: true,
        },
        StartupMenuItem {
            label: "Quit",
            enabled: true,
            destructive: false,
        },
    ]
}

fn startup_selection_hint(selected: usize) -> &'static str {
    match selected {
        0 => "Start a fresh lake from founder plants and fauna.",
        1 => "Resume the newest valid save and continue the same session id.",
        2 => "Browse existing sessions and load one into the simulation shell.",
        3 => "Delete a saved session and its matching ecology-history CSV export.",
        _ => "Exit the application without changing simulation files.",
    }
}

fn fit_text_to_width(text: &str, width: u16) -> String {
    let width = width as usize;
    if width == 0 {
        return String::new();
    }

    let count = text.chars().count();
    if count <= width {
        let mut result = String::with_capacity(width);
        result.push_str(text);
        result.push_str(&" ".repeat(width - count));
        return result;
    }

    if width <= 3 {
        return text.chars().take(width).collect();
    }

    let mut result: String = text.chars().take(width - 3).collect();
    result.push_str("...");
    result
}

fn full_width_line(text: &str, width: u16, style: Style) -> Line<'static> {
    Line::from(Span::styled(fit_text_to_width(text, width), style))
}

fn blank_line(width: u16, bg: Color) -> Line<'static> {
    full_width_line("", width, Style::default().bg(bg))
}

fn message_lines(message: &str, width: u16, style: Style) -> Vec<Line<'static>> {
    message
        .lines()
        .map(|line| full_width_line(line, width, style))
        .collect()
}

fn shadow_rect(area: Rect, frame_area: Rect) -> Rect {
    let max_width = frame_area
        .width
        .saturating_sub(area.x.saturating_sub(frame_area.x));
    let max_height = frame_area
        .height
        .saturating_sub(area.y.saturating_sub(frame_area.y));
    Rect::new(
        area.x
            .saturating_add(2)
            .min(frame_area.right().saturating_sub(1)),
        area.y
            .saturating_add(1)
            .min(frame_area.bottom().saturating_sub(1)),
        area.width.min(max_width.saturating_sub(2)),
        area.height.min(max_height.saturating_sub(1)),
    )
}

fn render_menu_background(frame: &mut ratatui::Frame) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    let pal = menu_palette();
    let substrate_height = area.height.min(2);
    let water_height = area.height.saturating_sub(substrate_height);
    let mut lines = Vec::with_capacity(area.height as usize);

    for y in 0..area.height {
        let mut row = String::with_capacity(area.width as usize);
        for x in 0..area.width {
            let ch = if y < water_height {
                if ((x as usize) + (y as usize)) % 4 == 0
                    || ((x as usize) + (y as usize)) % 4 == 2
                {
                    '~'
                } else {
                    ' '
                }
            } else if y == water_height {
                match ((x as usize) * 7 + (y as usize) * 3) % 5 {
                    0 | 2 => ':',
                    1 | 3 => '.',
                    _ => ' ',
                }
            } else {
                match ((x as usize) * 3 + (y as usize) * 7) % 3 {
                    0 | 2 => ':',
                    _ => '.',
                }
            };
            row.push(ch);
        }

        let (fg, bg) = if y < water_height {
            (pal.bubble_fg, pal.mid_water_bg)
        } else if y == water_height {
            (pal.reed_fg, pal.deep_water_bg)
        } else {
            (pal.substrate_fg, pal.substrate_bg)
        };

        lines.push(Line::from(Span::styled(
            row,
            Style::default().fg(fg).bg(bg),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::default().bg(pal.mid_water_bg))
            .alignment(Alignment::Left),
        area,
    );
}

fn render_menu_panel(frame: &mut ratatui::Frame, area: Rect, title: &str, subtitle: &str) -> Rect {
    let pal = menu_palette();
    let frame_area = frame.area();
    let shadow = shadow_rect(area, frame_area);
    frame.render_widget(
        Block::default().style(Style::default().bg(pal.panel_shadow)),
        shadow,
    );

    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(pal.panel_border))
        .style(Style::default().bg(pal.panel_bg).fg(pal.text_fg));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return inner;
    }

    frame.render_widget(
        Block::default().style(Style::default().bg(pal.panel_bg)),
        inner,
    );
    if inner.width < 4 || inner.height < 4 {
        return inner;
    }

    let header_height = inner.height.min(3);
    let header = Rect::new(inner.x, inner.y, inner.width, header_height);
    frame.render_widget(
        Block::default().style(Style::default().bg(pal.header_bg)),
        header,
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                title.to_string(),
                Style::default()
                    .fg(pal.title_fg)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                subtitle.to_string(),
                Style::default().fg(pal.subtitle_fg),
            )),
        ])
        .alignment(Alignment::Center)
        .style(Style::default().bg(pal.header_bg)),
        header,
    );

    let divider_y = header.y.saturating_add(header.height);
    if divider_y < inner.bottom() {
        frame.render_widget(
            Paragraph::new(full_width_line(
                "────────────────────────────────",
                inner.width,
                Style::default().bg(pal.panel_bg).fg(pal.panel_border),
            )),
            Rect::new(inner.x, divider_y, inner.width, 1),
        );
    }

    let body_top = divider_y.saturating_add(1);
    let body_height = inner.bottom().saturating_sub(body_top);
    Rect::new(
        inner.x.saturating_add(1),
        body_top,
        inner.width.saturating_sub(2),
        body_height.saturating_sub(1),
    )
}

fn render_startup_menu(frame: &mut ratatui::Frame, menu: &StartupMenuState) {
    let pal = menu_palette();
    render_menu_background(frame);
    let area = centered_rect(frame.area(), 70, 20);
    let title = format!("TuiQuarium {}", env!("CARGO_PKG_VERSION"));
    let inner = render_menu_panel(
        frame,
        area,
        &title,
        "Lake session control and analysis workspace",
    );
    let width = inner.width;
    let mut lines = vec![blank_line(width, pal.panel_bg)];

    for (index, item) in startup_menu_items(menu.has_valid_save, menu.has_any_save)
        .iter()
        .enumerate()
    {
        let selected = index == menu.selected;
        let prefix = if selected { ">" } else { " " };
        let style = if !item.enabled {
            Style::default().fg(pal.muted_fg).bg(pal.panel_bg)
        } else if item.destructive && selected {
            Style::default()
                .fg(pal.delete_fg)
                .bg(pal.delete_bg)
                .add_modifier(Modifier::BOLD)
        } else if selected {
            Style::default()
                .fg(pal.selected_fg)
                .bg(pal.selected_bg)
                .add_modifier(Modifier::BOLD)
        } else if item.destructive {
            Style::default().fg(pal.delete_fg).bg(pal.panel_bg)
        } else {
            Style::default().fg(pal.text_fg).bg(pal.panel_bg)
        };
        let suffix = if !item.enabled { " (unavailable)" } else { "" };
        lines.push(full_width_line(
            &format!(" {prefix} {}{suffix}", item.label),
            width,
            style,
        ));
    }

    lines.push(blank_line(width, pal.panel_bg));
    lines.push(full_width_line(
        startup_selection_hint(menu.selected),
        width,
        Style::default()
            .fg(pal.subtitle_fg)
            .bg(Color::Rgb(11, 34, 44)),
    ));
    lines.push(blank_line(width, pal.panel_bg));
    lines.push(full_width_line(
        "Up/Down Move   Enter Select   Esc Quit",
        width,
        Style::default().fg(pal.muted_fg).bg(pal.panel_bg),
    ));

    if let Some(message) = &menu.message {
        lines.push(blank_line(width, pal.panel_bg));
        lines.extend(message_lines(
            message,
            width,
            Style::default()
                .fg(pal.warning_fg)
                .bg(pal.warning_bg)
                .add_modifier(Modifier::BOLD),
        ));
    }

    frame.render_widget(Paragraph::new(lines).alignment(Alignment::Left), inner);
}

fn render_save_browser(frame: &mut ratatui::Frame, browser: &SaveBrowserState) {
    let pal = menu_palette();
    render_menu_background(frame);
    let area = centered_rect(frame.area(), 96, 24);
    let (heading, subtitle, hint) = match browser.mode {
        SaveBrowserMode::Load => (
            "Saved Sessions",
            "Load a saved lake and continue the same session lineage",
            "Up/Down Move   Enter Load   Del/Backspace Delete   Esc Back",
        ),
        SaveBrowserMode::Delete => (
            "Delete Sessions",
            "Remove a session JSON and its matching ecology-history CSV",
            "Up/Down Move   Enter Delete   Del/Backspace Delete   Esc Back",
        ),
    };
    let inner = render_menu_panel(frame, area, heading, subtitle);
    let width = inner.width;
    let mut lines = vec![
        full_width_line(
            hint,
            width,
            Style::default().fg(pal.muted_fg).bg(pal.panel_bg),
        ),
        blank_line(width, pal.panel_bg),
    ];

    if browser.entries.is_empty() {
        lines.push(full_width_line(
            "No save files found in ~/.tuiquarium/saves/",
            width,
            Style::default().fg(pal.muted_fg).bg(pal.panel_bg),
        ));
    } else {
        for (index, entry) in browser.entries.iter().enumerate() {
            let selected = index == browser.selected;
            let (top_style, detail_style) = if matches!(&entry.preview, Err(_)) {
                (
                    Style::default().fg(pal.invalid_fg).bg(pal.panel_bg),
                    Style::default().fg(pal.invalid_fg).bg(pal.panel_bg),
                )
            } else if selected && browser.mode == SaveBrowserMode::Delete {
                (
                    Style::default()
                        .fg(pal.delete_fg)
                        .bg(pal.delete_bg)
                        .add_modifier(Modifier::BOLD),
                    Style::default().fg(pal.delete_fg).bg(pal.delete_bg),
                )
            } else if selected {
                (
                    Style::default()
                        .fg(pal.selected_fg)
                        .bg(pal.selected_bg)
                        .add_modifier(Modifier::BOLD),
                    Style::default().fg(pal.selected_fg).bg(pal.selected_bg),
                )
            } else {
                (
                    Style::default().fg(pal.text_fg).bg(pal.panel_bg),
                    Style::default().fg(pal.muted_fg).bg(pal.panel_bg),
                )
            };
            let prefix = if selected { ">" } else { " " };
            lines.push(full_width_line(
                &format!(" {prefix} Session {}", entry.label),
                width,
                top_style,
            ));

            let detail = match &entry.preview {
                Ok(preview) => format!(
                    "   Day {}  |  Pop {}  |  Sp {}  |  Prod {:.1}  |  v{}  |  unix {}",
                    preview.summary.day,
                    preview.summary.creature_count,
                    preview.summary.species_count,
                    preview.summary.producer_biomass,
                    preview.app_version,
                    entry.saved_at_unix_secs
                ),
                Err(err) => format!("   Invalid save: {err}"),
            };
            lines.push(full_width_line(&detail, width, detail_style));
        }
    }

    if browser.confirm_delete {
        lines.push(blank_line(width, pal.panel_bg));
        lines.push(full_width_line(
            "Delete is armed for the selected session. Press delete again to confirm.",
            width,
            Style::default()
                .fg(pal.warning_fg)
                .bg(pal.warning_bg)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if let Some(message) = &browser.message {
        lines.push(blank_line(width, pal.panel_bg));
        let message_style = if message.starts_with("Deleted") {
            Style::default()
                .fg(pal.success_fg)
                .bg(pal.success_bg)
                .add_modifier(Modifier::BOLD)
        } else if message.starts_with("Delete failed")
            || message.starts_with("Load failed")
            || message.starts_with("Save is not loadable")
            || message.starts_with("Press delete again")
        {
            Style::default()
                .fg(pal.warning_fg)
                .bg(pal.warning_bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(pal.text_fg).bg(pal.panel_bg)
        };
        lines.extend(message_lines(message, width, message_style));
    }

    frame.render_widget(Paragraph::new(lines).alignment(Alignment::Left), inner);
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

    #[test]
    fn test_startup_menu_items_include_delete_entry() {
        let items = startup_menu_items(false, false);
        assert_eq!(items.len(), 5);
        assert_eq!(items[3].label, "Delete Save...");
        assert!(!items[1].enabled);
        assert!(!items[3].enabled);
        assert!(items[3].destructive);
    }

    #[test]
    fn test_centered_rect_handles_tiny_terminal() {
        assert_eq!(
            centered_rect(Rect::new(0, 0, 2, 1), 64, 18),
            Rect::new(0, 0, 2, 1)
        );
        assert_eq!(
            centered_rect(Rect::new(0, 0, 0, 0), 64, 18),
            Rect::new(0, 0, 0, 0)
        );
    }
}
