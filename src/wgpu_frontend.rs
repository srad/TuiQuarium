use std::error::Error;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pollster::block_on;
use ratatui::{style::Color, Terminal};
use ratatui_wgpu::{Builder, ColorTable, Dimensions, Font, Viewport, WgpuBackend};
use rustybuzz::Face;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes};

use crate::{
    AppAction, AppState, DEFAULT_WINDOW_HEIGHT_PX, DEFAULT_WINDOW_WIDTH_PX, FIXED_TANK_HEIGHT,
    FIXED_TANK_WIDTH, MAX_DIAGNOSTIC_HUD_ROWS, MIN_WINDOW_FONT_SIZE_PX,
};

const WINDOW_TITLE: &str = concat!("TuiQuarium ", env!("CARGO_PKG_VERSION"));
const WINDOW_FONT_NAME: &str = "Cascadia Mono";
const WINDOW_FONT_DATA: &[u8] = include_bytes!("../crates/tuiq-render/fonts/CascadiaMono.ttf");
const WGPU_TARGET_COLUMNS: u16 = FIXED_TANK_WIDTH + 2;
const WGPU_BASE_ROWS: u16 = FIXED_TANK_HEIGHT + 2 + 3;
const WGPU_DIAGNOSTIC_ROWS: u16 = FIXED_TANK_HEIGHT + 2 + MAX_DIAGNOSTIC_HUD_ROWS;
const RESIZE_SETTLE_DELAY: Duration = Duration::from_millis(120);

type WindowTerminal = Terminal<WgpuBackend<'static, 'static>>;

pub fn run_wgpu() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = WgpuApp::new()?;
    event_loop.run_app(&mut app)?;

    if let Some(message) = app.error_message {
        return Err(message.into());
    }

    Ok(())
}

struct WgpuApp {
    window: Option<Arc<Window>>,
    terminal: Option<WindowTerminal>,
    state: AppState,
    error_message: Option<String>,
    font: Font<'static>,
    font_metrics: WindowFontMetrics,
    layout: Option<TerminalLayoutConfig>,
    resize: ResizeDebounce,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResizeDebounce {
    pending_size: Option<(u32, u32)>,
    settle_deadline: Option<Instant>,
}

impl ResizeDebounce {
    fn begin(&mut self, width: u32, height: u32, now: Instant) {
        self.pending_size = Some((width, height));
        self.settle_deadline = Some(now + RESIZE_SETTLE_DELAY);
    }

    fn is_active(self, now: Instant) -> bool {
        self.settle_deadline.is_some_and(|deadline| now < deadline)
    }

    fn take_settled(&mut self, now: Instant) -> Option<(u32, u32)> {
        if self.is_active(now) {
            return None;
        }

        self.settle_deadline = None;
        self.pending_size.take()
    }
}

impl WgpuApp {
    fn new() -> Result<Self, Box<dyn Error>> {
        let font = Font::new(WINDOW_FONT_DATA)
            .ok_or_else(|| format!("bundled {WINDOW_FONT_NAME} font could not be parsed"))?;
        Ok(Self {
            window: None,
            terminal: None,
            state: AppState::new(),
            error_message: None,
            font: font.clone(),
            font_metrics: WindowFontMetrics::from_bytes(WINDOW_FONT_DATA)?,
            layout: None,
            resize: ResizeDebounce::default(),
        })
    }

    fn set_error_and_exit(&mut self, event_loop: &ActiveEventLoop, message: impl Into<String>) {
        self.error_message = Some(message.into());
        event_loop.exit();
    }

    fn create_window_and_terminal(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) -> Result<(), Box<dyn Error>> {
        if self.window.is_some() {
            return Ok(());
        }

        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title(WINDOW_TITLE)
                    .with_maximized(true)
                    .with_inner_size(LogicalSize::new(
                        DEFAULT_WINDOW_WIDTH_PX,
                        DEFAULT_WINDOW_HEIGHT_PX,
                    )),
            )?,
        );

        self.window = Some(window);
        self.sync_terminal_layout(event_loop);
        Ok(())
    }

    fn apply_terminal_layout(&mut self, width: u32, height: u32, event_loop: &ActiveEventLoop) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if width == 0 || height == 0 {
            return;
        }

        let display = self.state.display_state();
        let desired = self
            .font_metrics
            .layout_for_window(width, height, display.show_diagnostics);

        if self.layout == Some(desired) {
            return;
        }

        match build_terminal(window, self.font.clone(), desired) {
            Ok(terminal) => {
                self.terminal = Some(terminal);
                self.layout = Some(desired);
            }
            Err(err) => {
                self.set_error_and_exit(event_loop, format!("wgpu layout failed: {err}"));
            }
        }
    }

    fn sync_terminal_layout(&mut self, event_loop: &ActiveEventLoop) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        self.apply_terminal_layout(size.width, size.height, event_loop);
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        if self.resize.is_active(Instant::now()) {
            return;
        }

        let Some(terminal) = self.terminal.as_mut() else {
            return;
        };

        if let Err(err) = self.state.draw(terminal) {
            self.set_error_and_exit(event_loop, format!("wgpu draw failed: {err}"));
            return;
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.resize.begin(width, height, Instant::now());
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: &Key) {
        if let Some(action) = action_from_key(key) {
            match self.state.apply_action(action) {
                Ok(should_quit) => {
                    if action == AppAction::ToggleDiagnostics {
                        self.layout = None;
                    }
                    if should_quit {
                        event_loop.exit();
                    }
                }
                Err(err) => {
                    self.set_error_and_exit(event_loop, format!("frontend action failed: {err}"));
                }
            }
        }
    }
}

impl ApplicationHandler for WgpuApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        if let Err(err) = self.create_window_and_terminal(event_loop) {
            self.set_error_and_exit(event_loop, format!("wgpu init failed: {err}"));
            return;
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    let action = action_from_key(&event.logical_key);
                    if event.repeat
                        && action.is_some_and(|action| !crate::action_allows_repeat(action))
                    {
                        return;
                    }
                    self.handle_key(event_loop, &event.logical_key);
                }
            }
            WindowEvent::RedrawRequested => {
                self.redraw(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.state.advance();

        let now = Instant::now();
        if self.resize.is_active(now) {
            return;
        }

        if let Some((width, height)) = self.resize.take_settled(now) {
            self.apply_terminal_layout(width, height, event_loop);
        } else if self.layout.is_none() {
            self.sync_terminal_layout(event_loop);
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn action_from_key(key: &Key) -> Option<AppAction> {
    match key {
        Key::Named(NamedKey::Escape) => Some(AppAction::Quit),
        Key::Named(NamedKey::Enter) => Some(AppAction::Confirm),
        Key::Named(NamedKey::Delete | NamedKey::Backspace) => Some(AppAction::DeleteSelection),
        Key::Named(NamedKey::Space) => Some(AppAction::TogglePause),
        Key::Named(NamedKey::ArrowRight) => Some(AppAction::IncreaseSpeed),
        Key::Named(NamedKey::ArrowLeft) => Some(AppAction::DecreaseSpeed),
        Key::Named(NamedKey::ArrowUp) => Some(AppAction::IncreaseDiversity),
        Key::Named(NamedKey::ArrowDown) => Some(AppAction::DecreaseDiversity),
        Key::Character(text) if text.eq_ignore_ascii_case("q") => Some(AppAction::Quit),
        Key::Character(text) if text.eq_ignore_ascii_case("d") => {
            Some(AppAction::ToggleDiagnostics)
        }
        Key::Character(text) if text.eq_ignore_ascii_case("f") => Some(AppAction::DropFood),
        Key::Character(text) if text.eq_ignore_ascii_case("r") => Some(AppAction::Reset),
        Key::Character(text) if text == "?" || text.eq_ignore_ascii_case("h") => {
            Some(AppAction::ToggleHelp)
        }
        Key::Character(text) if text.eq_ignore_ascii_case("p") => Some(AppAction::SaveScreenshot),
        Key::Character(text) if text.eq_ignore_ascii_case("s") => Some(AppAction::SaveSession),
        Key::Character(text) if text.eq_ignore_ascii_case("e") => Some(AppAction::ExportHistory),
        Key::Character(text) if text.eq_ignore_ascii_case("g") => Some(AppAction::ToggleRecording),
        Key::Character(text) if text.eq_ignore_ascii_case("t") => Some(AppAction::NextTheme),
        _ => None,
    }
}

fn build_terminal(
    window: Arc<Window>,
    font: Font<'static>,
    layout: TerminalLayoutConfig,
) -> Result<WindowTerminal, Box<dyn Error>> {
    let backend = block_on(
        Builder::from_font(font)
            .with_font_size_px(layout.font_size_px.max(1))
            .with_width_and_height(Dimensions {
                width: NonZeroU32::new(layout.window_width_px.max(1)).unwrap(),
                height: NonZeroU32::new(layout.window_height_px.max(1)).unwrap(),
            })
            .with_viewport(layout.viewport())
            .with_color_table(terminal_color_table())
            .with_fg_color(Color::White)
            .with_bg_color(Color::Black)
            .build_with_target(window),
    )?;
    let mut terminal = Terminal::new(backend)?;
    terminal.resize(ratatui::layout::Rect::new(
        0,
        0,
        layout.target_columns,
        layout.target_rows,
    ))?;
    Ok(terminal)
}

fn terminal_color_table() -> ColorTable {
    ColorTable {
        BLACK: [12, 12, 12],
        RED: [197, 15, 31],
        GREEN: [19, 161, 14],
        YELLOW: [193, 156, 0],
        BLUE: [0, 55, 218],
        MAGENTA: [136, 23, 152],
        CYAN: [58, 150, 221],
        GRAY: [204, 204, 204],
        DARKGRAY: [118, 118, 118],
        LIGHTRED: [231, 72, 86],
        LIGHTGREEN: [22, 198, 12],
        LIGHTYELLOW: [249, 241, 165],
        LIGHTBLUE: [59, 120, 255],
        LIGHTMAGENTA: [180, 0, 158],
        LIGHTCYAN: [97, 214, 214],
        WHITE: [242, 242, 242],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TerminalLayoutConfig {
    window_width_px: u32,
    window_height_px: u32,
    font_size_px: u32,
    inset_width_px: u32,
    inset_height_px: u32,
    target_columns: u16,
    target_rows: u16,
}

impl TerminalLayoutConfig {
    fn viewport(self) -> Viewport {
        if self.inset_width_px == 0 && self.inset_height_px == 0 {
            Viewport::Full
        } else {
            Viewport::Shrink {
                width: self.inset_width_px,
                height: self.inset_height_px,
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WindowFontMetrics {
    advance_units: f32,
    height_units: f32,
}

impl WindowFontMetrics {
    fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn Error>> {
        let face = Face::from_slice(data, 0)
            .ok_or_else(|| format!("bundled {WINDOW_FONT_NAME} metrics could not be parsed"))?;
        let advance_units = face
            .glyph_hor_advance(face.glyph_index('m').unwrap_or_default())
            .unwrap_or_default() as f32;
        let height_units = face.height() as f32;
        if advance_units <= 0.0 || height_units <= 0.0 {
            return Err(format!("bundled {WINDOW_FONT_NAME} returned invalid metrics").into());
        }
        Ok(Self {
            advance_units,
            height_units,
        })
    }

    fn char_width_px(self, font_size_px: u32) -> u32 {
        ((self.advance_units * font_size_px as f32 / self.height_units) as u32).max(1)
    }

    #[cfg(test)]
    fn logical_size_for_font(self, width_px: u32, height_px: u32, font_size_px: u32) -> (u32, u32) {
        let font_size_px = font_size_px.max(1);
        (
            width_px / self.char_width_px(font_size_px),
            height_px / font_size_px,
        )
    }

    fn font_fits(self, width_px: u32, height_px: u32, font_size_px: u32, target_rows: u16) -> bool {
        let char_width = self.char_width_px(font_size_px);
        WGPU_TARGET_COLUMNS as u32 * char_width <= width_px
            && target_rows as u32 * font_size_px.max(1) <= height_px
    }

    fn max_fitting_font_size(self, width_px: u32, height_px: u32, target_rows: u16) -> u32 {
        if width_px == 0 || height_px == 0 {
            return MIN_WINDOW_FONT_SIZE_PX;
        }

        let mut best = MIN_WINDOW_FONT_SIZE_PX;
        let mut low = MIN_WINDOW_FONT_SIZE_PX;
        let mut high = height_px.max(MIN_WINDOW_FONT_SIZE_PX);

        while low <= high {
            let mid = low + (high - low) / 2;
            if self.font_fits(width_px, height_px, mid, target_rows) {
                best = mid;
                low = mid.saturating_add(1);
            } else {
                high = mid.saturating_sub(1);
            }
        }

        best
    }

    fn layout_for_window(
        self,
        width_px: u32,
        height_px: u32,
        show_diagnostics: bool,
    ) -> TerminalLayoutConfig {
        let target_rows = if show_diagnostics {
            WGPU_DIAGNOSTIC_ROWS
        } else {
            WGPU_BASE_ROWS
        };
        let font_size_px = self.max_fitting_font_size(width_px, height_px, target_rows);
        let char_width_px = self.char_width_px(font_size_px);

        TerminalLayoutConfig {
            window_width_px: width_px,
            window_height_px: height_px,
            font_size_px,
            inset_width_px: width_px.saturating_sub(WGPU_TARGET_COLUMNS as u32 * char_width_px),
            inset_height_px: height_px.saturating_sub(target_rows as u32 * font_size_px.max(1)),
            target_columns: WGPU_TARGET_COLUMNS,
            target_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metrics() -> WindowFontMetrics {
        WindowFontMetrics {
            advance_units: 6.0,
            height_units: 10.0,
        }
    }

    #[test]
    fn test_target_layout_matches_fixed_world_plus_hud() {
        assert_eq!(WGPU_TARGET_COLUMNS, 138);
        assert_eq!(WGPU_BASE_ROWS, 49);
        assert_eq!(WGPU_DIAGNOSTIC_ROWS, 55);
    }

    #[test]
    fn test_terminal_color_table_matches_terminal_palette() {
        let colors = terminal_color_table();
        assert_eq!(colors.GREEN, [19, 161, 14]);
        assert_eq!(colors.YELLOW, [193, 156, 0]);
        assert_eq!(colors.GRAY, [204, 204, 204]);
        assert_eq!(colors.DARKGRAY, [118, 118, 118]);
    }

    #[test]
    fn test_max_fitting_font_size_grows_for_larger_window() {
        let metrics = test_metrics();
        let smaller = metrics.max_fitting_font_size(1280, 840, WGPU_BASE_ROWS);
        let larger = metrics.max_fitting_font_size(1920, 1080, WGPU_BASE_ROWS);
        assert!(larger > smaller);
    }

    #[test]
    fn test_max_fitting_font_size_shrinks_for_smaller_window() {
        let metrics = test_metrics();
        let smaller = metrics.max_fitting_font_size(980, 430, WGPU_DIAGNOSTIC_ROWS);
        let larger = metrics.max_fitting_font_size(1960, 860, WGPU_DIAGNOSTIC_ROWS);
        assert!(smaller < larger);
    }

    #[test]
    fn test_max_fitting_font_size_never_drops_below_minimum() {
        let metrics = test_metrics();
        assert_eq!(
            metrics.max_fitting_font_size(100, 50, WGPU_DIAGNOSTIC_ROWS),
            MIN_WINDOW_FONT_SIZE_PX
        );
    }

    #[test]
    fn test_resize_debounce_delays_application_until_settled() {
        let mut resize = ResizeDebounce::default();
        let start = Instant::now();
        resize.begin(1000, 700, start);

        assert!(resize.is_active(start + Duration::from_millis(60)));
        assert_eq!(resize.take_settled(start + Duration::from_millis(60)), None);
        assert_eq!(
            resize.take_settled(start + Duration::from_millis(120)),
            Some((1000, 700))
        );
    }

    #[test]
    fn test_resize_debounce_only_applies_latest_size() {
        let mut resize = ResizeDebounce::default();
        let start = Instant::now();
        resize.begin(1000, 700, start);
        resize.begin(900, 640, start + Duration::from_millis(40));

        assert_eq!(
            resize.take_settled(start + Duration::from_millis(159)),
            None
        );
        assert_eq!(
            resize.take_settled(start + Duration::from_millis(160)),
            Some((900, 640))
        );
        assert_eq!(
            resize.take_settled(start + Duration::from_millis(200)),
            None
        );
    }

    #[test]
    fn test_layout_for_window_matches_exact_terminal_shape_without_diagnostics() {
        let metrics = test_metrics();
        let layout = metrics.layout_for_window(1960, 860, false);
        let drawable_width = layout.window_width_px - layout.inset_width_px;
        let drawable_height = layout.window_height_px - layout.inset_height_px;
        let (columns, rows) =
            metrics.logical_size_for_font(drawable_width, drawable_height, layout.font_size_px);
        assert!(columns >= WGPU_TARGET_COLUMNS as u32);
        assert_eq!(rows, WGPU_BASE_ROWS as u32);
        assert_eq!(layout.target_rows, WGPU_BASE_ROWS);
    }

    #[test]
    fn test_layout_for_window_matches_exact_terminal_shape_with_diagnostics() {
        let metrics = test_metrics();
        let layout = metrics.layout_for_window(1960, 860, true);
        let drawable_width = layout.window_width_px - layout.inset_width_px;
        let drawable_height = layout.window_height_px - layout.inset_height_px;
        let (columns, rows) =
            metrics.logical_size_for_font(drawable_width, drawable_height, layout.font_size_px);
        assert!(columns >= WGPU_TARGET_COLUMNS as u32);
        assert_eq!(rows, WGPU_DIAGNOSTIC_ROWS as u32);
        assert_eq!(layout.target_rows, WGPU_DIAGNOSTIC_ROWS);
    }
}
