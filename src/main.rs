use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use tuiq_core::components::*;
use tuiq_core::genome::CreatureGenome;
use tuiq_core::{AquariumSim, Simulation};
use tuiq_render::ascii::generate_frames;
use tuiq_render::{DisplayState, Renderer, RenderTheme, TuiRenderer};

const MIN_SIM_SPEED: f32 = 0.5;
const MAX_SIM_SPEED: f32 = 100.0;
const SIM_SPEED_STEP: f32 = 0.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        return Err(format!("unknown arguments: {}", args.join(" ")).into());
    }
    run_tui()?;
    Ok(())
}

fn increase_sim_speed(speed: f32) -> f32 {
    (speed + SIM_SPEED_STEP).min(MAX_SIM_SPEED)
}

fn decrease_sim_speed(speed: f32) -> f32 {
    (speed - SIM_SPEED_STEP).max(MIN_SIM_SPEED)
}

fn run_tui() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = &result {
        eprintln!("Error: {err}");
    }

    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let size = terminal.size()?;
    let tank_w = size.width.saturating_sub(2);
    let tank_h = size.height.saturating_sub(5);
    let mut sim = AquariumSim::new(tank_w, tank_h);
    let mut renderer = TuiRenderer::new();
    let bootstrap_creatures = sim.bootstrap_founder_web();
    let attach_creature_art = |sim: &mut AquariumSim, entity| {
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
    };

    for entity in bootstrap_creatures {
        attach_creature_art(&mut sim, entity);
    }

    let tw = tank_w as f32;

    let tick_rate = Duration::from_millis(50);
    let mut last_tick = Instant::now();
    let mut accumulator = Duration::ZERO;

    let mut paused = false;
    let mut speed = 1.0_f32;
    let mut show_diagnostics = false;
    let mut show_help = false;
    let mut theme = RenderTheme::Ocean;

    loop {
        let display = DisplayState {
            paused,
            speed,
            show_diagnostics,
            show_help,
            is_recording: renderer.is_recording(),
            recording_secs: renderer.recording_elapsed_secs(),
            theme,
        };

        terminal.draw(|frame| {
            renderer.render_with_hud(frame, &sim, &display);
        })?;

        let timeout = Duration::from_millis(16);
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char(' ') => paused = !paused,
                        KeyCode::Right => {
                            speed = increase_sim_speed(speed);
                        }
                        KeyCode::Left => {
                            speed = decrease_sim_speed(speed);
                        }
                        KeyCode::Char('d') => {
                            show_diagnostics = !show_diagnostics;
                        }
                        KeyCode::Char('f') => {
                            let food_x = (sim.stats().tick_count as f32 * 7.3) % tw;
                            let frame = AsciiFrame::from_rows(vec!["*"]);
                            let appearance = Appearance {
                                frame_sets: vec![vec![frame.clone()], vec![frame]],
                                facing: Direction::Right,
                                color_index: 1,
                            };
                            sim.spawn_food(
                                Position { x: food_x, y: 0.0 },
                                Velocity { vx: 0.0, vy: 1.5 },
                                BoundingBox { w: 1.0, h: 1.0 },
                                appearance,
                            );
                        }
                        KeyCode::Down => {
                            let cur = sim.diversity_coefficient();
                            sim.set_diversity_coefficient(cur - 0.1);
                        }
                        KeyCode::Up => {
                            let cur = sim.diversity_coefficient();
                            sim.set_diversity_coefficient(cur + 0.1);
                        }
                        KeyCode::Char('r') => {
                            speed = 1.0;
                            sim.set_diversity_coefficient(1.0);
                        }
                        KeyCode::Char('?') | KeyCode::Char('h') => {
                            show_help = !show_help;
                        }
                        KeyCode::Char('p') => {
                            let ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            match tuiq_render::screenshot::screenshots_dir() {
                                Ok(dir) => {
                                    let path = dir.join(format!("tuiquarium_{ts}.png"));
                                    match renderer.save_screenshot(&path) {
                                        Ok(()) => renderer.flash_message(format!(
                                            "Screenshot saved to {}",
                                            path.display()
                                        )),
                                        Err(e) => renderer.flash_message(format!(
                                            "Screenshot failed: {e}"
                                        )),
                                    }
                                }
                                Err(e) => renderer.flash_message(format!(
                                    "Cannot create screenshot dir: {e}"
                                )),
                            }
                        }
                        KeyCode::Char('g') => {
                            if renderer.is_recording() {
                                if let Some((path, count)) = renderer.stop_recording() {
                                    renderer.flash_message(format!(
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
                                        let path =
                                            dir.join(format!("tuiquarium_{ts}.gif"));
                                        match renderer.start_recording(path) {
                                            Ok(()) => {}
                                            Err(e) => renderer.flash_message(format!(
                                                "Recording failed: {e}"
                                            )),
                                        }
                                    }
                                    Err(e) => renderer.flash_message(format!(
                                        "Cannot create recordings dir: {e}"
                                    )),
                                }
                            }
                        }
                        KeyCode::Char('t') => {
                            theme = theme.next();
                            renderer.flash_message(format!(
                                "Theme: {}",
                                theme.label()
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }

        if !paused {
            let now = Instant::now();
            accumulator += now - last_tick;
            last_tick = now;

            while accumulator >= tick_rate {
                sim.tick(tick_rate.as_secs_f32() * speed);
                accumulator -= tick_rate;
            }

            for entity in sim.drain_births() {
                attach_creature_art(&mut sim, entity);
            }
        } else {
            last_tick = Instant::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
