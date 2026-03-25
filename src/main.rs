use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use tuiq_core::components::*;
use tuiq_core::genome::{CreatureGenome, PlantGenome};
use tuiq_core::{AquariumSim, Simulation};
use tuiq_render::ascii::generate_frames;
use tuiq_render::{DisplayState, TuiRenderer};

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {err}");
    }

    Ok(())
}

/// Spawn a creature from a genome, generating proper ASCII art from the render crate.
fn spawn_with_art(sim: &mut AquariumSim, genome: CreatureGenome, x: f32, y: f32) {
    let (swim_frames, idle_frames) = generate_frames(&genome);
    let w = swim_frames[0].width as f32;
    let h = swim_frames[0].height as f32;

    let color_index = genome.art.color_index();
    let appearance = Appearance {
        frame_sets: vec![swim_frames, idle_frames],
        facing: Direction::Right,
        color_index,
    };

    let entity = sim.spawn_from_genome(genome, x, y);

    // Replace the placeholder appearance with the real generated art
    let _ = sim.world_mut().insert(entity, (
        appearance,
        BoundingBox { w, h },
    ));
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let size = terminal.size()?;
    let tank_w = size.width.saturating_sub(2);
    let tank_h = size.height.saturating_sub(4);
    let mut sim = AquariumSim::new(tank_w, tank_h);
    let mut renderer = TuiRenderer::new();

    let tw = tank_w as f32;
    let th = tank_h as f32;

    let mut rng = rand::rng();

    // Spawn ~15 primordial cells
    for _ in 0..15 {
        let genome = CreatureGenome::minimal_cell(&mut rng);
        let x = rand::RngExt::random_range(&mut rng, 2.0..(tw - 2.0));
        let y = rand::RngExt::random_range(&mut rng, 2.0..(th - 4.0));
        spawn_with_art(&mut sim, genome, x, y);
    }

    // Spawn plants scaled to tank area (roughly 1 per 150 cells, min 12)
    // Each plant gets a random genome — evolution will shape them over time
    let plant_count = ((tw * th / 150.0) as usize).max(12);
    for i in 0..plant_count {
        let x = 3.0 + (i as f32 / plant_count as f32) * (tw - 6.0);
        let y = th * 0.5 + (i as f32 % 4.0) * 2.5;
        let genome = PlantGenome::minimal_plant(&mut rng);
        sim.spawn_plant(
            Position { x, y },
            genome,
        );
    }

    let tick_rate = Duration::from_millis(50);
    let mut last_tick = Instant::now();
    let mut accumulator = Duration::ZERO;

    let mut paused = false;
    let mut speed = 1.0_f32;

    loop {
        let display = DisplayState {
            paused,
            speed,
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
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            speed = (speed + 0.5).min(20.0);
                        }
                        KeyCode::Char('-') => {
                            speed = (speed - 0.5).max(0.5);
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

            // Regenerate ASCII art for any creatures born this frame
            for entity in sim.drain_births() {
                let genome_copy: Option<CreatureGenome> = sim
                    .world_mut()
                    .get::<&CreatureGenome>(entity)
                    .ok()
                    .map(|g| (*g).clone());
                if let Some(gc) = genome_copy {
                    let (swim_frames, idle_frames) = generate_frames(&gc);
                    let w = swim_frames[0].width as f32;
                    let h = swim_frames[0].height as f32;
                    let color_index = gc.art.color_index();
                    let appearance = Appearance {
                        frame_sets: vec![swim_frames, idle_frames],
                        facing: Direction::Right,
                        color_index,
                    };
                    let _ = sim.world_mut().insert(entity, (
                        appearance,
                        BoundingBox { w, h },
                    ));
                }
            }
        } else {
            last_tick = Instant::now();
        }
    }
}
