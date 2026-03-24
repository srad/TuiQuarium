use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use tuiq_core::components::*;
use tuiq_core::brain::BrainGenome;
use tuiq_core::genome::{
    AnimGenome, ArtGenome, BehaviorGenome, BodyPlan, CreatureGenome, DietType, EyeStyle,
    FillPattern, TailStyle,
};
use tuiq_core::{AquariumSim, Simulation};
use tuiq_render::templates::{CreatureTemplate, get_template_frames};
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

fn spawn_creature_from_template(
    sim: &mut AquariumSim,
    template: CreatureTemplate,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
) {
    let (swim_frames, idle_frames) = get_template_frames(template);
    let w = swim_frames[0].width as f32;
    let h = swim_frames[0].height as f32;

    let color_index = match template {
        CreatureTemplate::SmallFish => 0,
        CreatureTemplate::TropicalFish => 1,
        CreatureTemplate::Angelfish => 2,
        CreatureTemplate::Jellyfish => 3,
        CreatureTemplate::Crab => 4,
        CreatureTemplate::Seaweed => 5,
    };
    let appearance = Appearance {
        frame_sets: vec![swim_frames, idle_frames],
        facing: if vx >= 0.0 {
            Direction::Right
        } else {
            Direction::Left
        },
        color_index,
    };

    // For Seaweed, spawn as a plant (producer)
    if matches!(template, CreatureTemplate::Seaweed) {
        sim.spawn_plant(
            Position { x, y },
            BoundingBox { w, h },
            appearance,
            AnimationState::new(0.5), // Slower animation for plants
        );
        return;
    }

    // For creatures, construct a genome that matches the template style roughly
    let (diet, body_plan, tail_style, size, speed, schooling) = match template {
        CreatureTemplate::SmallFish => (
            DietType::Herbivore,
            BodyPlan::Slim,
            TailStyle::Forked,
            0.8,
            1.5,
            0.9,
        ),
        CreatureTemplate::TropicalFish => (
            DietType::Omnivore,
            BodyPlan::Flat,
            TailStyle::Fan,
            1.0,
            1.2,
            0.6,
        ),
        CreatureTemplate::Angelfish => (
            DietType::Omnivore,
            BodyPlan::Tall,
            TailStyle::Flowing,
            1.2,
            1.0,
            0.4,
        ),
        CreatureTemplate::Jellyfish => (
            DietType::Carnivore,
            BodyPlan::Round,
            TailStyle::Flowing,
            1.1,
            0.6,
            0.1,
        ),
        CreatureTemplate::Crab => (
            DietType::Omnivore,
            BodyPlan::Round,
            TailStyle::Pointed,
            0.9,
            0.8,
            0.2,
        ),
        CreatureTemplate::Seaweed => unreachable!(),
    };

    let genome = CreatureGenome {
        art: ArtGenome {
            body_plan,
            body_size: size,
            tail_style,
            tail_length: 1.0,
            has_dorsal_fin: true,
            has_pectoral_fins: true,
            fill_pattern: FillPattern::Solid,
            eye_style: EyeStyle::Dot,
            primary_color: color_index,
            secondary_color: (color_index + 1) % 8,
            color_brightness: 0.8,
        },
        anim: AnimGenome {
            swim_speed: speed,
            tail_amplitude: 0.8,
            idle_sway: 0.5,
            undulation: 0.5,
        },
        behavior: BehaviorGenome {
            schooling_affinity: schooling,
            aggression: if diet == DietType::Carnivore { 0.8 } else { 0.2 },
            timidity: if diet == DietType::Herbivore { 0.7 } else { 0.3 },
            speed_factor: 1.0,
            metabolism_factor: 1.0,
            diet,
            max_lifespan_factor: 1.0,
            reproduction_rate: 0.5,
        },
        brain: BrainGenome::random(&mut rand::rng()),
    };

    sim.spawn_full_creature(
        Position { x, y },
        Velocity { vx, vy },
        BoundingBox { w, h },
        appearance,
        AnimationState::new(0.2),
        genome,
    );
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let size = terminal.size()?;
    // Reserve 2 rows for HUD at bottom
    let tank_w = size.width.saturating_sub(2);
    let tank_h = size.height.saturating_sub(4);
    let mut sim = AquariumSim::new(tank_w, tank_h);
    let mut renderer = TuiRenderer::new();

    let tw = tank_w as f32;
    let th = tank_h as f32;

    // Spawn a diverse set of creatures
    spawn_creature_from_template(&mut sim, CreatureTemplate::TropicalFish, tw * 0.2, th * 0.3, 3.5, 0.5);
    spawn_creature_from_template(&mut sim, CreatureTemplate::TropicalFish, tw * 0.6, th * 0.2, -2.5, 0.3);
    spawn_creature_from_template(&mut sim, CreatureTemplate::SmallFish, tw * 0.3, th * 0.5, 4.0, -0.8);
    spawn_creature_from_template(&mut sim, CreatureTemplate::SmallFish, tw * 0.35, th * 0.52, 3.8, -0.6);
    spawn_creature_from_template(&mut sim, CreatureTemplate::SmallFish, tw * 0.32, th * 0.48, 4.2, -0.5);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Angelfish, tw * 0.7, th * 0.4, -1.5, 0.4);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Jellyfish, tw * 0.5, th * 0.1, 0.3, 0.5);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Crab, tw * 0.4, th * 0.82, 1.0, 0.0);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Seaweed, tw * 0.1, th * 0.65, 0.0, 0.0);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Seaweed, tw * 0.3, th * 0.70, 0.0, 0.0);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Seaweed, tw * 0.5, th * 0.68, 0.0, 0.0);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Seaweed, tw * 0.7, th * 0.65, 0.0, 0.0);
    spawn_creature_from_template(&mut sim, CreatureTemplate::Seaweed, tw * 0.9, th * 0.70, 0.0, 0.0);

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
                            // Drop food from a random-ish position at the top
                            let food_x = (sim.stats().tick_count as f32 * 7.3) % tw;
                            let frame = AsciiFrame::from_rows(vec!["*"]);
                            let appearance = Appearance {
                                frame_sets: vec![vec![frame.clone()], vec![frame]],
                                facing: Direction::Right,
                                color_index: 1, // yellow
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
        } else {
            last_tick = Instant::now();
        }
    }
}
