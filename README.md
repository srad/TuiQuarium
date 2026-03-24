<p align="center">
  <pre align="center">
    .---.         .---.           .---.
   / o o \       / o o \         / o o \
  (       ) =>  (       ) =>    (       )
   '-----'       '-----'         '-----'
    |\ /|         |/ \|           |\ /|
    | X |         | X |           | X |
    |/ \|         |\ /|           |/ \|
  </pre>
</p>

<h1 align="center">tuiquarium</h1>

<p align="center">
  <strong>A living, breathing ASCII aquarium in your terminal.</strong>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust&logoColor=white" alt="Rust"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Tests-72_passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Lines_of_Code-5k%2B-informational" alt="Lines of Code">
  <img src="https://img.shields.io/badge/Status-Alpha-yellow" alt="Status">
</p>

<p align="center">
  Neural network brains, genetic evolution, boids flocking, predator-prey ecosystems,<br>
  and a day/night cycle &mdash; all rendered as multi-line ASCII art in your terminal.
</p>

---

## Features

- **Neural network brains** &mdash; each creature has a 14&rarr;10&rarr;8&rarr;6 feedforward neural network (292 evolvable weights) that processes sensory inputs and drives steering, speed, and behavioral decisions
- **Large, detailed ASCII art creatures** &mdash; not single-character symbols, but multi-line art with swim cycles, tail swishes, fin waves, and idle breathing
- **Procedural generation** &mdash; creatures are built from an evolvable genome that controls body plan, fins, tail, eyes, colors, and fill patterns
- **Genetic evolution** &mdash; crossover, mutation, and speciation; offspring inherit blended traits *and* brain weights from parents
- **Morphology-driven fitness** &mdash; body shape *is* function: streamlined fish swim faster, large fish store more energy but have higher metabolism, bright colors attract mates *and* predators
- **Natural population regulation** &mdash; no artificial caps; food scarcity and metabolism create carrying capacity with boom/bust cycles
- **Boids flocking** &mdash; separation, alignment, cohesion with size-aware spacing and wall avoidance
- **Predator-prey ecosystem** &mdash; trophic roles (producer, herbivore, omnivore, carnivore), body-size ratio hunting, energy transfer
- **Realistic needs system** &mdash; hunger, safety, rest, social, reproduction, territory, curiosity
- **Day/night cycle** &mdash; sine-based lighting, palette shifts from bright day through dusk to dark night
- **Random events** &mdash; algae blooms, feeding frenzies, cold snaps, earthquakes
- **Allometric metabolism** &mdash; energy cost scales with mass^0.75 (real biological law)
- **Food decay** &mdash; uneaten food rots over time, preventing accumulation
- **Bubble effects** &mdash; cosmetic rising bubbles with wobble
- **HUD overlay** &mdash; creature count, births/deaths, day counter, time, temperature, light level, active events
- **Interactive controls** &mdash; pause, speed adjustment, manual food drops

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) 1.70 or later
- A terminal with color support (most modern terminals work)

### Build & Run

```bash
git clone https://github.com/srad/tuiquarium.git
cd tuiquarium
cargo run --release
```

### Run Tests

```bash
cargo test --workspace
```

## Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `Space` | Pause / Resume |
| `+` / `=` | Speed up (max 20x) |
| `-` | Slow down (min 0.5x) |
| `f` | Drop food |

## Architecture

tuiquarium strictly separates simulation from rendering through traits and dependency injection:

```
tuiquarium/
├── src/main.rs               # Entry point, game loop, DI wiring
├── crates/
│   ├── tuiq-core/            # Pure simulation logic (zero rendering deps)
│   │   ├── animation.rs      # Frame sequencing & timing
│   │   ├── behavior.rs       # Behavioral action types & speed multipliers
│   │   ├── boids.rs          # Boids flocking algorithm
│   │   ├── brain.rs          # Neural network brains, sensory input, brain_system
│   │   ├── components.rs     # ECS components (Position, Velocity, Appearance, ...)
│   │   ├── ecosystem.rs      # Energy, metabolism, hunting, death
│   │   ├── environment.rs    # Day/night cycle, temperature, currents, events
│   │   ├── genetics.rs       # Crossover, mutation, genomic distance
│   │   ├── genome.rs         # CreatureGenome: art + animation + behavior + brain genes
│   │   ├── needs.rs          # Realistic animal needs hierarchy
│   │   ├── phenotype.rs      # Genome -> physical stats derivation
│   │   ├── physics.rs        # Position integration, boundary handling
│   │   ├── spatial.rs        # Spatial hash grid for O(n) neighbor queries
│   │   └── spawner.rs        # Reproduction system
│   │
│   └── tuiq-render/          # Ratatui rendering (depends on tuiq-core)
│       ├── ascii.rs          # Procedural ASCII art from genome
│       ├── effects.rs        # Bubble particle system
│       ├── hud.rs            # Creature stats, time, controls overlay
│       ├── palette.rs        # Day/night color palette shifts
│       ├── tank.rs           # Tank border, water, substrate, creature rendering
│       └── templates.rs      # Pre-crafted creature art templates
```

### Design Principles

- **Simulation knows nothing about rendering.** The `Simulation` trait exposes read-only access to the ECS world. Rendering code never mutates simulation state.
- **ECS architecture** with [hecs](https://github.com/Ralith/hecs) &mdash; lightweight, no global state, manual system orchestration.
- **Fixed timestep game loop** (20 ticks/sec simulation, ~60fps rendering) with accumulator pattern.
- **Spatial hash grid** reduces neighbor queries from O(n&sup2;) to ~O(n).

## Neural Network Brains

Every creature has a small feedforward neural network that acts as its "brain." The network is evaluated each tick to produce steering forces and behavioral decisions.

### Network Architecture

```
Sensory Inputs (14)  -->  Hidden Layer 1 (10, tanh)  -->  Hidden Layer 2 (8, tanh)  -->  Outputs (6, tanh)
```

**292 total parameters** (weights + biases) &mdash; compact enough for real-time simulation of 50+ creatures.

### Sensory Inputs

| Input | Description |
|-------|-------------|
| Energy fraction | How full the creature's energy bar is (0&ndash;1) |
| Hunger | Current hunger need level (0&ndash;1) |
| Safety | Current safety/threat level (0&ndash;1) |
| Reproduction need | Urge to reproduce (0&ndash;1) |
| Nearest food distance | Proximity to closest edible target (0=far, 1=close) |
| Nearest food angle | Direction to food (&minus;1 to +1) |
| Nearest predator distance | Proximity to closest threat |
| Nearest predator angle | Direction of threat |
| Nearest ally distance | Proximity to same-species neighbor |
| Nearest ally angle | Direction to ally |
| Wall proximity X | Distance to left/right walls (&minus;1 to +1) |
| Wall proximity Y | Distance to top/bottom walls |
| Light level | Current ambient light (day/night cycle) |
| Own speed | Current speed as fraction of max |

### Outputs

| Output | Effect |
|--------|--------|
| Steer X, Y | Steering force direction |
| Speed multiplier | How fast to move (0.1x&ndash;1.5x) |
| Forage tendency | Drives food-seeking behavior |
| Flee tendency | Drives predator avoidance |
| Social tendency | Drives schooling behavior |

### Evolution

Brain weights are part of the genome. During reproduction:
- **Crossover**: each weight is randomly selected from one parent (uniform crossover)
- **Mutation**: weights are perturbed by &plusmn;0.3 with 15% probability, clamped to [&minus;3, +3]
- **Natural selection**: creatures that can't find food starve; effective brains propagate

Initial creatures have a positive bias on the forage output neuron, giving them an innate food-seeking drive that evolution can refine or override.

## Algorithms

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **Neural network brains** | `brain.rs` | Fixed-topology feedforward net (14&rarr;10&rarr;8&rarr;6) with tanh activation |
| **Boids flocking** | `boids.rs` | Craig Reynolds (1986) &mdash; separation, alignment, cohesion + wall avoidance |
| **Genetic evolution** | `genetics.rs` | Uniform crossover, Gaussian mutation, genomic distance for speciation |
| **Allometric scaling** | `phenotype.rs` | Metabolism ~ mass^0.75 (Kleiber's law) |
| **Predator-prey dynamics** | `ecosystem.rs` | Body size ratio + speed check hunting, trophic energy transfer |
| **Spatial hashing** | `spatial.rs` | Grid-based bucketing for efficient neighbor lookups |

## Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| [hecs](https://crates.io/crates/hecs) | 0.11 | Lightweight ECS |
| [ratatui](https://crates.io/crates/ratatui) | 0.29 | TUI rendering with double-buffered diffing |
| [crossterm](https://crates.io/crates/crossterm) | 0.28 | Cross-platform terminal I/O |
| [rand](https://crates.io/crates/rand) | 0.10 | RNG for procedural generation + simulation |

## Creature Types

| Creature | Size | Animation | Behavior |
|----------|------|-----------|----------|
| Small schooling fish | 10x4 | Tail swish, fin wave | Schools (boids), herbivore |
| Tropical fish | 18x7 | Full body undulation | Semi-schools, omnivore |
| Angelfish | 12x9 | Fin glide | Solitary/pairs, omnivore |
| Jellyfish | 12x12 | Bell pulse, tentacle wave | Passive drifter, carnivore |
| Crab | 16x7 | Walk cycle, claw snap | Bottom walker, omnivore |
| Seaweed | 5-8x6-15 | Sway animation | Rooted producer |

## Simulation Systems (per tick)

The ECS systems run in this order every tick:

1. **Environment** &mdash; advance time, light, temperature, currents, events
2. **Spatial grid rebuild** &mdash; refresh neighbor lookup structure
3. **Needs update** &mdash; hunger rises, safety decays, social drifts
4. **Brain system** &mdash; neural network processes sensory inputs, produces steering forces and actions
5. **Boids flocking** &mdash; compute social steering forces for schooling creatures
6. **Physics** &mdash; integrate velocities, enforce boundaries, update facing
7. **Metabolism + aging** &mdash; drain energy, increment age
8. **Hunting** &mdash; predator-prey interactions, energy transfer
9. **Reproduction** &mdash; mate pairing, genome crossover, offspring spawning
10. **Death** &mdash; remove starved/aged entities, food decay cleanup
11. **Animation** &mdash; advance frame timers, cycle animation frames

## Energy Economy

Population is naturally regulated by food scarcity, not artificial caps:

| Parameter | Value | Effect |
|-----------|-------|--------|
| Metabolism rate | mass^0.75 &times; 0.5 | Creatures burn energy continuously |
| Food pellet energy | 10 | Sustains a creature for ~30 seconds |
| Food spawn interval | 5 seconds | Limited food entering the system |
| Food decay | Rots over ~50 seconds | Uneaten food disappears |
| Plant photosynthesis | +0.25 energy/sec | Slow renewable food source |
| Reproduction cost | 40% of max energy per parent | Expensive investment |
| Offspring energy | 30% of max | Born hungry, must find food fast |

More creatures = more competition for food = starvation = fewer creatures. Fewer creatures = more food per individual = survival = reproduction. This creates natural boom/bust population cycles.

## Evolutionary Tradeoffs

Body morphology creates real tradeoffs, just like in nature:

| Trait | Benefit | Cost |
|-------|---------|------|
| Large body | More energy storage, stronger | Higher metabolism, slower, bigger target |
| Streamlined (Slim) | Faster, lower drag | Less energy storage |
| Bright colors | Attracts mates | Attracts predators |
| Large tail | Higher acceleration | More energy per movement |
| Large eyes | Better sensory range | Slight metabolism cost |

## Roadmap

- [x] Neural network brains &mdash; evolved feedforward networks drive creature behavior
- [x] Natural population regulation &mdash; food-limited carrying capacity
- [x] Food decay &mdash; uneaten food rots over time
- [ ] Save/load simulation state (serde serialization)
- [ ] More creature types (shark, seahorse, octopus, turtle, starfish, snail)
- [ ] Layered ASCII art compositor for richer procedural generation
- [ ] NEAT topology evolution &mdash; evolve network structure, not just weights
- [ ] Plant growth system with logistic curves
- [ ] Nutrient recycling from dead organisms
- [ ] Terminal resize handling
- [ ] Configuration file for simulation parameters
- [ ] Debug overlay (toggle with `d`)

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure `cargo test --workspace` passes before submitting.

## References

- Craig Reynolds, [Flocks, Herds, and Schools: A Distributed Behavioral Model](https://www.red3d.com/cwr/papers/1987/SIGGRAPH87.pdf) (SIGGRAPH 1987)
- [Steering Behaviors for Autonomous Characters](https://slsdo.github.io/steering-behaviors/)
- [Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) for predator-prey dynamics
- [Kleiber's law](https://en.wikipedia.org/wiki/Kleiber%27s_law) for allometric metabolic scaling
- [Wa-Tor simulation](https://beltoforion.de/en/wator/) for population dynamics inspiration
- [NEAT](https://www.cs.ucf.edu/~kstanley/neat.html) for neuroevolution concepts

## License

This project is licensed under the MIT License &mdash; see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with Rust, ratatui, and a love for virtual fish.</sub>
</p>
