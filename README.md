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
  <strong>An evolving ASCII aquarium in your terminal.</strong>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust&logoColor=white" alt="Rust"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Tests-148_passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Alpha-yellow" alt="Status">
</p>

<p align="center">
  No predefined species. No artificial food drops. Every creature emerges from a<br>
  continuous genome through mutation, crossover, and natural selection &mdash; rendered<br>
  as procedural ASCII art &mdash; sustained entirely by plant photosynthesis.
</p>

---

## Features

### Evolution &amp; Genetics

- **Bottom-up emergent evolution** &mdash; no templates or predefined species; all creatures emerge from a continuous genome through mutation and selection
- **Continuous genome** &mdash; 35+ float genes control body plan, appendages, eyes, coloring, behavior, and brain topology
- **Complexity as a master gate** &mdash; a single 0.0&ndash;1.0 gene controls which morphological features are expressed; always mutates on reproduction (±0.15 drift) ensuring continuous exploration
- **Asexual &amp; sexual reproduction** &mdash; simple cells divide asexually; complex creatures find compatible mates via genomic distance
- **Smooth complexity transitions** &mdash; gradual shift from asexual to sexual reproduction prevents evolutionary traps
- **Complexity rewards** &mdash; higher complexity grants +50% sensory range and ~25% metabolism efficiency
- **Evolvable mutation rate** &mdash; `mutation_rate_factor` gene (0.5&ndash;2.0) scales the base mutation rate; meta-evolution tunes evolvability itself
- **Sexual selection** &mdash; `mate_preference_hue` gene drives Fisherian runaway selection by preferring mates whose color matches the preference
- **Fitness sharing** &mdash; NEAT-style speciation protection; larger species have proportionally less reproduction chance, protecting novel innovations

### Neural Network Brains

- **NEAT-style evolving topology** &mdash; networks start minimal (16 inputs &rarr; 7 outputs, 112 connections) and grow via structural mutations that add hidden nodes and connections
- **Innovation numbers** &mdash; historical gene markings enable meaningful crossover between creatures with different network topologies
- **Hebbian lifetime learning** &mdash; evolvable `learning_rate` gene (0.0&ndash;0.1) allows weight plasticity during a creature's life; learned changes are NOT inherited (Baldwin Effect)
- **Sensory inputs** &mdash; energy, hunger, nearby food/predators/allies, walls, light, speed, pheromone concentration &amp; gradient
- **Behavioral outputs** &mdash; steering, speed, foraging, fleeing, schooling, pheromone emission
- **Evolved, not designed** &mdash; both weights and topology are part of the genome and evolve through crossover and mutation

### Pheromone Signaling

- **Chemical communication** &mdash; creatures deposit pheromones at their position; evolvable `pheromone_sensitivity` gene controls response strength
- **Grid-based diffusion** &mdash; pheromone concentrations spread to neighboring cells (5% per tick) and decay (×0.95 per tick, ~3.5s half-life)
- **Gradient sensing** &mdash; neural network receives local concentration and directional gradient as inputs, enabling trail following and collective behavior
- **Evolved, not designed** &mdash; pheromone emission is a neural network output; whether and how creatures use chemical signals emerges through evolution

### Self-Sustaining Ecosystem

- **Plant photosynthesis** &mdash; plants generate energy from light (negative metabolism), forming the base of the food web
- **Partial grazing** &mdash; creatures drain 30% of a plant's energy per feeding with complexity-scaled efficiency (50&ndash;100%); plants survive and regenerate
- **Logistic plant growth** &mdash; photosynthesis follows a logistic curve peaking at 50% population fill, preventing monocultures
- **Plant reproduction** &mdash; mature plants (&gt;70% energy) seed new plants nearby every ~5 seconds, capped by tank area
- **No artificial food rain** &mdash; the ecosystem sustains itself entirely through photosynthesis; manual food drops still available via `f` key
- **Nutrient cycling** &mdash; dead creatures become detritus entities (50% of max energy), which decay slowly and can be grazed
- **Depth zones** &mdash; surface (top 30%) has full light; mid-water has 70% light; deep zone (bottom 30%) has 40% light and 15% slower metabolism
- **Night metabolism stress** &mdash; creatures burn 30% more energy at night, favoring complex creatures with metabolism efficiency bonuses
- **Temperature effects** &mdash; Q10-based metabolic scaling affects both creatures and plants; cold snaps (&minus;10°C) create strong selection pressure
- **Morphology-driven feeding** &mdash; `FeedingCapability` derived from mouth size, aggression, and hunting instinct; full spectrum from grazers to apex predators emerges naturally
- **Predator-prey dynamics** &mdash; body-size ratio hunting, speed advantage checks, energy transfer on kill

### Simulation

- **Procedural ASCII art** &mdash; 4 complexity tiers (cells, simple, medium, complex) generated from genome; no hardcoded creature art
- **Boids flocking** &mdash; separation, alignment, cohesion with size-aware spacing and wall avoidance
- **Allometric metabolism** &mdash; energy cost scales with mass^0.75 (Kleiber's law)
- **Day/night cycle** &mdash; sine-based lighting, palette shifts from bright day through dusk to dark night
- **Random events** &mdash; algae blooms, feeding frenzies, cold snaps (&minus;10°C), earthquakes (every ~60s)
- **Multi-threaded** &mdash; brain, boids, and hunting systems parallelized with rayon
- **Soft population cap** &mdash; reproduction suppressed above 600 creatures to maintain responsiveness
- **HUD overlay** &mdash; population, generation, complexity, species count, births/deaths, day, time, temperature, light, speed

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

The simulation starts with **15 primordial cells** and **plants scaled to tank size** (roughly 1 per 150 cells, minimum 12). Creatures evolve from there.

### Run Tests

```bash
cargo test --workspace    # 148 tests (143 core + 5 render)
```

## Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `Space` | Pause / Resume |
| `+` / `=` | Speed up (0.5x increments, max 20x) |
| `-` | Slow down (0.5x decrements, min 0.5x) |
| `f` | Drop food pellet (manual only; no auto-spawning) |

## Architecture

tuiquarium strictly separates simulation from rendering through traits and dependency injection:

```
tuiquarium/
├── src/main.rs               # Entry point, game loop, input, DI wiring
├── crates/
│   ├── tuiq-core/            # Pure simulation logic (zero rendering deps)
│   │   ├── animation.rs      # Frame sequencing & timing
│   │   ├── behavior.rs       # Behavioral action types & speed multipliers
│   │   ├── boids.rs          # Boids flocking (rayon-parallelized)
│   │   ├── brain.rs          # NEAT neural network brains (rayon-parallelized)
│   │   ├── components.rs     # ECS components (Position, Velocity, Appearance, ...)
│   │   ├── ecosystem.rs      # Energy, metabolism, grazing/hunting, death
│   │   ├── environment.rs    # Day/night cycle, temperature, currents, events
│   │   ├── genetics.rs       # Crossover, mutation, genomic distance
│   │   ├── genome.rs         # CreatureGenome: art + anim + behavior + brain genes
│   │   ├── needs.rs          # Hunger, safety, reproduction needs
│   │   ├── phenotype.rs      # Genome → physical stats (FeedingCapability, DerivedPhysics)
│   │   ├── pheromone.rs      # Chemical signaling grid (deposit, diffusion, decay)
│   │   ├── physics.rs        # Position integration, boundary handling
│   │   ├── spatial.rs        # Spatial hash grid with distance-filtered queries
│   │   └── spawner.rs        # Asexual/sexual reproduction system
│   │
│   └── tuiq-render/          # Ratatui rendering (depends on tuiq-core)
│       ├── ascii.rs          # Procedural ASCII art generation from genome
│       ├── effects.rs        # Bubble particle system
│       ├── hud.rs            # Stats overlay (pop, gen, complexity, species, ...)
│       ├── palette.rs        # Day/night color palette shifts
│       └── tank.rs           # Tank border, water, substrate, creature rendering
```

### Design Principles

- **Simulation knows nothing about rendering.** The `Simulation` trait exposes read-only access to the ECS world. Rendering code never mutates simulation state.
- **ECS architecture** with [hecs](https://github.com/Ralith/hecs) &mdash; lightweight, no global state, manual system orchestration.
- **Fixed timestep game loop** &mdash; 50ms ticks (20 ticks/sec simulation), ~60fps rendering, accumulator pattern.
- **Spatial hash grid** reduces neighbor queries from O(n&sup2;) to ~O(n).
- **Shared entity info map** built once per tick, passed to brain/boids/hunting to eliminate redundant world queries and HashMap constructions.

## Neural Network Brains

Every creature has an evolving neural network (NEAT-style) evaluated each tick. Networks start minimal and grow via structural mutations.

### Network Architecture

```
Sensory Inputs (16)  →  [Evolving Hidden Topology]  →  Outputs (7, tanh)
```

Networks begin as direct input→output connections (112 weights) and grow via two structural mutations:

- **Add node** (3% per generation): splits an existing connection, inserting a hidden neuron
- **Add connection** (5% per generation): adds a new feedforward connection between non-connected nodes

Maximum topology: 60 nodes total, 300 connections. **Innovation numbers** track each structural mutation's history, enabling meaningful crossover between creatures with different topologies.

**Output scaling:** all outputs are scaled by `complexity.max(0.3)`, ensuring even simple creatures can take meaningful actions while complex creatures have full control authority.

**Hebbian learning:** an evolvable `learning_rate` gene (0.0–0.1) controls lifetime weight plasticity. When connected neurons co-activate, connections strengthen slightly. Learned weights are NOT inherited — only innate genome weights evolve (Baldwin Effect).

### Sensory Inputs

| # | Input | Description |
|---|-------|-------------|
| 1 | Energy fraction | How full the creature's energy bar is (0&ndash;1) |
| 2 | Hunger | Current hunger need level (0&ndash;1) |
| 3 | Safety | Current safety/threat level (0&ndash;1) |
| 4 | Reproduction need | Urge to reproduce (0&ndash;1) |
| 5 | Nearest food distance | Proximity to closest edible target (0=far, 1=close) |
| 6 | Nearest food angle | Direction to food (&minus;1 to +1, atan2/&pi;) |
| 7 | Nearest predator distance | Proximity to closest threat |
| 8 | Nearest predator angle | Direction of threat |
| 9 | Nearest ally distance | Proximity to similar-sized neighbor |
| 10 | Nearest ally angle | Direction to ally |
| 11 | Wall proximity X | Distance to left/right walls (&minus;1 to +1) |
| 12 | Wall proximity Y | Distance to top/bottom walls |
| 13 | Light level | Current ambient light (day/night cycle, 0&ndash;1) |
| 14 | Own speed | Current speed as fraction of max (0&ndash;1) |
| 15 | Pheromone concentration | Local pheromone level at creature's position (0&ndash;1) |
| 16 | Pheromone gradient | Directional pheromone gradient for trail following (&minus;1 to +1) |

### Behavioral Outputs

| Output | Effect |
|--------|--------|
| Steer X, Y | Steering force direction |
| Speed multiplier | How fast to move (0.1x&ndash;1.5x) |
| Forage tendency | Drives food-seeking behavior |
| Flee tendency | Drives predator avoidance |
| Social tendency | Drives schooling/flocking behavior |
| Pheromone emission | Deposits chemical signal at creature's position |

**Decision logic:** flee &gt; 0.3 with predator nearby &rarr; Flee; forage &gt; 0.3 with food nearby &rarr; Forage; social &gt; 0.3 &rarr; School; speed &lt; 0.3 &rarr; Rest; else &rarr; Explore.

## Evolution

### Genome Structure

All genes are continuous floats. There are no discrete categories or predefined species:

| Gene Group | Genes | Range |
|-----------|-------|-------|
| **Art** | body elongation, height ratio, size, tail fork/length, top/side appendages, pattern density, eye size, primary/secondary hue, brightness | 0&ndash;2 (varies) |
| **Animation** | swim speed, tail amplitude, idle sway, undulation | 0&ndash;2 |
| **Behavior** | schooling, aggression, timidity, speed factor, metabolism factor, lifespan factor, reproduction rate, mouth size, hunting instinct, mutation rate factor, mate preference hue, learning rate, pheromone sensitivity | 0&ndash;2 (varies) |
| **Brain** | NEAT genome: variable-length connection genes with innovation numbers, evolving topology | weights &minus;3 to +3 |
| **Complexity** | master gate controlling feature expression | 0.0&ndash;1.0 |
| **Generation** | inherited from parents + 1 | 0&ndash;&infin; |

### Primordial Cells

The simulation begins with `minimal_cell()` organisms:

- **Complexity:** 0.0&ndash;0.1 (simplest possible)
- **Small body:** 0.3&ndash;0.5 size, no appendages or tail
- **Timid grazers:** low aggression (0&ndash;0.2), small mouth (0&ndash;0.2), no hunting instinct
- **Random brain:** with biased forage neuron for innate food-seeking

### Reproduction

| Complexity | Mode | Details |
|------------|------|---------|
| &lt; 0.5 | Asexual only | Clone + mutate (25% mutation rate) |
| 0.5 &ndash; 0.7 | Sexual preferred, asexual fallback | Tries to find compatible mate first |
| 0.7 &ndash; 0.9 | Sexual preferred, rare asexual (30%) | Smooth transition zone |
| &gt; 0.9 | Sexual only | Must find a compatible mate |

**Readiness:** reproduction need &gt; 0.9 and energy &gt; 40% + 5% &times; complexity.

**Sexual crossover:** each gene randomly selected from one parent (uniform). Brain weights are a per-weight coin flip between parents. Offspring mutation rate: 15%.

**Asexual division:** clone parent genome + mutate at 25% rate. Higher drift enables faster exploration of gene space.

**Mate compatibility:** genomic distance must be &lt; 8.0. Distance calculated as sum of absolute gene differences with brain distance weighted at 0.5&times; to prevent instant speciation from brain divergence alone.

**Reproduction cost:** 30% of max energy per parent. Offspring start with 50% energy. Offspring spawn within ±3 cells of parent.

### Feeding Capability

Instead of fixed trophic roles, each creature derives a `FeedingCapability` from its genome:

- **max_prey_mass** &mdash; body_mass &times; mouth_size &times; 2.0
- **hunt_skill** &mdash; aggression &times; mouth_size &times; speed &times; hunting_instinct
- **graze_skill** &mdash; (1 &minus; aggression) &times; (1 &minus; mouth_size&times;0.5) &times; 0.5 + 0.5

This allows the full spectrum from grazers to apex predators to emerge naturally.

## Energy Economy

### Self-Sustaining Food Web

```
Sunlight → Plant Photosynthesis → Grazing by Creatures → Death → Detritus
                ↑                        ↓                         ↓
        Plant Reproduction ←── Surviving Plants Regenerate    Decomposition & Grazing
```

No artificial food is injected into the ecosystem. Energy enters only through plant photosynthesis, scales with depth zones (surface=100%, mid=70%, deep=40%), and follows logistic growth curves.

### Energy Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| Plant max energy | 15.0 | Small energy stores, rapid turnover |
| Plant metabolism | &minus;0.25 | Negative = photosynthesis (gains energy from light) |
| Plant seed threshold | &gt;70% energy | Must be well-fed to reproduce |
| Plant seed interval | 5 seconds | Rate-limited reproduction |
| Plant population cap | ~tank_area/150 (min 12) | Scales with tank size |
| Plant growth model | Logistic | Peaks at 50% population fill, min 0.1 factor |
| Detritus energy | 50% of dead creature's max | Nutrient recycling |
| Detritus decay rate | 0.5 × dt × base_metabolism | Slow decomposition |
| Graze drain | 30% of plant's current energy | Partial &mdash; plant survives |
| Graze efficiency | 50% + 50% &times; complexity | Complexity rewards better nutrient absorption |
| Creature max energy | 100 &times; body_mass | Larger creatures store more |
| Creature metabolism | mass^0.75 &times; 0.5 &times; 0.25 &times; (1 &minus; 0.25&times;complexity) | Allometric with complexity discount |
| Predation transfer | 80% of prey's max energy | High reward for successful hunts |
| Reproduction cost | 30% of max energy per parent | Investment in offspring |
| Offspring energy | 50% of max | Born with moderate reserves |

### Hunting Rules

| Prey Type | Requirements |
|-----------|-------------|
| **Plants** | graze_skill &gt; 0.2, within 5 cells |
| **Mobile prey** | hunt_skill &gt; 0.3, predator speed &ge; 0.8&times; prey speed, prey mass &lt; max_prey_mass, within 3 cells |

## Evolutionary Tradeoffs

| Trait | Benefit | Cost |
|-------|---------|------|
| Large body | More energy storage, can hunt larger prey | Higher metabolism, slower movement |
| Streamlined (elongated) | Faster, lower drag coefficient | Less energy storage |
| Bright colors | Higher visibility to allies | Attracts predators |
| Large eyes | Better sensory range (up to 18 cells) | Slight metabolism cost |
| High complexity | +50% sensory range, ~25% metabolism efficiency, richer morphology | Must find compatible mates to reproduce |
| High aggression | Better hunt_skill, can target mobile prey | Lower graze_skill |
| High timidity | Better flee response, avoids predators | Less time foraging |

## Needs System

Creatures have an internal needs system that drives behavior through the neural network:

| Need | Rate | Effect |
|------|------|--------|
| Hunger | Rises at 0.02/s | Fed by eating, drives foraging |
| Safety | Decays 50%/s when safe | Spikes near predators, drives fleeing |
| Reproduction | Rises at 0.02/s | Triggers breeding when &gt; 0.9 |
| Rest | Rises at 0.01/s | Drives resting behavior |
| Social | Drifts to baseline | Drives schooling |
| Curiosity | Rises at 0.01/s | Drives exploration |

`NeedWeights` are derived from each creature's genome, making need accumulation rates heritable and evolvable.

## Simulation Systems (per tick)

| # | System | Description |
|---|--------|-------------|
| 1 | **Environment** | Advance time, light, temperature, currents, random events |
| 2 | **Spatial grid** | Rebuild neighbor lookup structure |
| 3 | **Entity info map** | Build shared HashMap for all systems |
| 4 | **Needs** | Hunger rises, safety decays, reproduction builds |
| 5 | **Brain + Pheromone deposit** | Neural network sensory processing, decisions, chemical signaling (parallel) |
| 6 | **Pheromone grid** | Decay and diffuse pheromone concentrations |
| 7 | **Boids** | Separation, alignment, cohesion steering forces (parallel) |
| 8 | **Physics** | Integrate velocities, enforce boundaries, update facing |
| 9 | **Metabolism** | Drain energy (creatures, with night stress), gain energy (plants), increment age |
| 10 | **Hunting** | Predator-prey interactions, grazing, energy transfer (parallel) |
| 11 | **Reproduction** | Mate pairing or asexual division, fitness sharing, offspring spawning |
| 12 | **Death** | Remove starved or aged entities, spawn detritus for nutrient cycling |
| 13 | **Plant seeding** | Mature plants reproduce new plants nearby |
| 14 | **Animation** | Advance frame timers |

## Performance

Optimized for 600+ creatures at 20 ticks/sec:

| Optimization | Effect |
|-------------|--------|
| NEAT brain with topological sort | Efficient variable-topology forward pass |
| Rayon parallelism | Brain, boids, hunting run on multiple cores |
| Shared `EntityInfoMap` | One world query + one HashMap per tick (not three) |
| Spatial hash grid (cell_size=6) | Distance-filtered neighbor queries in ~O(1) |
| Sensory range cap (18.0) | Prevents tank-wide scans that defeat spatial partitioning |
| Stats caching | Species/complexity recomputed every 20 ticks, not every frame |
| Zero-alloc rendering | Left-facing flip by index reversal, no string cloning |
| Soft population cap (600) | Reproduction suppressed to maintain frame rate |

## Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| [hecs](https://crates.io/crates/hecs) | 0.11 | Lightweight ECS (entity component system) |
| [ratatui](https://crates.io/crates/ratatui) | 0.29 | TUI rendering with double-buffered diffing |
| [crossterm](https://crates.io/crates/crossterm) | 0.28 | Cross-platform terminal I/O |
| [rand](https://crates.io/crates/rand) | 0.10 | RNG for procedural generation + simulation |
| [rayon](https://crates.io/crates/rayon) | 1.10 | Data-parallel computation for brain/boids/hunting |

## Algorithms

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **NEAT neuroevolution** | `brain.rs` | Evolving topology networks with innovation numbers, add-node/add-connection mutations, topological sort forward pass |
| **Hebbian learning** | `brain.rs` | Lifetime weight plasticity via co-activation strengthening (Baldwin Effect) |
| **Boids flocking** | `boids.rs` | Craig Reynolds (1986) &mdash; separation, alignment, cohesion + wall avoidance |
| **Genetic evolution** | `genetics.rs` | Uniform crossover, Gaussian mutation, NEAT-aligned brain crossover, evolvable mutation rate |
| **Fitness sharing** | `spawner.rs` | NEAT-style speciation: greedy clustering by genomic distance, proportional reproduction penalty |
| **Allometric scaling** | `phenotype.rs` | Metabolism ~ mass^0.75 (Kleiber's law) |
| **Partial grazing** | `ecosystem.rs` | Plants lose energy when grazed but regenerate via photosynthesis |
| **Nutrient cycling** | `ecosystem.rs` | Dead creatures → detritus entities → grazable decomposition |
| **Depth zones** | `environment.rs` | 3-zone model: surface/mid/deep affecting light and metabolism |
| **Emergent predation** | `ecosystem.rs` | Morphology-derived feeding capability, body size ratio + speed checks |
| **Pheromone signaling** | `pheromone.rs` | Grid-based chemical communication with decay, diffusion, gradient sensing |
| **Spatial hashing** | `spatial.rs` | Grid-based bucketing with distance-filtered queries |

## Roadmap

- [x] Neural network brains &mdash; evolved feedforward networks drive creature behavior
- [x] NEAT topology evolution &mdash; evolve network structure (add nodes/connections), not just weights
- [x] Hebbian learning &mdash; lifetime weight plasticity with evolvable learning rate (Baldwin Effect)
- [x] Emergent evolution &mdash; continuous genome, no predefined species
- [x] Multi-threaded simulation &mdash; rayon parallelism for brain/boids/hunting
- [x] Performance optimization &mdash; shared entity maps, spatial capping
- [x] Self-sustaining ecosystem &mdash; plant photosynthesis replaces artificial food rain
- [x] Partial grazing &mdash; plants survive being eaten and regenerate
- [x] Plant reproduction &mdash; mature plants seed offspring, population-capped
- [x] Complexity-driven evolution &mdash; sensory and metabolism bonuses reward complexity
- [x] Nutrient cycling &mdash; dead creatures become grazable detritus entities
- [x] Depth zones &mdash; environmental heterogeneity with light and metabolism gradients
- [x] Fitness sharing &mdash; NEAT-style speciation protects novel innovations
- [x] Evolvable mutation rate &mdash; meta-evolution tunes mutation rates per lineage
- [x] Sexual selection &mdash; Fisherian runaway via mate color preference
- [x] Logistic plant growth &mdash; density-dependent carrying capacity
- [x] Pheromone signaling &mdash; grid-based chemical communication with decay, diffusion, and gradient sensing
- [ ] Save/load simulation state (serde serialization)
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

- Kenneth O. Stanley & Risto Miikkulainen, [Evolving Neural Networks Through Augmenting Topologies](https://www.cs.ucf.edu/~kstanley/neat.html) (Evolutionary Computation, 2002) &mdash; NEAT neuroevolution
- Christoph Adami, Charles Ofria, Travis C. Collier, [Evolution of Biological Complexity](https://www.pnas.org/doi/10.1073/pnas.97.9.4463) (PNAS, 2000) &mdash; complexity emergence via natural selection
- Larry Yaeger, [Computational Genetics, Physiology, Metabolism, Neural Systems, Learning, Vision, and Behavior](https://web.archive.org/web/20230325141123/http://pobox.com/~larryy/Yaeger.ALife3.pdf) (ALife III, 1993) &mdash; Polyworld ecosystem simulation
- Geoffrey Hinton & Steven Nowlan, [How Learning Can Guide Evolution](https://www.cs.toronto.edu/~hinton/absps/baldwin.pdf) (Complex Systems, 1987) &mdash; Baldwin Effect
- Craig Reynolds, [Flocks, Herds, and Schools: A Distributed Behavioral Model](https://www.red3d.com/cwr/papers/1987/SIGGRAPH87.pdf) (SIGGRAPH 1987)
- [Steering Behaviors for Autonomous Characters](https://slsdo.github.io/steering-behaviors/)
- [Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) for predator-prey dynamics
- [Kleiber's law](https://en.wikipedia.org/wiki/Kleiber%27s_law) for allometric metabolic scaling
- [Wa-Tor simulation](https://beltoforion.de/en/wator/) for population dynamics inspiration

## License

This project is licensed under the MIT License &mdash; see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with Rust, ratatui, and a love for emergent life.</sub>
</p>
