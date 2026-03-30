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
  <img src="https://img.shields.io/badge/Tests-239_passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Alpha-yellow" alt="Status">
</p>

<p align="center">
  No predefined species. No artificial food drops. Every creature emerges from a<br>
  continuous genome through mutation, crossover, and natural selection &mdash; rendered<br>
  as procedural ASCII art &mdash; sustained entirely by evolving genome-driven producer colonies.
</p>

---

## Features

### Evolution &amp; Genetics

- **Bottom-up emergent evolution** &mdash; no templates or predefined species; all creatures and producer colonies emerge from continuous genomes through mutation and selection
- **Creature genome** &mdash; 35+ float genes control body plan, appendages, eyes, coloring, behavior, and brain topology
- **Producer genome** &mdash; 20 float genes control colony geometry, light capture, nutrient affinity, palatability, reserve allocation, and dispersal strategy; producer strategies emerge continuously rather than from hardcoded species types
- **Complexity as a master gate** &mdash; a single 0.0&ndash;1.0 gene controls which morphological features are expressed in both creatures and producers; always mutates on reproduction (±0.15 drift) ensuring continuous exploration
- **Bioenergetic creature reproduction** &mdash; consumers mature gradually, accumulate reproductive buffer from sustained surplus, and only then reproduce asexually or sexually
- **Producer reproduction** &mdash; producer colonies allocate reserve surplus into broadcast propagules and local fragments; parent reserve is debited, establishment depends on light and crowding, and dispersal-vs-local-spread trade-offs are genome-controlled
- **Smooth complexity transitions** &mdash; gradual shift from asexual to sexual reproduction prevents evolutionary traps
- **Complexity rewards** &mdash; higher complexity grants +50% sensory range and ~25% metabolism efficiency for creatures; +15% photosynthesis efficiency for producers
- **Evolvable mutation rate** &mdash; `mutation_rate_factor` gene (0.5&ndash;2.0) scales the base mutation rate in both creatures and producers; meta-evolution tunes evolvability itself
- **Sexual selection** &mdash; `mate_preference_hue` gene drives Fisherian runaway selection by preferring mates whose color matches the preference
- **Fitness sharing** &mdash; NEAT-style speciation protection; larger species have proportionally less reproduction chance, protecting novel innovations
- **Runtime diversity coefficient** &mdash; arrow keys adjust a 0.25&ndash;2.5 slider that scales mutation rates and fitness-sharing strength in real time, letting you dial evolutionary pressure up or down without restarting
- **Trait-based producer trade-offs** &mdash; dispersal, nutrient affinity, light capture, reserve allocation, and grazing value are all continuous genes so producer strategies emerge from trait combinations rather than from species templates

### Neural Network Brains

- **NEAT-style evolving topology** &mdash; networks start minimal (16 inputs &rarr; 7 outputs, 112 connections) and grow via structural mutations that add hidden nodes (8%), connections (12%), and recurrent self-loops (6%)
- **Evolvable per-node activation functions** &mdash; each hidden/output node evolves its own activation (Tanh, ReLU, Sigmoid, Abs, Step, Identity) via swap mutations (2%), enabling diverse nonlinear representations
- **Per-node bias** &mdash; each node has an evolvable bias term that shifts the activation threshold; biases mutate alongside weights and are crossed over during reproduction
- **Innovation numbers** &mdash; historical gene markings enable meaningful crossover between creatures with different network topologies
- **Hebbian lifetime learning (Oja's rule)** &mdash; evolvable `learning_rate` gene (0.0&ndash;0.1) allows weight plasticity during a creature's life via self-normalizing Oja's rule; 10x faster than typical alife learning rates; learned changes are NOT inherited (Baldwin Effect)
- **Neuromodulation** &mdash; hidden nodes can evolve into Modulator role (1% rate), whose sigmoid-gated output scales the Oja learning rate for all connections, enabling selective plasticity
- **Attention mechanism** &mdash; hidden nodes can evolve into Attention role (0.5% rate), computing softmax-weighted blends of their inputs instead of simple weighted sums
- **Module duplication** &mdash; a rare mutation (0.5% rate) copies a connected subgraph of hidden nodes with new innovation numbers, enabling functional modularity
- **Recurrent self-connections** &mdash; hidden and output nodes can evolve self-loops that feed previous-tick activations back as input, giving creatures short-term memory without breaking feedforward topology
- **Sensory inputs** &mdash; energy, hunger, nearby food/predators/allies, walls, light, speed, pheromone concentration &amp; gradient
- **Behavioral outputs** &mdash; steering, speed, foraging, fleeing, schooling, pheromone emission
- **Evolved, not designed** &mdash; both weights and topology are part of the genome and evolve through crossover and mutation

### Pheromone Signaling

- **Chemical communication** &mdash; creatures deposit pheromones at their position; evolvable `pheromone_sensitivity` gene controls response strength
- **Grid-based diffusion** &mdash; pheromone concentrations spread to neighboring cells (5% per tick) and decay (×0.95 per tick, ~3.5s half-life)
- **Gradient sensing** &mdash; neural network receives local concentration and directional gradient as inputs, enabling trail following and collective behavior
- **Evolved, not designed** &mdash; pheromone emission is a neural network output; whether and how creatures use chemical signals emerges through evolution

### Self-Sustaining Ecosystem

- **Genome-driven producer colonies** &mdash; producers evolve via a 20-float `ProducerGenome`; colony geometry, physiology, nutrient strategy, and dispersal mode all emerge from the genome
- **LAI-based photosynthesis with nutrient co-limitation** &mdash; canopy capture follows Beer&ndash;Lambert-style light interception, then realized growth is limited by light, dissolved nitrogen, and dissolved phosphorus
- **Allometric producer metabolism** &mdash; maintenance and recovery scale with biomass rather than with fixed stage tables, following metabolic-scaling ideas
- **Producer storage and regeneration** &mdash; dormant biomass and regenerative banks buffer dark periods, post-grazing recovery, and local fragmentation
- **Rasterized canopy shading** &mdash; a tank-wide `LightField` applies continuous depth attenuation plus canopy, phytoplankton, and epiphyte shading
- **Demand-limited grazing** &mdash; grazers remove active producer biomass first, can scrub fouling load, and intake is limited by consumer demand rather than by a fixed prey damage fraction
- **Producer lifecycle stages** &mdash; colonies progress through Cell &rarr; Patch &rarr; Mature &rarr; Broadcasting &rarr; Collapsing from reserve status, biomass fill, stress load, and age
- **Procedural producer ASCII art** &mdash; 4 complexity tiers (speck, tuft, mat, plume) selected by raw genome complexity; producer size within each tier scales with lifecycle stage
- **Reserve-cost producer reproduction** &mdash; mature producer colonies invest reserve surplus into broadcast propagules and fragments; establishment depends on local light, depth, and crowding
- **No artificial food rain** &mdash; the ecosystem sustains itself entirely through producer photosynthesis; manual food drops still available via `f` key
- **Nutrient cycling** &mdash; dead creatures become detritus entities, producer turnover returns N and P to dissolved/sediment pools, and nutrient load can drive phytoplankton shading; nitrogen fixation prevents irreversible N-depletion and a nutrient floor (5% of initial) ensures the ecosystem can always recover
- **Substrate zones** &mdash; procedurally generated Sandy/Rocky/Planted substrate affects producer establishment; rocky zones favor high-hardiness producers, planted zones boost clonal spread
- **Continuous depth/light attenuation** &mdash; underwater light declines smoothly with depth and water clarity rather than via three hard depth bands
- **Night metabolism stress** &mdash; creatures burn 30% more energy at night, favoring complex creatures with metabolism efficiency bonuses
- **Temperature effects** &mdash; Q10-based metabolic scaling affects both creatures and producers; cold snaps (&minus;10°C) create strong selection pressure
- **Morphology-driven feeding** &mdash; `FeedingCapability` derived from mouth size, aggression, and hunting instinct; full spectrum from grazers to apex predators emerges naturally
- **Predator-prey dynamics** &mdash; body-size ratio hunting, speed advantage checks, energy transfer on kill

### Simulation

- **Procedural ASCII art** &mdash; 4 complexity tiers for both creatures (cells, simple, medium, complex) and producers (speck, tuft, mat, plume) generated from their respective genomes; no hardcoded art
- **Boids flocking** &mdash; separation, alignment, cohesion with size-aware spacing and wall avoidance
- **Allometric metabolism** &mdash; energy cost scales with mass^0.75 for creatures and producers (Kleiber's law)
- **Day/night cycle** &mdash; sine-based lighting, palette shifts from bright day through dusk to dark night
- **Random events** &mdash; algae blooms, feeding frenzies, cold snaps (&minus;10°C), earthquakes (every ~60s)
- **Multi-threaded** &mdash; brain, boids, and hunting systems parallelized with rayon
- **Soft population cap** &mdash; reproduction suppressed above 600 creatures to maintain responsiveness
- **HUD overlay** &mdash; population, generation, complexity, species count, diversity coefficient, split creature/producer birth-death counters, day, time, temperature, light, speed, plus a toggleable ecology diagnostics panel and a help popup (`?`) explaining all abbreviations
- **Single founder-web startup** &mdash; the visible run starts from low-biomass producer colonies and simple consumer founders, with no hidden warmup

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

The default visible run starts directly from a simple aquatic founder web:

- low-biomass producer colonies
- simple motile consumer founders
- no hidden warmup

### Run Tests

```bash
cargo test --workspace    # 239 passing tests (224 core + 13 render + 2 app)
```

## Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `Space` | Pause / Resume |
| `→` | Speed up (0.5x increments, max 20x) |
| `←` | Slow down (0.5x decrements, min 0.5x) |
| `↑` | Increase diversity coefficient (+0.1, max 2.5) |
| `↓` | Decrease diversity coefficient (&minus;0.1, min 0.25) |
| `r` | Reset speed and diversity to defaults (1.0) |
| `f` | Drop food pellet (manual only; no auto-spawning) |
| `d` | Toggle ecology diagnostics overlay |
| `?` / `h` | Toggle help popup (explains all HUD abbreviations) |

## Architecture

tuiquarium strictly separates simulation from rendering through traits and dependency injection:

```
tuiquarium/
├── src/main.rs               # Entry point, TUI event loop
├── crates/
│   ├── tuiq-core/            # Pure simulation logic (zero rendering deps)
│   │   ├── animation.rs      # Frame sequencing & timing
│   │   ├── behavior.rs       # Behavioral action types & speed multipliers
│   │   ├── boids.rs          # Boids flocking (rayon-parallelized)
│   │   ├── bootstrap.rs      # Founder genomes & ecosystem initialization
│   │   ├── brain.rs          # NEAT neural network brains (rayon-parallelized)
│   │   ├── calibration.rs    # Runtime calibration parameters
│   │   ├── components.rs     # ECS components (Position, Velocity, Appearance, ...)
│   │   ├── ecosystem.rs      # Energy, metabolism, grazing/hunting, death
│   │   ├── environment.rs    # Day/night cycle, temperature, currents, events, substrate zones
│   │   ├── genetics.rs       # Crossover, mutation, genomic distance
│   │   ├── genome.rs         # CreatureGenome + ProducerGenome
│   │   ├── lib.rs            # AquariumSim orchestrator, Simulation trait, tick()
│   │   ├── needs.rs          # Hunger/safety/social drift
│   │   ├── phenotype.rs      # Genome → physical stats (creatures + producers)
│   │   ├── pheromone.rs      # Chemical signaling grid (deposit, diffusion, decay)
│   │   ├── physics.rs        # Position integration, boundary handling
│   │   ├── producer_lifecycle.rs  # Producer genome → procedural ASCII art, lifecycle stages
│   │   ├── producer_reproduction.rs # Producer broadcast & clonal propagation
│   │   ├── spatial.rs        # Spatial hash grid with distance-filtered queries
│   │   ├── spawner.rs        # Asexual/sexual reproduction system
│   │   ├── stats.rs          # SimStats, EcologyInstant, EcologyDiagnostics
│   │   └── systems.rs        # Trait abstractions: BrainSystem, EcosystemSystem, HuntingSystem, ReproductionSystem, ProducerLifecycleSystem
│   │
│   └── tuiq-render/          # Ratatui rendering (depends on tuiq-core)
│       ├── ascii.rs          # Procedural ASCII art generation from genome
│       ├── effects.rs        # Bubble particle system
│       ├── hud.rs            # Stats overlay (pop, gen, complexity, species, diversity, ...)
│       ├── palette.rs        # Day/night color palette shifts
│       └── tank.rs           # Tank border, water, substrate, creature rendering
```

### Design Principles

- **Simulation knows nothing about rendering.** The `Simulation` trait exposes read-only access to the ECS world. Rendering code never mutates simulation state.
- **Trait-based system architecture.** Each major ECS system (brain, ecosystem, hunting, reproduction, producer lifecycle) is defined as a trait with a concrete zero-cost implementation. The `tick()` orchestrator delegates through trait methods, enabling testability and future extensibility.
- **ECS architecture** with [hecs](https://github.com/Ralith/hecs) &mdash; lightweight, no global state, manual system orchestration.
- **Fixed timestep game loop** &mdash; 50ms ticks (20 ticks/sec simulation), ~60fps rendering, accumulator pattern.
- **Spatial hash grid** reduces neighbor queries from O(n&sup2;) to ~O(n).
- **Shared entity info map** built once per tick, passed to brain/boids/hunting to eliminate redundant world queries and HashMap constructions.
- **Runtime diversity coefficient** &mdash; a single `[0.25, 2.5]` slider scales mutation rates and fitness-sharing strength, letting the user tune evolutionary pressure interactively without restarting.

## Neural Network Brains

Every creature has an evolving neural network (NEAT-style) evaluated each tick. Networks start minimal and grow via structural mutations. Each node has its own activation function, bias, and optional specialized role.

### Network Architecture

```
Sensory Inputs (16, Identity)  →  [Evolving Hidden Topology]  →  Outputs (7, per-node activation)
                                   │ Per-node: ActivationFn + bias + role
                                   │ Roles: Standard, Modulator, Attention
```

Networks begin as direct input→output connections (112 weights) and grow via structural mutations:

- **Add node** (8% per generation): splits an existing connection, inserting a hidden neuron with a random activation (Tanh/ReLU/Sigmoid)
- **Add connection** (12% per generation): adds a new feedforward connection between non-connected nodes
- **Add self-connection** (6% per generation): adds a recurrent self-loop on a hidden or output node, enabling short-term memory
- **Activation swap** (2% per generation): changes a random non-input node's activation function to a randomly chosen one from {Tanh, ReLU, Sigmoid, Abs, Step, Identity}
- **Modulator flip** (1% per generation): converts a hidden node to Modulator role (forced Sigmoid activation), whose output gates the Oja learning rate globally
- **Attention flip** (0.5% per generation): converts a hidden node to Attention role (forced Identity activation), computing softmax-weighted input blends
- **Module duplication** (0.5% per generation): copies a 1-hop subgraph around a random hidden node, creating duplicate modules with new node IDs and innovation numbers

Maximum topology: 60 nodes total, 300 connections. **Innovation numbers** track each structural mutation's history, enabling meaningful crossover between creatures with different topologies.

**Per-node genes:** Each node has an `ActivationFn` (6 variants), a `bias` (−2.0 to +2.0), and a `NodeRole` (Standard/Modulator/Attention). These are stored in a `NodeGene` struct per node and evolve through mutation and crossover.

**Output scaling:** all outputs are scaled by `complexity.max(0.3)`, ensuring even simple creatures can take meaningful actions while complex creatures have full control authority.

**Hebbian learning (Oja's rule):** an evolvable `learning_rate` gene (0.0–0.1) controls lifetime weight plasticity. Weights update via Oja's rule: Δw = η · post · (pre − post · w), which is self-normalizing and prevents weight saturation. The learning rate is scaled by 0.01 (10x faster than typical alife rates) so creatures can adapt within their lifespan. A small weight decay (0.02% per tick) prevents drift. Biases also update via an analogous Oja-like rule. If Modulator nodes are present, their sigmoid-squashed output gates the learning rate for all connections — modulators near 0.0 suppress learning, near 1.0 allow it. Learned weights are NOT inherited — only innate genome weights evolve (Baldwin Effect).

**Attention nodes:** Instead of computing a weighted sum, Attention nodes compute a softmax over connection weights to produce attention scores, then blend input activations proportionally. This enables selective focus on the most relevant inputs.

**Module duplication:** When triggered, this mutation selects a random hidden node, finds all nodes within 1 connection hop, duplicates the subgraph with fresh node IDs and innovation numbers, and wires the copy's external connections analogously to the original. This enables functional modularity — successful subnetworks can be reused and diverge independently.

**Recurrent self-connections:** hidden and output nodes can evolve self-loop connections where the previous tick's activation feeds back as additional input. This gives creatures short-term memory — a node's output depends not just on current sensory input but on its recent history. Self-loops are separated from the feedforward topological sort and use a dedicated activation buffer, so they don't create cycles in the forward pass.

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
| **Brain** | NEAT genome: variable-length connection genes with innovation numbers, per-node genes (activation function, bias, role), evolving topology | weights &minus;3 to +3, bias &minus;2 to +2 |
| **Complexity** | master gate controlling feature expression | 0.0&ndash;1.0 |
| **Generation** | inherited from parents + 1 | 0&ndash;&infin; |

### Producer Genome

Producer colonies have their own 20-float genome plus generation tracking. It drives colony geometry, physiology, stress tolerance, nutrient use, herbivory resistance, and dispersal strategy. Producer genomes evolve through mutation during reproduction (no crossover &mdash; producer colonies stay asexual in this model).

| Gene Group | Genes | Range | Ecological Basis |
|-----------|-------|-------|-----------------|
| **Morphology** | stem_thickness, height_factor, leaf_area (capture-area proxy), branching, curvature, primary_hue | 0&ndash;1 | aquatic producer geometry; light access and colony-shape trade-offs |
| **Physiology** | photosynthesis_rate, max_energy_factor, hardiness, nutritional_value, nutrient_affinity, epiphyte_resistance, reserve_allocation | 0&ndash;1.5 (varies) | metabolic scaling, nutrient co-limitation, herbivory and attached-growth trade-offs |
| **Propagation** | seed_range, seed_count, seed_size, lifespan_factor, clonal_spread | 0&ndash;2 (varies) | broadcast-vs-local spread trade-offs, propagule-pressure ecology |
| **Evolution** | complexity, mutation_rate_factor | 0&ndash;2 | Meta-evolution |
| **Lineage** | generation | integer | Heritable lineage tracking |

**Key models:**

- **Beer&ndash;Lambert canopy capture** &mdash; effective LAI = leaf_area &times; (1 + branching&times;0.5) &times; (0.5 + height&times;0.5); light capture has diminishing returns with canopy density
- **Monod-style resource limitation** &mdash; realized growth is limited by saturating responses to light, dissolved nitrogen, and dissolved phosphorus rather than by hard thresholds
- **Allometric maintenance** &mdash; biomass maintenance scales sublinearly with size, while tissue turnover and senescence increase under chronic stress
- **Reserve-cost propagation** &mdash; reserve surplus is partitioned into broadcast propagules and local fragments; `seed_count`, `seed_size`, and `clonal_spread` define strategy trade-offs
- **Visual complexity tiers** &mdash; producer appearance is complexity-gated by raw genome complexity: &lt;0.12 speck, 0.12&ndash;0.35 tuft, 0.35&ndash;0.65 mat, &ge;0.65 plume; within each tier, size scales with lifecycle stage

### Primordial Producer Colonies

Initial producers are spawned with `minimal_producer()`:

- **Complexity:** 0.05&ndash;0.25 (starts as speck or tuft tier)
- **Low-profile geometry:** all morphology genes randomized to colony-like low-to-moderate values
- **Moderate physiology:** photosynthesis_rate ~1.0, balanced broadcast vs fragmentation strategy
- **Staggered start:** initial energy randomized 50&ndash;100%, initial age randomized 0&ndash;20s to desynchronize lifecycle stages
- **Generation 0:** first generation, no parent lineage

### Primordial Cells

The simulation begins with `minimal_cell()` organisms:

- **Complexity:** 0.0&ndash;0.1 (simplest possible)
- **Small body:** 0.3&ndash;0.5 size, no appendages or tail
- **Timid grazers:** low aggression (0&ndash;0.2), small mouth (0&ndash;0.2), no hunting instinct
- **Random brain:** with biased forage neuron for innate food-seeking

### Reproduction

| Complexity | Mode | Details |
|------------|------|---------|
| &lt; 0.32 | Asexual only | Clone + mutate (25% mutation rate) |
| 0.32 &ndash; 0.62 | Sexual preferred, asexual fallback | Tries to find compatible mate first |
| 0.62 &ndash; 0.82 | Sexual preferred, rare asexual (45%) | Smooth transition zone |
| &ge; 0.82 | Sexual only | Must find a compatible mate |

**Readiness:** consumers must be mature, hold reproductive buffer above an offspring threshold, clear a brood cooldown, and maintain strong body condition. `Needs.reproduction` is now a derived behavioral signal, not a timer.

**Sexual crossover:** each gene randomly selected from one parent (uniform). Brain weights are a per-weight coin flip between parents. Offspring mutation rate: 15%.

**Asexual division:** clone parent genome + mutate at 25% rate. Higher drift enables faster exploration of gene space.

**Mate compatibility:** genomic distance must be &lt; 8.0. Distance calculated as sum of absolute gene differences with brain distance weighted at 0.5&times; to prevent instant speciation from brain divergence alone.

**Reproduction cost:** explicit parental energy plus reproductive-buffer investment. Offspring start energy is reduced, and parents enter a cooldown instead of immediately breeding again.

### Feeding Capability

Instead of fixed trophic roles, each creature derives a `FeedingCapability` from its genome:

- **max_prey_mass** &mdash; body_mass &times; mouth_size &times; 2.0
- **hunt_skill** &mdash; aggression &times; mouth_size &times; speed &times; hunting_instinct
- **graze_skill** &mdash; (1 &minus; aggression) &times; (1 &minus; mouth_size&times;0.5) &times; 0.5 + 0.5

This allows the full spectrum from grazers to apex predators to emerge naturally.

## Energy Economy

### Self-Sustaining Food Web

```
Sunlight → Producer Photosynthesis → Grazing by Creatures → Death → Detritus
                 ↑                           ↓                         ↓
        Producer Reproduction ←── Surviving Producers Recover   Decomposition & Grazing
```

No artificial food is injected into the ecosystem. Energy enters through producer photosynthesis, but realized producer growth is filtered by canopy shading, water-column attenuation, dissolved nutrients, fouling load, and herbivory.

### Energy Parameters

| Parameter | Model | Effect |
|-----------|-------|--------|
| Producer max energy | 15 &times; max_energy_factor &times; (1 + mass) | Genome-controlled reserve storage |
| Producer photosynthesis | Beer&ndash;Lambert capture &times; Monod light/N/P limitation &times; canopy/fouling shading | Carbon gain limited by the scarcest resource |
| Producer maintenance | Allometric biomass maintenance + tissue turnover + senescence turnover | Older or chronically stressed producers lose reserve and tissue gradually |
| Producer reserve buffering | Dormant biomass + regenerative bank translocation | Attached producer stands can survive dark/stress periods and regrow after defoliation |
| Producer reproduction | Reserve surplus &times; reserve_allocation &times; maturity | Parent reserve is debited; broadcast and fragment propagules differ in cost and dispersal |
| Producer establishment | Local light &times; crowding filter &times; propagule type | Fragments establish reliably nearby; broadcast propagules disperse farther but fail more often |
| Producer mortality | Reserve exhaustion or severe tissue loss | No deterministic age kill-switch for healthy producer colonies |
| Nutrient pool | Dissolved/sediment N and P + phytoplankton load | Couples producer growth, detritus recycling, and turbidity |
| Grazer intake | Consumer energy deficit, body mass, graze skill, and handling-limited intake over time | Satiated grazers do less damage; hungry grazers strip leaf tissue first |
| Detritus recycling | Dead creatures and plant turnover feed dissolved/sediment nutrients | Closes the nutrient loop |
| Creature max energy | 100 &times; body_mass | Larger creatures store more |
| Creature metabolism | mass^0.75 with complexity discount | Allometric cost for mobile consumers |
| Predation transfer | Fraction of prey reserve on successful capture | Hunting remains high-reward relative to grazing |
| Reproduction cost | Fraction of parental reserve | Investment in offspring remains explicit |

### Hunting Rules

| Prey Type | Requirements |
|-----------|-------------|
| **Plants** | graze_skill &gt; 0.2, non-trivial hunger/energy deficit, within 4 cells |
| **Mobile prey** | hunt_skill &gt; 0.3, hunger/energy deficit, predator speed &ge; 0.8&times; prey speed, prey mass &lt; max_prey_mass, within 3 cells |

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
| Reproduction | Derived from maturity + reproductive buffer | Triggers mate-seeking only when life-history gates are satisfied |
| Rest | Rises at 0.01/s | Drives resting behavior |
| Social | Drifts to baseline | Drives schooling |
| Curiosity | Rises at 0.01/s | Drives exploration |

`NeedWeights` are derived from each creature's genome, making baseline hunger/rest/social drift evolvable. Reproductive timing is handled separately by `ConsumerState`.

## Simulation Systems (per tick)

| # | System | Description |
|---|--------|-------------|
| 1 | **Environment** | Advance time, light, temperature, currents, random events |
| 2 | **Spatial grid** | Rebuild neighbor lookup structure |
| 3 | **Entity info map** | Build shared HashMap for all systems |
| 4 | **Needs** | Hunger rises, safety decays, and behavioral needs drift |
| 5 | **Brain + Pheromone deposit** | Neural network sensory processing, decisions, chemical signaling (parallel) |
| 6 | **Pheromone grid** | Decay and diffuse pheromone concentrations |
| 7 | **Boids** | Separation, alignment, cohesion steering forces (parallel) |
| 8 | **Physics** | Integrate velocities, enforce boundaries, update facing |
| 9 | **Nutrients + LightField** | Update dissolved/sediment nutrients, phytoplankton load, and rasterized canopy shading |
| 10 | **Metabolism + Producer ecology** | Drain creature energy, recycle detritus, run producer photosynthesis/maintenance/turnover, and buffer regenerative reserves |
| 10b | **Consumer life history** | Update maturation, reproductive buffer, and brood cooldown from energetic state |
| 11 | **Hunting** | Predator-prey interactions, demand-limited grazing, energy transfer (parallel) |
| 12 | **Reproduction** | Mate pairing or asexual division for creatures, then reserve-cost broadcast/fragment propagation for producers |
| 13 | **Death** | Remove starved entities, recycle producer biomass, spawn detritus for nutrient cycling |
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
| **NEAT neuroevolution** | `brain.rs` | Evolving topology networks with innovation numbers, per-node activation functions (6 variants), bias, structural/role mutations, topological sort forward pass |
| **Oja's rule** | `brain.rs` | Self-normalizing lifetime weight plasticity: Δw = η·post·(pre − post·w) + weight decay, with neuromodulation gating (Baldwin Effect) |
| **Boids flocking** | `boids.rs` | Craig Reynolds (1986) &mdash; separation, alignment, cohesion + wall avoidance |
| **Genetic evolution** | `genetics.rs` | Uniform crossover, Gaussian mutation, NEAT-aligned brain crossover, evolvable mutation rate; producer mutation with independent perturbation per gene |
| **Fitness sharing** | `spawner.rs` | NEAT-style speciation: greedy clustering by genomic distance, proportional reproduction penalty |
| **Allometric scaling** | `phenotype.rs`, `ecosystem.rs` | Metabolism ~ mass^0.75 for creatures and producers (Kleiber, 1947) |
| **Beer&ndash;Lambert canopy capture** | `phenotype.rs`, `ecosystem.rs` | LAI-based light interception with diminishing returns |
| **Monod resource limitation** | `ecosystem.rs` | Saturating light, nitrogen, and phosphorus limitation for producer growth |
| **Procedural producer art** | `producer_lifecycle.rs` | Genome-driven colony ASCII art with stage-scaled speck/tuft/mat/plume forms |
| **Dispersal trade-offs** | `genome.rs`, `lib.rs` | Broadcast propagule count vs size and local fragmentation trade-offs |
| **Producer lifecycle** | `producer_lifecycle.rs` | Biomass/reserve/stress-based stage transitions with cell/patch/mature/broadcasting/collapsing states |
| **Demand-limited grazing** | `ecosystem.rs` | Grazers remove active producer biomass and reserve according to consumer demand rather than fixed prey damage |
| **Nutrient cycling** | `ecosystem.rs` | Dead creatures &rarr; detritus entities &rarr; grazable decomposition |
| **Rasterized light field** | `ecosystem.rs` | Continuous depth attenuation plus canopy, phytoplankton, and epiphyte shading |
| **Emergent predation** | `ecosystem.rs` | Morphology-derived feeding capability, body size ratio + speed checks |
| **Pheromone signaling** | `pheromone.rs` | Grid-based chemical communication with decay, diffusion, gradient sensing |
| **Spatial hashing** | `spatial.rs` | Grid-based bucketing with distance-filtered queries |

## Roadmap

- [x] Neural network brains &mdash; evolved feedforward networks drive creature behavior
- [x] NEAT topology evolution &mdash; evolve network structure (add nodes/connections), not just weights
- [x] Hebbian learning &mdash; Oja's rule lifetime weight plasticity with evolvable learning rate (Baldwin Effect)
- [x] Emergent evolution &mdash; continuous genome, no predefined species
- [x] Multi-threaded simulation &mdash; rayon parallelism for brain/boids/hunting
- [x] Performance optimization &mdash; shared entity maps, spatial capping
- [x] Self-sustaining ecosystem &mdash; producer photosynthesis replaces artificial food rain
- [x] Partial grazing &mdash; producer colonies survive being eaten and regenerate
- [x] Producer reproduction &mdash; reserve-cost broadcast and fragment propagation with establishment filters
- [x] Genome-driven producers &mdash; 20-float `ProducerGenome` with allometric scaling, LAI-style light capture, nutrient limitation, and procedural ASCII art
- [x] Complexity-driven evolution &mdash; sensory and metabolism bonuses reward complexity
- [x] Nutrient cycling &mdash; dead creatures become grazable detritus entities
- [x] Rasterized light field &mdash; continuous depth and canopy shading instead of hard depth zones
- [x] Fitness sharing &mdash; NEAT-style speciation protects novel innovations
- [x] Evolvable mutation rate &mdash; meta-evolution tunes mutation rates per lineage
- [x] Sexual selection &mdash; Fisherian runaway via mate color preference
- [x] Nutrient-limited producer ecology &mdash; Monod-style resource limitation, gradual senescence, and demand-limited grazing
- [x] Pheromone signaling &mdash; grid-based chemical communication with decay, diffusion, and gradient sensing
- [x] Runtime diversity coefficient &mdash; ↑/↓ keys scale mutation rates and fitness sharing for interactive evolutionary tuning
- [x] Substrate zones &mdash; procedural Sandy/Rocky/Planted substrate affecting producer establishment and clonal spread
- [x] Nutrient resilience &mdash; nitrogen fixation, increased benthic P release, and nutrient floor prevent irreversible crashes
- [x] Trait-based system architecture &mdash; modular trait abstractions for brain, ecosystem, hunting, reproduction, and producer lifecycle
- [x] Evolvable activation functions &mdash; per-node activation (Tanh/ReLU/Sigmoid/Abs/Step/Identity) evolved through swap mutations
- [x] Per-node bias &mdash; evolvable bias terms with Oja-like lifetime updates
- [x] Neuromodulation &mdash; Modulator nodes gate Oja learning rate for selective plasticity
- [x] Attention mechanism &mdash; Attention nodes compute softmax-weighted input blends
- [x] Module duplication &mdash; mutation copies connected subgraphs for functional modularity
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

### Neuroevolution &amp; Artificial Life

- Kenneth O. Stanley & Risto Miikkulainen, [Evolving Neural Networks Through Augmenting Topologies](https://www.cs.ucf.edu/~kstanley/neat.html) (Evolutionary Computation, 2002) &mdash; NEAT neuroevolution
- Christoph Adami, Charles Ofria, Travis C. Collier, [Evolution of Biological Complexity](https://www.pnas.org/doi/10.1073/pnas.97.9.4463) (PNAS, 2000) &mdash; complexity emergence via natural selection
- Larry Yaeger, [Computational Genetics, Physiology, Metabolism, Neural Systems, Learning, Vision, and Behavior](https://web.archive.org/web/20230325141123/http://pobox.com/~larryy/Yaeger.ALife3.pdf) (ALife III, 1993) &mdash; Polyworld ecosystem simulation
- Geoffrey Hinton & Steven Nowlan, [How Learning Can Guide Evolution](https://www.cs.toronto.edu/~hinton/absps/baldwin.pdf) (Complex Systems, 1987) &mdash; Baldwin Effect

### Flocking &amp; Behavioral Models

- Craig Reynolds, [Flocks, Herds, and Schools: A Distributed Behavioral Model](https://www.red3d.com/cwr/papers/1987/SIGGRAPH87.pdf) (SIGGRAPH 1987)
- [Steering Behaviors for Autonomous Characters](https://slsdo.github.io/steering-behaviors/)

### Aquatic Producer Ecology

- Litchman & Klausmeier, [Trait-Based Community Ecology of Phytoplankton](https://doi.org/10.1146/annurev.ecolsys.39.110707.173549) (Annual Review of Ecology, Evolution, and Systematics, 2008) &mdash; supports modeling aquatic producers as continuous trait combinations along light capture, nutrient use, and dispersal trade-offs
- Azam et al., [The Ecological Role of Water-Column Microbes in the Sea](https://doi.org/10.3354/meps010257) (Marine Ecology Progress Series, 1983) &mdash; supports the founder-web framing of phototrophic producers, heterotrophs, detritus, and recycling in a microbial aquatic ecosystem
- Liu et al., [Coupling Between Carbon and Nitrogen Metabolic Processes Mediated by Coastal Microbes in Synechococcus-Derived Organic Matter Addition Incubations](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2020.593561/full) (Frontiers in Microbiology, 2020) &mdash; supports routing part of active producer carbon through a labile microbial-loop pathway instead of forcing all heterotroph intake through direct tissue grazing
- Weedall & Hall, [Sexual reproduction and genetic exchange in parasitic protists](https://pubmed.ncbi.nlm.nih.gov/25529755/) (Parasitology, 2015) &mdash; supports allowing genetic exchange in morphologically simple aquatic consumers instead of restricting crossover to animal-like complexity
- Robert H. MacArthur & Edward O. Wilson, *The Theory of Island Biogeography* (Princeton University Press, 1967) &mdash; r/K selection theory: trade-off between many cheap offspring (r) and few costly offspring (K); maps to propagule count vs propagule size
- He et al., [Trait-mediated light and depth responses in submerged macrophytes](https://pubmed.ncbi.nlm.nih.gov/30842784/) (2019) &mdash; supports morphology-driven biomass targets and canopy access to brighter water
- Yu et al., [Nitrogen enrichment and indirect shading effects on submerged plants](https://pmc.ncbi.nlm.nih.gov/articles/PMC6300520/) (2018) &mdash; supports phytoplankton/periphyton shading and canopy-form vs low-form trade-offs
- Mebane et al., [Nutrient co-limitation in aquatic primary producers](https://pubmed.ncbi.nlm.nih.gov/34143815/) (2021) &mdash; supports explicit dissolved N and P limitation instead of single-resource producer growth
- Li et al., [Vegetative Propagule Pressure and Water Depth Affect Biomass and Evenness of Submerged Macrophyte Communities](https://pmc.ncbi.nlm.nih.gov/articles/PMC4641593/) (2015) &mdash; supports local establishment limits and patch-occupancy effects on propagules
- Thompson & Eckert, [Trade-offs between sexual and clonal reproduction in aquatic plants](https://pubmed.ncbi.nlm.nih.gov/15149401/) (2004) &mdash; supports reserve-cost dispersal vs local spread allocation
- Ren et al., [Water depth affects submersed macrophyte more than herbivorous snail in mesotrophic lakes](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140150/) (2024) &mdash; supports grazing that can reduce attached growth while depth/light remains the dominant producer driver

### Allometric Scaling &amp; Metabolism

- Max Kleiber, [Body Size and Metabolic Rate](https://doi.org/10.1152/physrev.1947.27.4.511) (Physiological Reviews, 1947) &mdash; Kleiber's law: metabolic rate scales with mass^0.75; used for both creature and producer maintenance costs
- Gillooly et al., [Effects of Size and Temperature on Developmental Time](https://doi.org/10.1038/417070a) (Nature, 2002) &mdash; supports scaling consumer maturation with body size on a weaker timescale than total lifespan in the aquatic founder web
- Brown et al., [Toward a Metabolic Theory of Ecology](https://doi.org/10.1890/03-9000) (Ecology, 2004) &mdash; supports allometric maintenance and reserve-allocation framing across organisms

### Photosynthesis &amp; Light Capture

- Beer&ndash;Lambert law &mdash; exponential light attenuation through a medium; used for LAI-based canopy capture and water-column attenuation
- Jacques Monod, [The Growth of Bacterial Cultures](https://doi.org/10.1146/annurev.mi.03.100149.002103) (Annual Review of Microbiology, 1949) &mdash; provides the saturating half-saturation model used for light/N/P limitation
- C. S. Holling, [The Components of Predation as Revealed by a Study of Small-Mammal Predation of the European Pine Sawfly](https://doi.org/10.4039/Ent91293-5) (The Canadian Entomologist, 1959) &mdash; informs the demand-limited, saturating consumer-resource framing used for grazing

### Population Dynamics

- [Lotka&ndash;Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) &mdash; predator-prey dynamics
- [Wa-Tor simulation](https://beltoforion.de/en/wator/) &mdash; population dynamics inspiration
- [Fisherian runaway selection](https://en.wikipedia.org/wiki/Fisherian_runaway) &mdash; sexual selection via mate preference feedback loops

## License

This project is licensed under the MIT License &mdash; see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with Rust, ratatui, and a love for emergent life.</sub>
</p>
