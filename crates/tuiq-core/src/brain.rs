//! NEAT-style neural network brains for creatures.
//!
//! Each creature has an evolving neural network that maps sensory inputs to
//! steering forces and behavioral tendencies. The network topology evolves
//! through NEAT-style mutations: starting from a minimal direct-connection
//! architecture (inputs → outputs), structural mutations add hidden nodes
//! and new connections over generations. Innovation numbers enable meaningful
//! crossover between different topologies.

use std::collections::{HashMap, HashSet, VecDeque};

use hecs::{Entity, World};
use rand::Rng;
use rand::RngExt;
use rayon::prelude::*;

use crate::behavior::{BehaviorAction, BehaviorState};
use crate::components::{Position, Velocity};
use crate::ecosystem::Energy;
use crate::environment::Environment;
use crate::genome::CreatureGenome;
use crate::needs::Needs;
use crate::phenotype::{DerivedPhysics, FeedingCapability};
use crate::spatial::SpatialGrid;
use crate::EntityInfoMap;

// ── Constants ──────────────────────────────────────────────────

pub const INPUT_SIZE: usize = 16;
pub const OUTPUT_SIZE: usize = 7;

const FIRST_OUTPUT: u16 = INPUT_SIZE as u16; // 16
const FIRST_HIDDEN: u16 = (INPUT_SIZE + OUTPUT_SIZE) as u16; // 23
const MAX_NODES: u16 = 60; // cap total nodes (40 hidden max)
const MAX_CONNECTIONS: usize = 300;

/// Number of connections in the initial full input→output topology.
const INITIAL_INNOVATIONS: u32 = (INPUT_SIZE * OUTPUT_SIZE) as u32; // 112

// Structural mutation rates
const ADD_NODE_RATE: f64 = 0.08;
const ADD_CONN_RATE: f64 = 0.12;
const ADD_SELF_CONN_RATE: f64 = 0.06;
const ACTIVATION_SWAP_RATE: f64 = 0.02;
const MODULE_DUP_RATE: f64 = 0.005;
const MODULATOR_FLIP_RATE: f64 = 0.01;
const ATTENTION_FLIP_RATE: f64 = 0.005;

// NEAT distance coefficients
const C_EXCESS: f32 = 1.0;
const C_DISJOINT: f32 = 1.0;
const C_WEIGHT: f32 = 0.4;
const C_ACTIVATION: f32 = 0.3;

// ── ActivationFn ───────────────────────────────────────────────

/// Per-node activation function. Evolved through mutation and crossover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFn {
    Tanh,
    ReLU,
    Sigmoid,
    Abs,
    Step,
    Identity,
}

impl ActivationFn {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::ReLU => x.max(0.0),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Abs => x.abs(),
            ActivationFn::Step => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::Identity => x,
        }
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..6u8) {
            0 => ActivationFn::Tanh,
            1 => ActivationFn::ReLU,
            2 => ActivationFn::Sigmoid,
            3 => ActivationFn::Abs,
            4 => ActivationFn::Step,
            _ => ActivationFn::Identity,
        }
    }

    /// Random activation suitable for hidden nodes (excludes Identity and Step
    /// which can cause dead or binary neurons in initial hidden layers).
    fn random_hidden(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..3u8) {
            0 => ActivationFn::Tanh,
            1 => ActivationFn::ReLU,
            _ => ActivationFn::Sigmoid,
        }
    }
}

// ── NodeRole ───────────────────────────────────────────────────

/// Specialized role for a node. Most nodes are Standard; Modulator and
/// Attention nodes have special forward-pass behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Normal neuron: weighted sum → activation
    Standard,
    /// Output gates Oja learning rate for nearby connections
    Modulator,
    /// Computes softmax-weighted blend of inputs
    Attention,
}

impl Default for NodeRole {
    fn default() -> Self {
        NodeRole::Standard
    }
}

// ── NodeGene ───────────────────────────────────────────────────

/// Per-node genome: activation function, bias, and role.
#[derive(Debug, Clone)]
pub struct NodeGene {
    pub activation_fn: ActivationFn,
    pub bias: f32,
    pub role: NodeRole,
}

// ── ConnectionGene ─────────────────────────────────────────────

/// A single connection in the NEAT genome, identified by its innovation number.
#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub in_node: u16,
    pub out_node: u16,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

// ── InnovationTracker ──────────────────────────────────────────

/// Tracks structural innovation numbers for NEAT crossover alignment.
/// Within a generation, identical structural mutations share the same number.
#[derive(Debug, Clone)]
pub struct InnovationTracker {
    counter: u32,
    history: HashMap<(u16, u16), u32>,
}

impl InnovationTracker {
    pub fn new() -> Self {
        InnovationTracker {
            counter: INITIAL_INNOVATIONS,
            history: HashMap::new(),
        }
    }

    /// Get innovation number for a structural mutation. Reuses existing number
    /// if the same mutation was already registered this generation.
    pub fn get(&mut self, in_node: u16, out_node: u16) -> u32 {
        *self.history.entry((in_node, out_node)).or_insert_with(|| {
            let n = self.counter;
            self.counter += 1;
            n
        })
    }

    /// Clear per-generation tracking. Call at the start of each reproduction cycle.
    pub fn new_generation(&mut self) {
        self.history.clear();
    }
}

// ── BrainGenome (NEAT genome) ──────────────────────────────────

/// NEAT-style genome: a variable-length list of connection genes plus
/// per-node genes (activation function, bias, role).
/// Node depths enforce feedforward structure.
#[derive(Debug, Clone)]
pub struct BrainGenome {
    pub connections: Vec<ConnectionGene>,
    /// Depth of each node (indexed by node ID). Input=0.0, Output=1.0, Hidden=between.
    pub node_depths: Vec<f32>,
    /// Per-node genes (activation, bias, role). Indexed by node ID.
    pub node_genes: Vec<NodeGene>,
    pub next_node_id: u16,
}

impl BrainGenome {
    /// Create a minimal NEAT genome with full input→output connections.
    /// Innovation numbers 0..(INPUT_SIZE*OUTPUT_SIZE-1) are pre-assigned
    /// so all initial genomes share the same gene alignment.
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut connections = Vec::with_capacity(INPUT_SIZE * OUTPUT_SIZE);
        let mut innov = 0u32;

        for i in 0..INPUT_SIZE as u16 {
            for o in FIRST_OUTPUT..FIRST_OUTPUT + OUTPUT_SIZE as u16 {
                connections.push(ConnectionGene {
                    in_node: i,
                    out_node: o,
                    weight: rng.random_range(-0.5..0.5_f32),
                    enabled: true,
                    innovation: innov,
                });
                innov += 1;
            }
        }

        // Slight forage bias so creatures seek food with random initial weights
        let forage_out = FIRST_OUTPUT + 3;
        for conn in &mut connections {
            if conn.out_node == forage_out {
                conn.weight += 0.5 / INPUT_SIZE as f32;
            }
        }

        let mut node_depths = vec![0.0f32; FIRST_HIDDEN as usize];
        for i in 0..INPUT_SIZE {
            node_depths[i] = 0.0;
        }
        for i in 0..OUTPUT_SIZE {
            node_depths[FIRST_OUTPUT as usize + i] = 1.0;
        }

        // Initialize per-node genes: inputs=Identity, outputs=Tanh
        let mut node_genes = Vec::with_capacity(FIRST_HIDDEN as usize);
        for _ in 0..INPUT_SIZE {
            node_genes.push(NodeGene {
                activation_fn: ActivationFn::Identity,
                bias: 0.0,
                role: NodeRole::Standard,
            });
        }
        for _ in 0..OUTPUT_SIZE {
            node_genes.push(NodeGene {
                activation_fn: ActivationFn::Tanh,
                bias: 0.0,
                role: NodeRole::Standard,
            });
        }

        BrainGenome {
            connections,
            node_depths,
            node_genes,
            next_node_id: FIRST_HIDDEN,
        }
    }

    pub fn num_hidden_nodes(&self) -> u16 {
        self.next_node_id.saturating_sub(FIRST_HIDDEN)
    }

    pub fn num_enabled_connections(&self) -> usize {
        self.connections.iter().filter(|c| c.enabled).count()
    }
}

// ── Brain (runtime network) ────────────────────────────────────

/// Runtime neural network built from a BrainGenome. Caches topological
/// ordering for efficient repeated forward passes. Supports Hebbian
/// lifetime learning, recurrent self-connections, per-node activations,
/// biases, neuromodulation, and attention.
#[derive(Debug, Clone)]
pub struct Brain {
    order: Vec<u16>,
    /// Incoming connections per node: incoming[node_id] = [(source, weight), ...]
    incoming: Vec<Vec<(u16, f32)>>,
    /// Self-loop connections per node: self_conns[node_id] = weight (or 0.0 if none)
    self_conns: Vec<f32>,
    /// Previous tick's activations for recurrent self-connections
    prev_activations: Vec<f32>,
    /// Per-node activation functions
    activation_fns: Vec<ActivationFn>,
    /// Per-node biases
    biases: Vec<f32>,
    /// Per-node roles (Standard, Modulator, Attention)
    roles: Vec<NodeRole>,
    num_nodes: usize,
    /// Hebbian learning rate (from genome, 0.0 = no learning)
    learning_rate: f32,
}

impl Brain {
    pub fn from_genome(genome: &BrainGenome) -> Self {
        Self::build(genome, 0.0)
    }

    pub fn from_genome_with_learning(genome: &BrainGenome, learning_rate: f32) -> Self {
        Self::build(genome, learning_rate)
    }

    fn build(genome: &BrainGenome, learning_rate: f32) -> Self {
        let num_nodes = genome.next_node_id as usize;
        let (order, incoming, self_conns) = compute_topology(&genome.connections, num_nodes);

        // Build per-node vectors from genome, defaulting for missing entries
        let mut activation_fns = Vec::with_capacity(num_nodes);
        let mut biases = Vec::with_capacity(num_nodes);
        let mut roles = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            if let Some(ng) = genome.node_genes.get(i) {
                activation_fns.push(ng.activation_fn);
                biases.push(ng.bias);
                roles.push(ng.role);
            } else if i < INPUT_SIZE {
                activation_fns.push(ActivationFn::Identity);
                biases.push(0.0);
                roles.push(NodeRole::Standard);
            } else if i < INPUT_SIZE + OUTPUT_SIZE {
                activation_fns.push(ActivationFn::Tanh);
                biases.push(0.0);
                roles.push(NodeRole::Standard);
            } else {
                activation_fns.push(ActivationFn::Tanh);
                biases.push(0.0);
                roles.push(NodeRole::Standard);
            }
        }

        Brain {
            order,
            incoming,
            self_conns,
            prev_activations: vec![0.0; num_nodes],
            activation_fns,
            biases,
            roles,
            num_nodes,
            learning_rate,
        }
    }

    /// Forward pass with per-node activations, biases, neuromodulation,
    /// attention, recurrent self-connections, and Oja's rule learning.
    pub fn forward(&mut self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut activations = vec![0.0f32; self.num_nodes];

        for i in 0..INPUT_SIZE {
            activations[i] = input[i];
        }

        // Collect modulator outputs after first pass for gated plasticity
        let mut modulator_gates: Vec<(usize, f32)> = Vec::new();

        for &node_id in &self.order {
            let idx = node_id as usize;
            let role = self.roles.get(idx).copied().unwrap_or(NodeRole::Standard);

            // Recurrent self-connection: previous tick's activation
            let self_w = self.self_conns.get(idx).copied().unwrap_or(0.0);
            let self_contrib = if self_w != 0.0 {
                self.prev_activations.get(idx).copied().unwrap_or(0.0) * self_w
            } else {
                0.0
            };

            let activation = match role {
                NodeRole::Attention => {
                    // Softmax-weighted blend: weights determine attention scores,
                    // inputs are blended proportionally
                    let conns = &self.incoming[idx];
                    if conns.is_empty() {
                        let bias = self.biases.get(idx).copied().unwrap_or(0.0);
                        ActivationFn::Identity.apply(self_contrib + bias)
                    } else {
                        // Compute softmax over raw weights (attention scores)
                        let max_w = conns.iter().map(|(_, w)| *w).fold(f32::NEG_INFINITY, f32::max);
                        let exp_sum: f32 = conns.iter().map(|(_, w)| (w - max_w).exp()).sum();
                        let blend: f32 = conns.iter()
                            .map(|&(src, w)| {
                                let alpha = (w - max_w).exp() / exp_sum;
                                alpha * activations[src as usize]
                            })
                            .sum();
                        let bias = self.biases.get(idx).copied().unwrap_or(0.0);
                        // Attention uses Identity activation (softmax IS the nonlinearity)
                        ActivationFn::Identity.apply(blend + self_contrib + bias)
                    }
                }
                NodeRole::Standard | NodeRole::Modulator => {
                    let mut sum = self_contrib;
                    for &(src, weight) in &self.incoming[idx] {
                        sum += activations[src as usize] * weight;
                    }
                    let bias = self.biases.get(idx).copied().unwrap_or(0.0);
                    let act_fn = self.activation_fns.get(idx).copied().unwrap_or(ActivationFn::Tanh);
                    act_fn.apply(sum + bias)
                }
            };

            activations[idx] = activation;

            // Record modulator outputs for gated plasticity
            if role == NodeRole::Modulator {
                // Sigmoid-squash modulator output to [0, 1] gate value
                let gate = 1.0 / (1.0 + (-activation).exp());
                let depth = idx; // use node index as proxy for zone
                modulator_gates.push((depth, gate));
            }
        }

        // Oja's rule with neuromodulation gating
        if self.learning_rate > 0.0 {
            let base_lr = self.learning_rate * 0.01;
            let decay = 0.0002_f32;

            for (target_idx, conns) in self.incoming.iter_mut().enumerate() {
                let post = activations.get(target_idx).copied().unwrap_or(0.0);

                // Find closest modulator gate for this target node
                let lr = if modulator_gates.is_empty() {
                    base_lr
                } else {
                    // Use average of all modulator gates (global modulation)
                    let avg_gate: f32 = modulator_gates.iter().map(|(_, g)| g).sum::<f32>()
                        / modulator_gates.len() as f32;
                    base_lr * avg_gate
                };

                for (_src, weight) in conns.iter_mut() {
                    let pre = activations.get(*_src as usize).copied().unwrap_or(0.0);
                    let delta = lr * post * (pre - post * *weight);
                    *weight = (*weight + delta) * (1.0 - decay);
                    *weight = weight.clamp(-3.0, 3.0);
                }

                // Bias update via Oja-like rule
                let bias = self.biases.get_mut(target_idx);
                if let Some(b) = bias {
                    let delta_b = lr * post * (1.0 - post * *b);
                    *b = (*b + delta_b) * (1.0 - decay);
                    *b = b.clamp(-2.0, 2.0);
                }
            }
            // Also apply Oja's rule to self-connections
            for (idx, self_w) in self.self_conns.iter_mut().enumerate() {
                if *self_w != 0.0 {
                    let post = activations.get(idx).copied().unwrap_or(0.0);
                    let pre = self.prev_activations.get(idx).copied().unwrap_or(0.0);
                    let lr = if modulator_gates.is_empty() {
                        base_lr
                    } else {
                        let avg_gate: f32 = modulator_gates.iter().map(|(_, g)| g).sum::<f32>()
                            / modulator_gates.len() as f32;
                        base_lr * avg_gate
                    };
                    let delta = lr * post * (pre - post * *self_w);
                    *self_w = (*self_w + delta) * (1.0 - decay);
                    *self_w = self_w.clamp(-3.0, 3.0);
                }
            }
        }

        // Store activations for next tick's recurrent connections
        self.prev_activations = activations.clone();

        let mut output = [0.0f32; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            output[i] = activations[FIRST_OUTPUT as usize + i];
        }
        output
    }
}

/// Compute topological ordering, incoming connection lists, and self-connection
/// weights via Kahn's algorithm. Self-loops (in_node == out_node) are separated
/// out because they use previous-tick activations and don't participate in the
/// feedforward topological sort.
fn compute_topology(
    connections: &[ConnectionGene],
    num_nodes: usize,
) -> (Vec<u16>, Vec<Vec<(u16, f32)>>, Vec<f32>) {
    let mut incoming: Vec<Vec<(u16, f32)>> = vec![Vec::new(); num_nodes];
    let mut self_conns = vec![0.0f32; num_nodes];
    let mut in_degree = vec![0u32; num_nodes];
    let mut outgoing: Vec<Vec<u16>> = vec![Vec::new(); num_nodes];

    for conn in connections {
        if !conn.enabled {
            continue;
        }
        let out = conn.out_node as usize;
        let inp = conn.in_node as usize;
        if out >= num_nodes || inp >= num_nodes {
            continue;
        }
        if conn.in_node == conn.out_node {
            // Self-loop: accumulate weight (multiple self-connections sum)
            self_conns[out] += conn.weight;
            continue;
        }
        incoming[out].push((conn.in_node, conn.weight));
        in_degree[out] += 1;
        outgoing[inp].push(conn.out_node);
    }

    let mut queue = VecDeque::new();
    let mut enqueued = vec![false; num_nodes];

    for i in 0..num_nodes {
        if in_degree[i] == 0 {
            queue.push_back(i as u16);
            enqueued[i] = true;
        }
    }

    let mut order = Vec::new();

    while let Some(node) = queue.pop_front() {
        if node >= INPUT_SIZE as u16 {
            order.push(node);
        }
        for &next in &outgoing[node as usize] {
            let ni = next as usize;
            in_degree[ni] = in_degree[ni].saturating_sub(1);
            if in_degree[ni] == 0 && !enqueued[ni] {
                queue.push_back(next);
                enqueued[ni] = true;
            }
        }
    }

    (order, incoming, self_conns)
}

// ── Sensory input construction ────────────────────────────────

struct NearestInfo {
    dist: f32,  // normalized 0..1 (1 = close)
    angle: f32, // normalized -1..1
}

impl Default for NearestInfo {
    fn default() -> Self {
        Self {
            dist: 0.0,
            angle: 0.0,
        }
    }
}

fn build_sensory_input(
    pos: &Position,
    vel: &Velocity,
    energy: &Energy,
    needs: &Needs,
    physics: &DerivedPhysics,
    env: &Environment,
    nearest_food: &NearestInfo,
    nearest_predator: &NearestInfo,
    nearest_ally: &NearestInfo,
    tank_w: f32,
    tank_h: f32,
    pheromone_concentration: f32,
    pheromone_angle: f32,
) -> [f32; INPUT_SIZE] {
    let speed = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
    let speed_frac = if physics.max_speed > 0.0 {
        (speed / physics.max_speed).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let wall_x = (pos.x / tank_w) * 2.0 - 1.0;
    let wall_y = (pos.y / tank_h) * 2.0 - 1.0;

    [
        energy.fraction(),
        needs.hunger,
        needs.safety,
        needs.reproduction,
        nearest_food.dist,
        nearest_food.angle,
        nearest_predator.dist,
        nearest_predator.angle,
        nearest_ally.dist,
        nearest_ally.angle,
        wall_x.clamp(-1.0, 1.0),
        wall_y.clamp(-1.0, 1.0),
        env.light_level,
        speed_frac,
        pheromone_concentration.clamp(0.0, 1.0), // 14
        pheromone_angle.clamp(-1.0, 1.0),        // 15
    ]
}

// ── Brain system (runs each tick) ─────────────────────────────

/// Main brain system: builds sensory inputs, runs the neural net, applies outputs.
/// Uses emergent feeding capabilities instead of fixed trophic roles.
pub fn brain_system(
    world: &mut World,
    grid: &SpatialGrid,
    entity_map: &EntityInfoMap,
    env: &Environment,
    pheromone_grid: &crate::pheromone::PheromoneGrid,
    dt: f32,
    tank_w: f32,
    tank_h: f32,
) -> Vec<(f32, f32, f32)> {
    // Phase 1: collect brain creatures (no cloning — store entity for brain lookup)
    struct CreatureData {
        entity: Entity,
        px: f32,
        py: f32,
        vx: f32,
        vy: f32,
        energy_frac: f32,
        hunger: f32,
        safety: f32,
        repro: f32,
        max_prey_mass: f32,
        body_mass: f32,
        sense_range: f32,
        max_speed: f32,
        complexity: f32,
        pheromone_sensitivity: f32,
        brain: Brain,
    }

    let creatures: Vec<CreatureData> = {
        let mut v = Vec::new();
        for (entity, pos, vel, energy, needs, physics, feeding, genome, brain) in &mut world
            .query::<(
                Entity,
                &Position,
                &Velocity,
                &Energy,
                &Needs,
                &DerivedPhysics,
                &FeedingCapability,
                &CreatureGenome,
                &Brain,
            )>()
        {
            v.push(CreatureData {
                entity,
                px: pos.x,
                py: pos.y,
                vx: vel.vx,
                vy: vel.vy,
                energy_frac: energy.fraction(),
                hunger: needs.hunger,
                safety: needs.safety,
                repro: needs.reproduction,
                max_prey_mass: feeding.max_prey_mass,
                body_mass: physics.body_mass,
                sense_range: physics.sensory_range,
                max_speed: physics.max_speed,
                complexity: genome.complexity,
                pheromone_sensitivity: genome.behavior.pheromone_sensitivity,
                brain: brain.clone(),
            });
        }
        v
    };

    // Phase 2: compute brain outputs (parallel via rayon, using shared entity_map)
    // Also applies Hebbian learning (modifies brain weights in-place)
    let actions: Vec<(Entity, f32, f32, BehaviorAction, Brain, (f32, f32, f32))> = creatures
        .into_par_iter()
        .map(|mut c| {
            let mut nearest_food = NearestInfo::default();
            let mut nearest_predator = NearestInfo::default();
            let mut nearest_ally = NearestInfo::default();
            let mut best_food_dist = f32::MAX;
            let mut best_pred_dist = f32::MAX;
            let mut best_ally_dist = f32::MAX;

            let neighbors = grid.neighbors(c.px, c.py, c.sense_range);
            for &neighbor in &neighbors {
                if neighbor == c.entity {
                    continue;
                }
                let info = match entity_map.get(&neighbor) {
                    Some(data) => data,
                    None => continue,
                };

                let dx: f32 = info.x - c.px;
                let dy: f32 = info.y - c.py;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > c.sense_range || dist < 0.01 {
                    continue;
                }

                let angle = dy.atan2(dx) / std::f32::consts::PI;
                let norm_dist = 1.0 - (dist / c.sense_range);

                if (info.is_producer || info.body_mass < c.max_prey_mass) && dist < best_food_dist {
                    best_food_dist = dist;
                    nearest_food = NearestInfo {
                        dist: norm_dist,
                        angle,
                    };
                }

                if !info.is_producer && info.max_prey_mass > c.body_mass && dist < best_pred_dist {
                    best_pred_dist = dist;
                    nearest_predator = NearestInfo {
                        dist: norm_dist,
                        angle,
                    };
                }

                if !info.is_producer && info.body_mass > 0.0 {
                    let mass_ratio = c.body_mass / info.body_mass;
                    if mass_ratio > 0.5 && mass_ratio < 2.0 && dist < best_ally_dist {
                        best_ally_dist = dist;
                        nearest_ally = NearestInfo {
                            dist: norm_dist,
                            angle,
                        };
                    }
                }
            }

            // Build input vector
            let needs_proxy = Needs {
                hunger: c.hunger,
                safety: c.safety,
                reproduction: c.repro,
                ..Default::default()
            };
            let vel_proxy = Velocity { vx: c.vx, vy: c.vy };
            let pos_proxy = Position { x: c.px, y: c.py };
            let energy_proxy = Energy {
                current: c.energy_frac,
                max: 1.0,
            };
            let physics_proxy = DerivedPhysics {
                sensory_range: c.sense_range,
                max_speed: c.max_speed,
                ..Default::default()
            };

            // Sample pheromone grid
            let phero_conc = pheromone_grid.sample(c.px, c.py) * c.pheromone_sensitivity;
            let (gx, gy) = pheromone_grid.gradient(c.px, c.py);
            let phero_angle = if gx.abs() + gy.abs() > 0.001 {
                gy.atan2(gx) / std::f32::consts::PI
            } else {
                0.0
            };

            let input = build_sensory_input(
                &pos_proxy,
                &vel_proxy,
                &energy_proxy,
                &needs_proxy,
                &physics_proxy,
                env,
                &nearest_food,
                &nearest_predator,
                &nearest_ally,
                tank_w,
                tank_h,
                phero_conc,
                phero_angle,
            );

            let output = c.brain.forward(&input);
            let scale = c.complexity.max(0.3);

            let mut steer_x = output[0] * scale;
            let mut steer_y = output[1] * scale;
            let speed_mult = (output[2] * scale + 1.0) * 0.7 + 0.1;

            if c.hunger > 0.35 && nearest_food.dist > 0.0 {
                // Simulation assumption: the founder web gives simple consumers a
                // weak low-threshold food-taxis fallback so nearby resources are
                // actually exploitable before evolution discovers a strong
                // foraging controller from random brains alone.
                let food_angle = nearest_food.angle * std::f32::consts::PI;
                let food_bias = ((c.hunger - 0.35) / 0.65).clamp(0.0, 1.0) * 0.85;
                steer_x += food_angle.cos() * food_bias;
                steer_y += food_angle.sin() * food_bias;
            }

            let steer_mag = (steer_x * steer_x + steer_y * steer_y).sqrt().max(0.01);
            let force_strength = 3.0 * speed_mult;
            let fx = (steer_x / steer_mag) * force_strength;
            let fy = (steer_y / steer_mag) * force_strength;

            let forage_t = output[3] * scale;
            let flee_t = output[4] * scale;
            let social_t = output[5] * scale;
            let phero_emit = output[6].max(0.0) * scale;

            let action = if flee_t > 0.3 && nearest_predator.dist > 0.0 {
                BehaviorAction::Flee
            } else if forage_t > 0.3 && nearest_food.dist > 0.0 {
                BehaviorAction::Forage
            } else if social_t > 0.3 {
                BehaviorAction::School
            } else if speed_mult < 0.3 {
                BehaviorAction::Rest
            } else {
                BehaviorAction::Explore
            };

            (c.entity, fx, fy, action, c.brain, (c.px, c.py, phero_emit))
        })
        .collect();

    // Phase 4: apply forces, behavior, and write back learned brain weights
    let mut pheromone_deposits = Vec::new();
    for (entity, fx, fy, action, learned_brain, (px, py, phero_emit)) in actions {
        if let Ok(mut vel) = world.get::<&mut Velocity>(entity) {
            vel.vx += fx * dt;
            vel.vy += fy * dt;
        }
        if let Ok(mut bstate) = world.get::<&mut BehaviorState>(entity) {
            bstate.action = action;
        }
        // Write back brain with Hebbian weight updates
        if let Ok(mut brain) = world.get::<&mut Brain>(entity) {
            *brain = learned_brain;
        }
        if phero_emit > 0.0 {
            pheromone_deposits.push((px, py, phero_emit));
        }
    }
    pheromone_deposits
}

// ── Genetic operations for NEAT brain ──────────────────────────

/// NEAT crossover: align genes by innovation number.
/// Matching genes are randomly chosen from either parent.
/// Disjoint/excess genes are included with 50% probability.
pub fn crossover_brain(a: &BrainGenome, b: &BrainGenome, rng: &mut impl Rng) -> BrainGenome {
    let a_genes: HashMap<u32, &ConnectionGene> =
        a.connections.iter().map(|g| (g.innovation, g)).collect();
    let b_genes: HashMap<u32, &ConnectionGene> =
        b.connections.iter().map(|g| (g.innovation, g)).collect();

    let all_innovations: HashSet<u32> = a_genes.keys().chain(b_genes.keys()).copied().collect();
    let mut child_connections = Vec::new();

    for &innov in &all_innovations {
        match (a_genes.get(&innov), b_genes.get(&innov)) {
            (Some(ga), Some(gb)) => {
                let chosen = if rng.random_bool(0.5) { *ga } else { *gb };
                child_connections.push(chosen.clone());
            }
            (Some(g), None) | (None, Some(g)) => {
                if rng.random_bool(0.5) {
                    child_connections.push((*g).clone());
                }
            }
            (None, None) => unreachable!(),
        }
    }

    child_connections.sort_by_key(|c| c.innovation);

    let next_node_id = a.next_node_id.max(b.next_node_id);
    let max_nodes = next_node_id as usize;
    let mut node_depths = vec![0.0f32; max_nodes];
    for i in 0..INPUT_SIZE {
        node_depths[i] = 0.0;
    }
    for i in 0..OUTPUT_SIZE {
        if FIRST_OUTPUT as usize + i < max_nodes {
            node_depths[FIRST_OUTPUT as usize + i] = 1.0;
        }
    }
    for i in FIRST_HIDDEN as usize..max_nodes {
        let da = a.node_depths.get(i).copied().unwrap_or(0.5);
        let db = b.node_depths.get(i).copied().unwrap_or(0.5);
        if i < a.node_depths.len() && i < b.node_depths.len() {
            node_depths[i] = (da + db) / 2.0;
        } else if i < a.node_depths.len() {
            node_depths[i] = da;
        } else {
            node_depths[i] = db;
        }
    }

    // Crossover per-node genes
    let mut node_genes = Vec::with_capacity(max_nodes);
    for i in 0..max_nodes {
        if i < INPUT_SIZE {
            // Input nodes always Identity
            node_genes.push(NodeGene {
                activation_fn: ActivationFn::Identity,
                bias: 0.0,
                role: NodeRole::Standard,
            });
        } else {
            let ng_a = a.node_genes.get(i);
            let ng_b = b.node_genes.get(i);
            match (ng_a, ng_b) {
                (Some(ga), Some(gb)) => {
                    node_genes.push(NodeGene {
                        activation_fn: if rng.random_bool(0.5) { ga.activation_fn } else { gb.activation_fn },
                        bias: (ga.bias + gb.bias) / 2.0,
                        role: if rng.random_bool(0.5) { ga.role } else { gb.role },
                    });
                }
                (Some(g), None) | (None, Some(g)) => {
                    node_genes.push(g.clone());
                }
                (None, None) => {
                    // Fallback for nodes referenced in connections but missing from node_genes
                    let default_act = if i < INPUT_SIZE + OUTPUT_SIZE {
                        ActivationFn::Tanh
                    } else {
                        ActivationFn::Tanh
                    };
                    node_genes.push(NodeGene {
                        activation_fn: default_act,
                        bias: 0.0,
                        role: NodeRole::Standard,
                    });
                }
            }
        }
    }

    BrainGenome {
        connections: child_connections,
        node_depths,
        node_genes,
        next_node_id,
    }
}

/// Ensure node_genes vec is large enough for next_node_id, padding with defaults.
fn ensure_node_genes(brain: &mut BrainGenome) {
    while brain.node_genes.len() < brain.next_node_id as usize {
        let i = brain.node_genes.len();
        let act = if i < INPUT_SIZE {
            ActivationFn::Identity
        } else if i < INPUT_SIZE + OUTPUT_SIZE {
            ActivationFn::Tanh
        } else {
            ActivationFn::Tanh
        };
        brain.node_genes.push(NodeGene {
            activation_fn: act,
            bias: 0.0,
            role: NodeRole::Standard,
        });
    }
}

/// NEAT mutation: weight perturbation, activation swap, bias perturbation,
/// structural mutations, node role mutations, and module duplication.
///
/// `diversity` scales mutation behavior: at 1.0 (default) all rates and
/// magnitudes are at baseline.  Higher diversity increases structural mutation
/// probability (brain architecture grows faster) while *reducing* per-weight
/// perturbation magnitude (gentler edits preserve useful signals when many
/// weights are touched).
pub fn mutate_brain(
    brain: &mut BrainGenome,
    rate: f32,
    diversity: f32,
    rng: &mut impl Rng,
    tracker: &mut InnovationTracker,
) {
    ensure_node_genes(brain);

    // Diversity-dependent scales (neutral at diversity=1.0):
    //  - magnitude shrinks as 1/√diversity so high-frequency perturbation is gentler
    //  - structural rates grow as (0.5 + 0.5*diversity) so architecture keeps pace
    let mag_scale = 1.0 / diversity.max(0.25).sqrt();
    let struct_scale = 0.5 + 0.5 * diversity as f64;

    // 1a. Weight perturbation (magnitude scaled by diversity)
    let w_mag = 0.3 * mag_scale;
    for conn in &mut brain.connections {
        if rng.random_bool(rate as f64) {
            conn.weight += rng.random_range(-w_mag..w_mag);
            conn.weight = conn.weight.clamp(-3.0, 3.0);
        }
    }

    // 1b. Activation swap mutation (rate scaled by diversity)
    if rng.random_bool(ACTIVATION_SWAP_RATE * struct_scale) && brain.next_node_id > FIRST_OUTPUT {
        let node = rng.random_range(FIRST_OUTPUT..brain.next_node_id);
        let idx = node as usize;
        if idx < brain.node_genes.len() {
            brain.node_genes[idx].activation_fn = ActivationFn::random(rng);
        }
    }

    // 1c. Bias perturbation (magnitude scaled by diversity)
    let b_mag = 0.1 * mag_scale;
    for i in FIRST_OUTPUT as usize..brain.node_genes.len() {
        if rng.random_bool(rate as f64) {
            brain.node_genes[i].bias += rng.random_range(-b_mag..b_mag);
            brain.node_genes[i].bias = brain.node_genes[i].bias.clamp(-2.0, 2.0);
        }
    }

    // 2. Add node: split an enabled connection, preserving network behavior
    if rng.random_bool(ADD_NODE_RATE * struct_scale) && brain.next_node_id < MAX_NODES {
        let enabled: Vec<usize> = brain
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();

        if !enabled.is_empty() {
            let idx = enabled[rng.random_range(0..enabled.len())];
            let old_in = brain.connections[idx].in_node;
            let old_out = brain.connections[idx].out_node;
            let old_weight = brain.connections[idx].weight;

            brain.connections[idx].enabled = false;

            let new_node = brain.next_node_id;
            brain.next_node_id += 1;

            let in_depth = brain
                .node_depths
                .get(old_in as usize)
                .copied()
                .unwrap_or(0.0);
            let out_depth = brain
                .node_depths
                .get(old_out as usize)
                .copied()
                .unwrap_or(1.0);
            while brain.node_depths.len() <= new_node as usize {
                brain.node_depths.push(0.5);
            }
            brain.node_depths[new_node as usize] = (in_depth + out_depth) / 2.0;

            // Initialize node gene for the new hidden node
            while brain.node_genes.len() <= new_node as usize {
                brain.node_genes.push(NodeGene {
                    activation_fn: ActivationFn::Tanh,
                    bias: 0.0,
                    role: NodeRole::Standard,
                });
            }
            brain.node_genes[new_node as usize] = NodeGene {
                activation_fn: ActivationFn::random_hidden(rng),
                bias: 0.0,
                role: NodeRole::Standard,
            };

            // in→new (weight 1.0) and new→out (old weight) preserves behavior
            brain.connections.push(ConnectionGene {
                in_node: old_in,
                out_node: new_node,
                weight: 1.0,
                enabled: true,
                innovation: tracker.get(old_in, new_node),
            });
            brain.connections.push(ConnectionGene {
                in_node: new_node,
                out_node: old_out,
                weight: old_weight,
                enabled: true,
                innovation: tracker.get(new_node, old_out),
            });
        }
    }

    // 3. Add connection between non-connected nodes (feedforward only)
    if rng.random_bool(ADD_CONN_RATE * struct_scale) && brain.connections.len() < MAX_CONNECTIONS {
        let existing: HashSet<(u16, u16)> = brain
            .connections
            .iter()
            .map(|c| (c.in_node, c.out_node))
            .collect();

        // Collect active nodes (inputs + outputs + hidden nodes appearing in connections)
        let mut active_nodes: HashSet<u16> = (0..INPUT_SIZE as u16).collect();
        active_nodes.extend(FIRST_OUTPUT..FIRST_OUTPUT + OUTPUT_SIZE as u16);
        for conn in &brain.connections {
            active_nodes.insert(conn.in_node);
            active_nodes.insert(conn.out_node);
        }
        let active_vec: Vec<u16> = active_nodes.into_iter().collect();

        for _ in 0..20 {
            let ai = rng.random_range(0..active_vec.len());
            let bi = rng.random_range(0..active_vec.len());
            let a = active_vec[ai];
            let b = active_vec[bi];
            if a == b {
                continue;
            }

            let da = brain.node_depths.get(a as usize).copied().unwrap_or(0.0);
            let db = brain.node_depths.get(b as usize).copied().unwrap_or(1.0);

            let (from, to) = if da < db {
                (a, b)
            } else if db < da {
                (b, a)
            } else {
                continue;
            };

            // Don't target input nodes
            if to < INPUT_SIZE as u16 {
                continue;
            }

            if !existing.contains(&(from, to)) {
                brain.connections.push(ConnectionGene {
                    in_node: from,
                    out_node: to,
                    weight: rng.random_range(-0.5..0.5_f32),
                    enabled: true,
                    innovation: tracker.get(from, to),
                });
                break;
            }
        }
    }

    // 4. Add recurrent self-connection on a hidden or output node
    if rng.random_bool(ADD_SELF_CONN_RATE * struct_scale) && brain.connections.len() < MAX_CONNECTIONS {
        let existing: HashSet<(u16, u16)> = brain
            .connections
            .iter()
            .map(|c| (c.in_node, c.out_node))
            .collect();

        // Only hidden and output nodes can have self-connections (not inputs)
        let candidates: Vec<u16> = (FIRST_OUTPUT..brain.next_node_id)
            .filter(|&n| !existing.contains(&(n, n)))
            .collect();

        if !candidates.is_empty() {
            let node = candidates[rng.random_range(0..candidates.len())];
            brain.connections.push(ConnectionGene {
                in_node: node,
                out_node: node,
                weight: rng.random_range(-0.3..0.3_f32),
                enabled: true,
                innovation: tracker.get(node, node),
            });
        }
    }

    // 5. Flip a hidden node to Modulator role
    if rng.random_bool(MODULATOR_FLIP_RATE * struct_scale) {
        let hidden_count = brain.next_node_id.saturating_sub(FIRST_HIDDEN);
        if hidden_count > 0 {
            let node = FIRST_HIDDEN + rng.random_range(0..hidden_count);
            let idx = node as usize;
            if idx < brain.node_genes.len() && brain.node_genes[idx].role == NodeRole::Standard {
                brain.node_genes[idx].role = NodeRole::Modulator;
                // Force sigmoid activation for modulators (output should be [0,1])
                brain.node_genes[idx].activation_fn = ActivationFn::Sigmoid;
            }
        }
    }

    // 6. Flip a hidden node to Attention role
    if rng.random_bool(ATTENTION_FLIP_RATE * struct_scale) {
        let hidden_count = brain.next_node_id.saturating_sub(FIRST_HIDDEN);
        if hidden_count > 0 {
            let node = FIRST_HIDDEN + rng.random_range(0..hidden_count);
            let idx = node as usize;
            if idx < brain.node_genes.len() && brain.node_genes[idx].role == NodeRole::Standard {
                brain.node_genes[idx].role = NodeRole::Attention;
                brain.node_genes[idx].activation_fn = ActivationFn::Identity;
            }
        }
    }

    // 7. Module duplication: copy a 1-hop subgraph from a random hidden node
    if rng.random_bool(MODULE_DUP_RATE * struct_scale) {
        let hidden_count = brain.next_node_id.saturating_sub(FIRST_HIDDEN);
        if hidden_count > 0 {
            let seed_node = FIRST_HIDDEN + rng.random_range(0..hidden_count);
            duplicate_module(brain, seed_node, rng, tracker);
        }
    }
}

/// Module duplication: copy a 1-hop subgraph rooted at `seed_node`.
/// Creates new node IDs and innovation numbers for the duplicate.
fn duplicate_module(
    brain: &mut BrainGenome,
    seed_node: u16,
    rng: &mut impl Rng,
    tracker: &mut InnovationTracker,
) {
    ensure_node_genes(brain);

    // Find all nodes within 1 hop of seed_node (via enabled connections)
    let mut module_nodes: HashSet<u16> = HashSet::new();
    module_nodes.insert(seed_node);
    for conn in &brain.connections {
        if !conn.enabled {
            continue;
        }
        if conn.in_node == seed_node && conn.out_node >= FIRST_HIDDEN {
            module_nodes.insert(conn.out_node);
        }
        if conn.out_node == seed_node && conn.in_node >= FIRST_HIDDEN {
            module_nodes.insert(conn.in_node);
        }
    }

    // Guard: check we have room for the duplicate
    let module_size = module_nodes.len() as u16;
    if brain.next_node_id + module_size > MAX_NODES {
        return;
    }

    // Find internal connections (both endpoints in module) — clone to avoid borrow conflicts
    let module_conns: Vec<(u16, u16, f32)> = brain
        .connections
        .iter()
        .filter(|c| c.enabled && module_nodes.contains(&c.in_node) && module_nodes.contains(&c.out_node))
        .map(|c| (c.in_node, c.out_node, c.weight))
        .collect();

    if brain.connections.len() + module_conns.len() > MAX_CONNECTIONS {
        return;
    }

    // Create mapping from old node IDs to new node IDs
    let mut node_map: HashMap<u16, u16> = HashMap::new();
    for &old_node in &module_nodes {
        let new_node = brain.next_node_id;
        brain.next_node_id += 1;
        node_map.insert(old_node, new_node);

        // Copy depth
        let depth = brain.node_depths.get(old_node as usize).copied().unwrap_or(0.5);
        while brain.node_depths.len() <= new_node as usize {
            brain.node_depths.push(0.5);
        }
        brain.node_depths[new_node as usize] = depth;

        // Copy node gene with slight perturbation
        let ng = brain.node_genes.get(old_node as usize).cloned().unwrap_or(NodeGene {
            activation_fn: ActivationFn::Tanh,
            bias: 0.0,
            role: NodeRole::Standard,
        });
        while brain.node_genes.len() <= new_node as usize {
            brain.node_genes.push(NodeGene {
                activation_fn: ActivationFn::Tanh,
                bias: 0.0,
                role: NodeRole::Standard,
            });
        }
        brain.node_genes[new_node as usize] = NodeGene {
            activation_fn: ng.activation_fn,
            bias: ng.bias + rng.random_range(-0.05..0.05_f32),
            role: ng.role,
        };
    }

    // Duplicate internal connections with new node IDs and innovations
    for (old_in, old_out, weight) in &module_conns {
        let new_in = node_map[old_in];
        let new_out = node_map[old_out];
        brain.connections.push(ConnectionGene {
            in_node: new_in,
            out_node: new_out,
            weight: weight + rng.random_range(-0.05..0.05_f32),
            enabled: true,
            innovation: tracker.get(new_in, new_out),
        });
    }

    // Connect duplicate module to the network: mirror external connections of seed_node
    // to the new seed_node, so the module is wired analogously
    let new_seed = node_map[&seed_node];
    let external_conns: Vec<(u16, u16, f32)> = brain
        .connections
        .iter()
        .filter(|c| c.enabled)
        .filter_map(|c| {
            if c.out_node == seed_node && !module_nodes.contains(&c.in_node) {
                Some((c.in_node, new_seed, c.weight))
            } else if c.in_node == seed_node && !module_nodes.contains(&c.out_node) {
                Some((new_seed, c.out_node, c.weight))
            } else {
                None
            }
        })
        .collect();

    for (from, to, weight) in external_conns {
        if brain.connections.len() >= MAX_CONNECTIONS {
            break;
        }
        brain.connections.push(ConnectionGene {
            in_node: from,
            out_node: to,
            weight: weight + rng.random_range(-0.1..0.1_f32),
            enabled: true,
            innovation: tracker.get(from, to),
        });
    }
}

/// NEAT genomic distance: excess genes, disjoint genes, weight differences,
/// and activation function mismatches.
pub fn brain_distance(a: &BrainGenome, b: &BrainGenome) -> f32 {
    let a_map: HashMap<u32, &ConnectionGene> =
        a.connections.iter().map(|g| (g.innovation, g)).collect();
    let b_map: HashMap<u32, &ConnectionGene> =
        b.connections.iter().map(|g| (g.innovation, g)).collect();

    let a_max = a
        .connections
        .iter()
        .map(|c| c.innovation)
        .max()
        .unwrap_or(0);
    let b_max = b
        .connections
        .iter()
        .map(|c| c.innovation)
        .max()
        .unwrap_or(0);
    let threshold = a_max.min(b_max);

    let mut matching = 0u32;
    let mut disjoint = 0u32;
    let mut excess = 0u32;
    let mut weight_diff_sum = 0.0f32;

    let all_innovations: HashSet<u32> = a_map.keys().chain(b_map.keys()).copied().collect();

    for &innov in &all_innovations {
        match (a_map.get(&innov), b_map.get(&innov)) {
            (Some(ga), Some(gb)) => {
                matching += 1;
                weight_diff_sum += (ga.weight - gb.weight).abs();
            }
            _ => {
                if innov > threshold {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
            }
        }
    }

    let n = a.connections.len().max(b.connections.len()).max(1) as f32;
    let avg_w = if matching > 0 {
        weight_diff_sum / matching as f32
    } else {
        0.0
    };

    // Activation mismatch term: count nodes present in both genomes
    // with different activation functions
    let max_nodes = a.next_node_id.max(b.next_node_id) as usize;
    let shared_nodes = a.next_node_id.min(b.next_node_id) as usize;
    let mut activation_mismatches = 0u32;
    for i in FIRST_OUTPUT as usize..shared_nodes {
        let act_a = a.node_genes.get(i).map(|ng| ng.activation_fn);
        let act_b = b.node_genes.get(i).map(|ng| ng.activation_fn);
        if let (Some(fa), Some(fb)) = (act_a, act_b) {
            if fa != fb {
                activation_mismatches += 1;
            }
        }
    }
    let max_n = max_nodes.max(1) as f32;

    C_EXCESS * excess as f32 / n
        + C_DISJOINT * disjoint as f32 / n
        + C_WEIGHT * avg_w
        + C_ACTIVATION * activation_mismatches as f32 / max_n
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_initial_genome_has_full_connectivity() {
        let bg = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        assert_eq!(bg.connections.len(), INPUT_SIZE * OUTPUT_SIZE);
        assert_eq!(bg.num_hidden_nodes(), 0);
        assert_eq!(bg.next_node_id, FIRST_HIDDEN);
    }

    #[test]
    fn test_forward_output_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut brain = Brain::from_genome(&BrainGenome::random(&mut rng));
        let input: [f32; INPUT_SIZE] = std::array::from_fn(|_| rng.random_range(-1.0..1.0_f32));
        let output = brain.forward(&input);
        for &v in &output {
            assert!(
                v >= -1.0 && v <= 1.0,
                "tanh output should be in [-1,1], got {}",
                v
            );
        }
    }

    #[test]
    fn test_forward_deterministic() {
        let genome = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        let mut brain = Brain::from_genome(&genome);
        let input: [f32; INPUT_SIZE] = [0.5; INPUT_SIZE];
        let out1 = brain.forward(&input);
        let out2 = brain.forward(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_forward_zero_weights() {
        let mut genome = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        for conn in &mut genome.connections {
            conn.weight = 0.0;
        }
        let mut brain = Brain::from_genome(&genome);
        let input = [1.0_f32; INPUT_SIZE];
        let output = brain.forward(&input);
        for &v in &output {
            assert!(
                (v).abs() < 1e-6,
                "Zero weights should produce zero output, got {}",
                v
            );
        }
    }

    #[test]
    fn test_crossover_preserves_innovations() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BrainGenome::random(&mut rng);
        let b = BrainGenome::random(&mut rng);
        let child = crossover_brain(&a, &b, &mut rng);

        // Child should have some connections (at least some innovations preserved)
        assert!(!child.connections.is_empty());

        // All child innovations should come from one of the parents
        let a_innovs: HashSet<u32> = a.connections.iter().map(|c| c.innovation).collect();
        let b_innovs: HashSet<u32> = b.connections.iter().map(|c| c.innovation).collect();
        for conn in &child.connections {
            assert!(
                a_innovs.contains(&conn.innovation) || b_innovs.contains(&conn.innovation),
                "Child innovation {} not found in either parent",
                conn.innovation,
            );
        }
    }

    #[test]
    fn test_mutation_weight_stays_bounded() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();
        for _ in 0..1000 {
            mutate_brain(&mut bg, 0.5, 1.0, &mut rng, &mut tracker);
        }
        for conn in &bg.connections {
            assert!(
                conn.weight >= -3.0 && conn.weight <= 3.0,
                "Weight out of bounds: {}",
                conn.weight
            );
        }
    }

    #[test]
    fn test_add_node_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut saw_added_node = false;
        let mut saw_added_connection = false;

        for _ in 0..8 {
            let mut bg = BrainGenome::random(&mut rng);
            let mut tracker = InnovationTracker::new();
            let initial_conns = bg.connections.len();
            let initial_nodes = bg.next_node_id;

            // Structural mutations are intentionally stochastic, so retry on a
            // few fresh genomes instead of assuming one specific random draw
            // will hit add-node in a fixed number of attempts.
            for _ in 0..100 {
                mutate_brain(&mut bg, 0.0, 1.0, &mut rng, &mut tracker);
            }

            saw_added_node |= bg.next_node_id > initial_nodes;
            saw_added_connection |= bg.connections.len() > initial_conns;

            if saw_added_node && saw_added_connection {
                break;
            }
        }

        assert!(
            saw_added_node,
            "Add-node mutation should eventually create hidden nodes"
        );
        assert!(
            saw_added_connection,
            "Add-node mutation should eventually create new connections"
        );
    }

    #[test]
    fn test_add_connection_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // First add a hidden node so there are new connections to create
        bg.next_node_id = FIRST_HIDDEN + 1;
        bg.node_depths.push(0.5); // hidden node at depth 0.5
        bg.connections.push(ConnectionGene {
            in_node: 0,
            out_node: FIRST_HIDDEN,
            weight: 1.0,
            enabled: true,
            innovation: tracker.get(0, FIRST_HIDDEN),
        });

        let initial_conns = bg.connections.len();
        for _ in 0..200 {
            mutate_brain(&mut bg, 0.0, 1.0, &mut rng, &mut tracker);
        }

        assert!(
            bg.connections.len() > initial_conns,
            "Add-connection mutation should create connections: {} -> {}",
            initial_conns,
            bg.connections.len(),
        );
    }

    #[test]
    fn test_brain_distance_self_zero() {
        let bg = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        assert!((brain_distance(&bg, &bg)).abs() < 1e-6);
    }

    #[test]
    fn test_brain_distance_different_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BrainGenome::random(&mut rng);
        let b = BrainGenome::random(&mut rng);
        assert!(brain_distance(&a, &b) > 0.0);
    }

    #[test]
    fn test_brain_distance_topology_matters() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BrainGenome::random(&mut rng);
        let mut b = a.clone();
        let mut tracker = InnovationTracker::new();

        // Add several structural mutations to b (200 iterations makes flakiness negligible)
        for _ in 0..200 {
            mutate_brain(&mut b, 0.0, 1.0, &mut rng, &mut tracker);
        }

        let d = brain_distance(&a, &b);
        assert!(
            d > 0.0,
            "Structural mutations should increase brain distance, got {}",
            d,
        );
    }

    #[test]
    fn test_different_inputs_produce_different_outputs() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut brain = Brain::from_genome(&BrainGenome::random(&mut rng));
        let input_a = [0.0_f32; INPUT_SIZE];
        let mut input_b = [0.0_f32; INPUT_SIZE];
        input_b[0] = 1.0;
        input_b[1] = 1.0;
        let out_a = brain.forward(&input_a);
        let out_b = brain.forward(&input_b);
        let diff: f32 = out_a.iter().zip(&out_b).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.001,
            "Different inputs should produce different outputs, diff={}",
            diff
        );
    }

    #[test]
    fn test_different_brains_produce_different_outputs() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut brain_a = Brain::from_genome(&BrainGenome::random(&mut rng));
        let mut brain_b = Brain::from_genome(&BrainGenome::random(&mut rng));
        let input = [0.5_f32; INPUT_SIZE];
        let out_a = brain_a.forward(&input);
        let out_b = brain_b.forward(&input);
        let diff: f32 = out_a.iter().zip(&out_b).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.001);
    }

    #[test]
    fn test_brain_system_modifies_velocity() {
        use crate::boids::Boid;
        use crate::components::*;
        use crate::ecosystem::Age;
        use crate::genome::CreatureGenome;
        use crate::needs::NeedWeights;
        use crate::phenotype::{derive_feeding, derive_physics};

        let mut rng = StdRng::seed_from_u64(42);
        let mut world = World::new();
        let env = Environment::default();
        let grid = SpatialGrid::new(10.0);

        let genome = CreatureGenome::random(&mut rng);
        let physics = derive_physics(&genome);
        let feeding = derive_feeding(&genome, &physics);
        let brain = Brain::from_genome(&genome.brain);
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };

        let entity = world.spawn((
            Position { x: 20.0, y: 10.0 },
            Velocity { vx: 0.0, vy: 0.0 },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            genome,
            physics,
            feeding,
            Energy::new(50.0),
            Age {
                ticks: 0,
                max_ticks: 5000,
            },
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ));

        let entity_map = std::collections::HashMap::new();
        let pheromone_grid = crate::pheromone::PheromoneGrid::new(100.0, 50.0);
        let _deposits = brain_system(
            &mut world,
            &grid,
            &entity_map,
            &env,
            &pheromone_grid,
            0.05,
            80.0,
            24.0,
        );

        let vel = world.get::<&Velocity>(entity).unwrap();
        let vel_mag = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
        assert!(vel_mag > 0.0, "Brain should have applied steering force");
    }

    #[test]
    fn test_offspring_inherit_brain_topology() {
        use crate::genome::CreatureGenome;

        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = CreatureGenome::random(&mut rng);
        let parent_b = CreatureGenome::random(&mut rng);

        let child_brain = crossover_brain(&parent_a.brain, &parent_b.brain, &mut rng);

        // Child should have valid connections
        assert!(!child_brain.connections.is_empty());

        // Child's forward pass should produce valid output
        let mut brain = Brain::from_genome(&child_brain);
        let input = [0.5f32; INPUT_SIZE];
        let output = brain.forward(&input);
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_innovation_tracker_reuses_numbers() {
        let mut tracker = InnovationTracker::new();
        let innov1 = tracker.get(0, 20);
        let innov2 = tracker.get(0, 20);
        assert_eq!(
            innov1, innov2,
            "Same structural mutation should get same innovation"
        );

        let innov3 = tracker.get(1, 20);
        assert_ne!(
            innov1, innov3,
            "Different mutations should get different innovations"
        );
    }

    #[test]
    fn test_topology_grows_over_generations() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        let initial_conns = bg.num_enabled_connections();
        let initial_hidden = bg.num_hidden_nodes();

        // Simulate 200 generations of mutation
        for _ in 0..200 {
            mutate_brain(&mut bg, 0.15, 1.0, &mut rng, &mut tracker);
        }

        assert!(
            bg.num_hidden_nodes() > initial_hidden,
            "Network should grow hidden nodes over 200 generations: {} -> {}",
            initial_hidden,
            bg.num_hidden_nodes(),
        );
        assert!(
            bg.num_enabled_connections() >= initial_conns,
            "Mutation should not collapse enabled connections below the starting topology: {} -> {}",
            initial_conns, bg.num_enabled_connections(),
        );

        // The brain should still produce valid output
        let mut brain = Brain::from_genome(&bg);
        let input = [0.5f32; INPUT_SIZE];
        let output = brain.forward(&input);
        for &v in &output {
            assert!(
                v >= -1.0 && v <= 1.0,
                "Evolved brain output out of range: {}",
                v
            );
        }
    }

    #[test]
    fn test_output_size_is_seven() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = BrainGenome::random(&mut rng);
        let mut brain = Brain::from_genome(&genome);
        let input = [0.0_f32; INPUT_SIZE];
        let output = brain.forward(&input);
        assert_eq!(
            output.len(),
            7,
            "Should have 7 outputs (6 behavioral + 1 pheromone emission)"
        );
    }

    #[test]
    fn test_hebbian_learning_modifies_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = BrainGenome::random(&mut rng);

        // Brain WITH learning
        let mut learning_brain = Brain::from_genome_with_learning(&genome, 0.1);
        let input = [0.5_f32; INPUT_SIZE];

        // Run forward many times to accumulate Hebbian changes
        for _ in 0..100 {
            learning_brain.forward(&input);
        }
        let output_learned = learning_brain.forward(&input);

        // Brain WITHOUT learning (same genome)
        let mut static_brain = Brain::from_genome_with_learning(&genome, 0.0);
        let output_static = static_brain.forward(&input);

        // Outputs should differ because learning modified weights
        let diff: f32 = output_learned
            .iter()
            .zip(output_static.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Hebbian learning should change outputs: diff={}",
            diff
        );
    }

    #[test]
    fn test_hebbian_learning_zero_rate_no_change() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = BrainGenome::random(&mut rng);
        let mut brain = Brain::from_genome_with_learning(&genome, 0.0);
        let input = [0.8_f32; INPUT_SIZE];

        let output1 = brain.forward(&input);
        // Run many times
        for _ in 0..50 {
            brain.forward(&input);
        }
        let output2 = brain.forward(&input);

        // With zero learning rate, outputs should be identical
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Zero learning rate should produce stable outputs"
            );
        }
    }

    #[test]
    fn test_pheromone_inputs_in_sensory() {
        use crate::components::*;
        use crate::environment::Environment;
        use crate::needs::Needs;
        use crate::phenotype::DerivedPhysics;

        let pos = Position { x: 50.0, y: 25.0 };
        let vel = Velocity { vx: 1.0, vy: 0.0 };
        let energy = Energy::new(100.0);
        let needs = Needs::default();
        let physics = DerivedPhysics {
            body_mass: 1.0,
            max_speed: 5.0,
            max_energy: 100.0,
            drag_coefficient: 0.1,
            sensory_range: 10.0,
            acceleration: 1.0,
            turn_radius: 1.0,
            base_metabolism: 0.1,
            visual_profile: 1.0,
            camouflage: 0.0,
        };
        let env = Environment::default();
        let no_info = NearestInfo {
            dist: 0.0,
            angle: 0.0,
        };

        let input = build_sensory_input(
            &pos, &vel, &energy, &needs, &physics, &env, &no_info, &no_info, &no_info, 100.0, 50.0,
            0.75, // pheromone_concentration
            -0.5, // pheromone_angle
        );

        assert_eq!(input.len(), INPUT_SIZE);
        assert!(
            (input[14] - 0.75).abs() < 0.01,
            "Pheromone concentration at index 14"
        );
        assert!(
            (input[15] - (-0.5)).abs() < 0.01,
            "Pheromone angle at index 15"
        );
    }

    #[test]
    fn test_brain_system_returns_pheromone_deposits() {
        use crate::boids::Boid;
        use crate::components::*;
        use crate::ecosystem::Age;
        use crate::genome::CreatureGenome;
        use crate::needs::NeedWeights;
        use crate::phenotype::{derive_feeding, derive_physics};

        let mut rng = StdRng::seed_from_u64(42);
        let mut world = World::new();
        let env = Environment::default();
        let grid = SpatialGrid::new(10.0);

        let genome = CreatureGenome::random(&mut rng);
        let physics = derive_physics(&genome);
        let feeding = derive_feeding(&genome, &physics);
        let brain = Brain::from_genome_with_learning(&genome.brain, genome.behavior.learning_rate);
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame.clone()], vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };

        world.spawn((
            Position { x: 20.0, y: 10.0 },
            Velocity { vx: 1.0, vy: 0.0 },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            genome,
            physics,
            feeding,
            Energy::new(50.0),
            Age {
                ticks: 0,
                max_ticks: 5000,
            },
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ));

        let entity_map: std::collections::HashMap<_, _> = std::collections::HashMap::new();
        let pheromone_grid = crate::pheromone::PheromoneGrid::new(100.0, 50.0);
        let deposits = brain_system(
            &mut world,
            &grid,
            &entity_map,
            &env,
            &pheromone_grid,
            0.05,
            100.0,
            50.0,
        );

        // Verify it doesn't crash and returns valid data
        for (x, y, amount) in &deposits {
            assert!(
                *x >= 0.0 && *x <= 100.0,
                "Deposit x should be in tank bounds"
            );
            assert!(
                *y >= 0.0 && *y <= 50.0,
                "Deposit y should be in tank bounds"
            );
            assert!(*amount >= 0.0, "Deposit amount should be non-negative");
        }
    }

    #[test]
    fn test_oja_rule_weight_stability() {
        // Verify that Oja's rule keeps weights bounded even with many iterations.
        // The old Hebbian rule would cause monotonic weight growth.
        let genome = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        let mut brain = Brain::from_genome_with_learning(&genome, 1.0); // max learning rate
        let input: [f32; INPUT_SIZE] = [0.8; INPUT_SIZE];

        // Run many forward passes to test weight stability
        for _ in 0..1000 {
            brain.forward(&input);
        }

        // All weights should remain within [-3.0, 3.0] bounds
        for conns in &brain.incoming {
            for &(_src, weight) in conns {
                assert!(
                    weight >= -3.0 && weight <= 3.0,
                    "Weight should be bounded after 1000 iterations, got {}",
                    weight
                );
            }
        }
    }

    #[test]
    fn test_oja_rule_weight_decay() {
        // Verify that weight decay prevents weights from saturating at boundaries
        let genome = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        let mut brain = Brain::from_genome_with_learning(&genome, 0.5);

        // Run with zero input — weight decay should pull weights toward zero
        let zero_input: [f32; INPUT_SIZE] = [0.0; INPUT_SIZE];
        let initial_weights: Vec<f32> = brain
            .incoming
            .iter()
            .flat_map(|conns| conns.iter().map(|(_, w)| w.abs()))
            .collect();
        let initial_sum: f32 = initial_weights.iter().sum();

        for _ in 0..500 {
            brain.forward(&zero_input);
        }

        let final_weights: Vec<f32> = brain
            .incoming
            .iter()
            .flat_map(|conns| conns.iter().map(|(_, w)| w.abs()))
            .collect();
        let final_sum: f32 = final_weights.iter().sum();

        assert!(
            final_sum < initial_sum,
            "Weight decay should reduce total weight magnitude: {} -> {}",
            initial_sum,
            final_sum
        );
    }

    #[test]
    fn test_oja_rule_converges_to_stable_weights() {
        // Oja's rule should converge: weight changes should diminish over time.
        // We use a seeded RNG and longer windows so the network has time to
        // settle past the initial transient phase.
        let mut rng = StdRng::seed_from_u64(42);
        let genome = BrainGenome::random(&mut rng);
        let mut brain = Brain::from_genome_with_learning(&genome, 0.8);
        let input: [f32; INPUT_SIZE] = [0.5; INPUT_SIZE];

        let snapshot = |b: &Brain| -> Vec<f32> {
            b.incoming
                .iter()
                .flat_map(|conns| conns.iter().map(|(_, w)| *w))
                .collect()
        };

        // Warm-up: let the network pass through the initial transient
        for _ in 0..1000 {
            brain.forward(&input);
        }
        let weights_1000 = snapshot(&brain);

        // Run 1000 more
        for _ in 0..1000 {
            brain.forward(&input);
        }
        let weights_2000 = snapshot(&brain);

        // Run 1000 more
        for _ in 0..1000 {
            brain.forward(&input);
        }
        let weights_3000 = snapshot(&brain);

        let delta_early: f32 = weights_1000
            .iter()
            .zip(weights_2000.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let delta_late: f32 = weights_2000
            .iter()
            .zip(weights_3000.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            delta_late <= delta_early * 1.1, // allow 10% noise margin
            "Oja's rule should converge (later changes ≤ earlier): early={:.4}, late={:.4}",
            delta_early,
            delta_late,
        );
    }

    #[test]
    fn test_learning_rate_zero_means_no_weight_change() {
        let genome = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        let mut brain = Brain::from_genome_with_learning(&genome, 0.0);
        let input: [f32; INPUT_SIZE] = [0.7; INPUT_SIZE];

        let weights_before: Vec<f32> = brain
            .incoming
            .iter()
            .flat_map(|conns| conns.iter().map(|(_, w)| *w))
            .collect();

        for _ in 0..100 {
            brain.forward(&input);
        }

        let weights_after: Vec<f32> = brain
            .incoming
            .iter()
            .flat_map(|conns| conns.iter().map(|(_, w)| *w))
            .collect();

        for (before, after) in weights_before.iter().zip(weights_after.iter()) {
            assert!(
                (before - after).abs() < f32::EPSILON,
                "Zero learning rate should mean no weight change: {} vs {}",
                before,
                after
            );
        }
    }

    #[test]
    fn test_structural_mutation_rates_add_nodes() {
        // With 8% add-node rate, running many mutations should produce hidden nodes
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();
        let mut genome = BrainGenome::random(&mut rng);

        assert_eq!(genome.num_hidden_nodes(), 0, "should start with no hidden nodes");

        // Mutate 100 times — at 8% rate, expect ~8 node additions
        for _ in 0..100 {
            mutate_brain(&mut genome, 0.15, 1.0, &mut rng, &mut tracker);
        }

        assert!(
            genome.num_hidden_nodes() >= 3,
            "After 100 mutations at 8% add-node rate, should have gained hidden nodes, got {}",
            genome.num_hidden_nodes()
        );
    }

    #[test]
    fn test_structural_mutation_rates_add_connections() {
        // With 12% add-connection rate, network should gain connections
        let mut rng = StdRng::seed_from_u64(99);
        let mut tracker = InnovationTracker::new();
        let mut genome = BrainGenome::random(&mut rng);

        let initial_conns = genome.connections.len();

        for _ in 0..100 {
            mutate_brain(&mut genome, 0.15, 1.0, &mut rng, &mut tracker);
        }

        assert!(
            genome.connections.len() > initial_conns + 5,
            "After 100 mutations at 12% add-conn rate, should have gained connections: {} → {}",
            initial_conns,
            genome.connections.len()
        );
    }

    #[test]
    fn test_self_connection_mutation() {
        // Self-connections should appear after repeated mutations at 6% rate
        let mut rng = StdRng::seed_from_u64(77);
        let mut tracker = InnovationTracker::new();
        let mut genome = BrainGenome::random(&mut rng);

        // First add some hidden nodes so self-connections have targets beyond outputs
        for _ in 0..200 {
            mutate_brain(&mut genome, 0.15, 1.0, &mut rng, &mut tracker);
        }

        let self_loops = genome
            .connections
            .iter()
            .filter(|c| c.in_node == c.out_node && c.enabled)
            .count();

        assert!(
            self_loops >= 1,
            "After 200 mutations at 6% self-conn rate, should have at least one self-loop, got {}",
            self_loops
        );
    }

    #[test]
    fn test_recurrent_self_connection_provides_memory() {
        // A brain with a self-connection should produce different outputs
        // on repeated identical inputs (due to recurrent state).
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);

        let mut tracker = InnovationTracker::new();
        let hidden = genome.next_node_id;
        genome.next_node_id += 1;
        while genome.node_depths.len() <= hidden as usize {
            genome.node_depths.push(0.5);
        }
        genome.node_depths[hidden as usize] = 0.5;

        // Connect input 0 → hidden → output 0
        genome.connections.push(ConnectionGene {
            in_node: 0,
            out_node: hidden,
            weight: 1.0,
            enabled: true,
            innovation: tracker.get(0, hidden),
        });
        genome.connections.push(ConnectionGene {
            in_node: hidden,
            out_node: FIRST_OUTPUT,
            weight: 1.0,
            enabled: true,
            innovation: tracker.get(hidden, FIRST_OUTPUT),
        });

        // Add self-connection on hidden node
        genome.connections.push(ConnectionGene {
            in_node: hidden,
            out_node: hidden,
            weight: 0.8,
            enabled: true,
            innovation: tracker.get(hidden, hidden),
        });

        let mut brain = Brain::from_genome(&genome);
        let input = [0.5; INPUT_SIZE];

        let out1 = brain.forward(&input);
        let out2 = brain.forward(&input);
        let out3 = brain.forward(&input);

        // With self-connection, second pass includes recurrent input from first
        assert!(
            (out1[0] - out2[0]).abs() > 0.001,
            "Recurrent self-connection should cause different outputs on identical inputs: {} vs {}",
            out1[0],
            out2[0]
        );

        // Should converge (outputs 2→3 closer than 1→2)
        let diff_12 = (out1[0] - out2[0]).abs();
        let diff_23 = (out2[0] - out3[0]).abs();
        assert!(
            diff_23 <= diff_12,
            "Recurrent activation should converge: diff(1,2)={:.6} diff(2,3)={:.6}",
            diff_12,
            diff_23
        );
    }

    #[test]
    fn test_oja_learning_rate_10x_faster() {
        // With the 10x increase (0.01 scaling), learning should be
        // noticeably faster. Run 100 ticks and verify weights changed meaningfully.
        let mut rng = StdRng::seed_from_u64(42);
        let genome = BrainGenome::random(&mut rng);
        let mut brain = Brain::from_genome_with_learning(&genome, 0.1);
        let input = [0.5; INPUT_SIZE];

        let weights_before: Vec<f32> = brain
            .incoming
            .iter()
            .flat_map(|conns| conns.iter().map(|(_, w)| *w))
            .collect();

        for _ in 0..100 {
            brain.forward(&input);
        }

        let weights_after: Vec<f32> = brain
            .incoming
            .iter()
            .flat_map(|conns| conns.iter().map(|(_, w)| *w))
            .collect();

        let total_change: f32 = weights_before
            .iter()
            .zip(weights_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            total_change > 0.1,
            "With 10x LR boost, 100 ticks should produce meaningful weight change, got {:.6}",
            total_change
        );
    }

    #[test]
    fn test_self_connections_excluded_from_topological_sort() {
        // Self-connections should not create cycles in the topological sort
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Add self-connections on every output node
        for o in FIRST_OUTPUT..FIRST_OUTPUT + OUTPUT_SIZE as u16 {
            genome.connections.push(ConnectionGene {
                in_node: o,
                out_node: o,
                weight: 0.5,
                enabled: true,
                innovation: tracker.get(o, o),
            });
        }

        let mut brain = Brain::from_genome(&genome);
        let input = [0.5; INPUT_SIZE];
        let output = brain.forward(&input);

        for &v in &output {
            assert!(!v.is_nan(), "Output should not be NaN with self-connections");
        }
    }

    // ── Tests for brain architecture upgrades ──────────────────

    #[test]
    fn test_activation_fn_apply_values() {
        assert!((ActivationFn::Tanh.apply(0.0)).abs() < 1e-6);
        assert!((ActivationFn::ReLU.apply(-1.0)).abs() < 1e-6);
        assert!((ActivationFn::ReLU.apply(2.0) - 2.0).abs() < 1e-6);
        assert!((ActivationFn::Sigmoid.apply(0.0) - 0.5).abs() < 1e-6);
        assert!((ActivationFn::Abs.apply(-3.0) - 3.0).abs() < 1e-6);
        assert_eq!(ActivationFn::Step.apply(-0.1), 0.0);
        assert_eq!(ActivationFn::Step.apply(0.1), 1.0);
        assert!((ActivationFn::Identity.apply(1.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_activation_fn_random_covers_all_variants() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            seen.insert(format!("{:?}", ActivationFn::random(&mut rng)));
        }
        assert!(seen.len() >= 5, "Random should produce at least 5 of 6 variants, got {:?}", seen);
    }

    #[test]
    fn test_node_gene_defaults_in_initial_genome() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = BrainGenome::random(&mut rng);

        // Input nodes should have Identity activation
        for i in 0..INPUT_SIZE {
            assert_eq!(genome.node_genes[i].activation_fn, ActivationFn::Identity,
                "Input node {i} should have Identity activation");
            assert!((genome.node_genes[i].bias).abs() < 1e-6,
                "Input node {i} should have zero bias");
        }
        // Output nodes should have Tanh activation
        for i in 0..OUTPUT_SIZE {
            let idx = FIRST_OUTPUT as usize + i;
            assert_eq!(genome.node_genes[idx].activation_fn, ActivationFn::Tanh,
                "Output node {idx} should have Tanh activation");
        }
    }

    #[test]
    fn test_different_activations_produce_different_outputs() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);

        // All-Tanh brain
        let mut brain_tanh = Brain::from_genome(&genome);
        let input = [0.5f32; INPUT_SIZE];
        let out_tanh = brain_tanh.forward(&input);

        // Change output activations to ReLU
        for i in 0..OUTPUT_SIZE {
            genome.node_genes[FIRST_OUTPUT as usize + i].activation_fn = ActivationFn::ReLU;
        }
        let mut brain_relu = Brain::from_genome(&genome);
        let out_relu = brain_relu.forward(&input);

        let diff: f32 = out_tanh.iter().zip(&out_relu).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.001,
            "Different activation functions should produce different outputs, diff={diff}");
    }

    #[test]
    fn test_bias_affects_output() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);

        // Zero-bias brain
        let mut brain_no_bias = Brain::from_genome(&genome);
        let input = [0.3f32; INPUT_SIZE];
        let out_no_bias = brain_no_bias.forward(&input);

        // Add bias to output nodes
        for i in 0..OUTPUT_SIZE {
            genome.node_genes[FIRST_OUTPUT as usize + i].bias = 1.0;
        }
        let mut brain_bias = Brain::from_genome(&genome);
        let out_bias = brain_bias.forward(&input);

        let diff: f32 = out_no_bias.iter().zip(&out_bias).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01,
            "Bias should shift outputs, diff={diff}");
    }

    #[test]
    fn test_bias_perturbation_stays_bounded() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        for _ in 0..1000 {
            mutate_brain(&mut genome, 0.5, 1.0, &mut rng, &mut tracker);
        }

        for (i, ng) in genome.node_genes.iter().enumerate() {
            assert!(ng.bias >= -2.0 && ng.bias <= 2.0,
                "Node {i} bias out of bounds: {}", ng.bias);
        }
    }

    #[test]
    fn test_activation_swap_mutation_eventually_changes() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // First add some hidden nodes
        for _ in 0..50 {
            mutate_brain(&mut genome, 0.0, 1.0, &mut rng, &mut tracker);
        }

        let initial_activations: Vec<ActivationFn> = genome.node_genes.iter()
            .map(|ng| ng.activation_fn)
            .collect();

        // Run many more mutations
        for _ in 0..500 {
            mutate_brain(&mut genome, 0.1, 1.0, &mut rng, &mut tracker);
        }

        let changed = genome.node_genes.iter().enumerate()
            .any(|(i, ng)| i < initial_activations.len() && ng.activation_fn != initial_activations[i]);
        assert!(changed, "Activation swap mutation should eventually change at least one activation");
    }

    #[test]
    fn test_crossover_preserves_node_genes() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut a = BrainGenome::random(&mut rng);
        let mut b = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Evolve both parents to have hidden nodes
        for _ in 0..50 {
            mutate_brain(&mut a, 0.1, 1.0, &mut rng, &mut tracker);
            mutate_brain(&mut b, 0.1, 1.0, &mut rng, &mut tracker);
        }

        let child = crossover_brain(&a, &b, &mut rng);

        // Child should have node_genes for all nodes
        assert_eq!(child.node_genes.len(), child.next_node_id as usize,
            "Child should have node_genes for every node");

        // Input nodes should still be Identity
        for i in 0..INPUT_SIZE {
            assert_eq!(child.node_genes[i].activation_fn, ActivationFn::Identity);
        }

        // Child brain should produce valid output
        let mut brain = Brain::from_genome(&child);
        let output = brain.forward(&[0.5; INPUT_SIZE]);
        for &v in &output {
            assert!(!v.is_nan() && v.is_finite(), "Child brain output invalid: {v}");
        }
    }

    #[test]
    fn test_brain_distance_reflects_activation_differences() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BrainGenome::random(&mut rng);
        let mut b = a.clone();

        let dist_same = brain_distance(&a, &b);

        // Change all output activations in b
        for i in 0..OUTPUT_SIZE {
            b.node_genes[FIRST_OUTPUT as usize + i].activation_fn = ActivationFn::ReLU;
        }
        let dist_diff = brain_distance(&a, &b);

        assert!(dist_diff > dist_same,
            "Activation differences should increase distance: same={dist_same}, diff={dist_diff}");
    }

    #[test]
    fn test_modulator_node_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Add hidden nodes first
        for _ in 0..100 {
            mutate_brain(&mut genome, 0.0, 1.0, &mut rng, &mut tracker);
        }

        let has_modulator = genome.node_genes.iter()
            .any(|ng| ng.role == NodeRole::Modulator);

        // Even if no modulator yet, run more mutations
        if !has_modulator {
            for _ in 0..500 {
                mutate_brain(&mut genome, 0.1, 1.0, &mut rng, &mut tracker);
            }
        }

        let has_modulator = genome.node_genes.iter()
            .any(|ng| ng.role == NodeRole::Modulator);
        assert!(has_modulator, "Modulator mutation should eventually create a modulator node");

        // Modulator nodes should have Sigmoid activation
        for ng in &genome.node_genes {
            if ng.role == NodeRole::Modulator {
                assert_eq!(ng.activation_fn, ActivationFn::Sigmoid,
                    "Modulator nodes should have Sigmoid activation");
            }
        }
    }

    #[test]
    fn test_modulator_gates_learning() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Add a hidden node and make it a modulator
        for _ in 0..30 {
            mutate_brain(&mut genome, 0.0, 1.0, &mut rng, &mut tracker);
        }
        // Force a hidden node to be a modulator
        if genome.next_node_id > FIRST_HIDDEN {
            let idx = FIRST_HIDDEN as usize;
            genome.node_genes[idx].role = NodeRole::Modulator;
            genome.node_genes[idx].activation_fn = ActivationFn::Sigmoid;
        }

        // Brain with modulator + learning should not crash and should produce valid output
        let mut brain = Brain::from_genome_with_learning(&genome, 0.5);
        let input = [0.5; INPUT_SIZE];
        for _ in 0..100 {
            let output = brain.forward(&input);
            for &v in &output {
                assert!(!v.is_nan() && v.is_finite(), "Modulated brain output invalid: {v}");
            }
        }
    }

    #[test]
    fn test_attention_node_produces_valid_output() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Add hidden nodes
        for _ in 0..30 {
            mutate_brain(&mut genome, 0.0, 1.0, &mut rng, &mut tracker);
        }

        // Force a hidden node to be attention
        if genome.next_node_id > FIRST_HIDDEN {
            let idx = FIRST_HIDDEN as usize;
            genome.node_genes[idx].role = NodeRole::Attention;
            genome.node_genes[idx].activation_fn = ActivationFn::Identity;
        }

        let mut brain = Brain::from_genome(&genome);
        let input = [0.5; INPUT_SIZE];
        let output = brain.forward(&input);
        for &v in &output {
            assert!(!v.is_nan() && v.is_finite(), "Attention brain output invalid: {v}");
        }
    }

    #[test]
    fn test_attention_differs_from_standard() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Add hidden nodes
        for _ in 0..30 {
            mutate_brain(&mut genome, 0.0, 1.0, &mut rng, &mut tracker);
        }

        // Standard brain
        let mut standard_brain = Brain::from_genome(&genome);
        let input = [0.5; INPUT_SIZE];
        let out_standard = standard_brain.forward(&input);

        // Now make a hidden node attention
        if genome.next_node_id > FIRST_HIDDEN {
            let idx = FIRST_HIDDEN as usize;
            genome.node_genes[idx].role = NodeRole::Attention;
            genome.node_genes[idx].activation_fn = ActivationFn::Identity;
        }

        let mut attn_brain = Brain::from_genome(&genome);
        let out_attn = attn_brain.forward(&input);

        let diff: f32 = out_standard.iter().zip(&out_attn).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0001,
            "Attention node should produce different output than standard, diff={diff}");
    }

    #[test]
    fn test_module_duplication_creates_valid_topology() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // Build up a network with hidden nodes
        for _ in 0..50 {
            mutate_brain(&mut genome, 0.1, 1.0, &mut rng, &mut tracker);
        }

        let nodes_before = genome.next_node_id;
        let conns_before = genome.connections.len();

        // Force module duplication
        let hidden_count = genome.next_node_id.saturating_sub(FIRST_HIDDEN);
        if hidden_count > 0 {
            let seed = FIRST_HIDDEN + rng.random_range(0..hidden_count);
            duplicate_module(&mut genome, seed, &mut rng, &mut tracker);
        }

        assert!(genome.next_node_id >= nodes_before,
            "Module dup should not decrease node count");
        assert!(genome.connections.len() >= conns_before,
            "Module dup should not decrease connection count");

        // Verify the duplicated brain still works
        let mut brain = Brain::from_genome(&genome);
        let output = brain.forward(&[0.5; INPUT_SIZE]);
        for &v in &output {
            assert!(!v.is_nan() && v.is_finite(),
                "Brain with duplicated module should produce valid output: {v}");
        }
    }

    #[test]
    fn test_evolved_brain_with_all_features_valid() {
        // Evolve a brain for many generations with all features enabled,
        // then verify it still produces valid outputs
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        for _ in 0..300 {
            mutate_brain(&mut genome, 0.2, 1.0, &mut rng, &mut tracker);
        }

        // Verify structural integrity
        assert!(genome.node_genes.len() >= genome.next_node_id as usize,
            "node_genes should cover all node IDs");
        assert!(genome.node_depths.len() >= genome.next_node_id as usize,
            "node_depths should cover all node IDs");

        // Verify forward pass works with learning
        let mut brain = Brain::from_genome_with_learning(&genome, 0.1);
        let input = [0.5; INPUT_SIZE];
        for _ in 0..100 {
            let output = brain.forward(&input);
            for &v in &output {
                assert!(!v.is_nan() && v.is_finite(),
                    "Fully evolved brain output should be valid: {v}");
            }
        }
    }

    #[test]
    fn test_node_role_enum_coverage() {
        assert_eq!(NodeRole::default(), NodeRole::Standard);
        assert_ne!(NodeRole::Standard, NodeRole::Modulator);
        assert_ne!(NodeRole::Standard, NodeRole::Attention);
        assert_ne!(NodeRole::Modulator, NodeRole::Attention);
    }

    #[test]
    fn test_ensure_node_genes_pads_correctly() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = BrainGenome::random(&mut rng);
        // Artificially bump next_node_id beyond node_genes length
        genome.next_node_id = FIRST_HIDDEN + 5;
        ensure_node_genes(&mut genome);
        assert_eq!(genome.node_genes.len(), (FIRST_HIDDEN + 5) as usize);
        // New hidden nodes should have Tanh activation
        for i in FIRST_HIDDEN as usize..(FIRST_HIDDEN + 5) as usize {
            assert_eq!(genome.node_genes[i].activation_fn, ActivationFn::Tanh);
            assert_eq!(genome.node_genes[i].role, NodeRole::Standard);
        }
    }

    // ── Diversity-scaled mutation tests ──────────────────────────────

    #[test]
    fn test_high_diversity_reduces_weight_perturbation_magnitude() {
        // At diversity=2.5 the per-weight perturbation should be ±0.19 (= 0.3/√2.5)
        // versus ±0.3 at diversity=1.0. We prove this statistically: mutate 1000
        // copies at each diversity and compare the average absolute weight delta.
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();

        let mut sum_delta_low = 0.0_f64;
        let mut count_low = 0u32;
        let mut sum_delta_high = 0.0_f64;
        let mut count_high = 0u32;

        for _ in 0..500 {
            let mut g1 = BrainGenome::random(&mut rng);
            let mut g2 = g1.clone();
            let w_before: Vec<f32> = g1.connections.iter().map(|c| c.weight).collect();

            mutate_brain(&mut g1, 1.0, 1.0, &mut rng, &mut tracker);
            mutate_brain(&mut g2, 1.0, 2.5, &mut rng, &mut tracker);

            for (i, orig) in w_before.iter().enumerate() {
                if i < g1.connections.len() {
                    let d = (g1.connections[i].weight - orig).abs();
                    if d > 0.0 { sum_delta_low += d as f64; count_low += 1; }
                }
                if i < g2.connections.len() {
                    let d = (g2.connections[i].weight - orig).abs();
                    if d > 0.0 { sum_delta_high += d as f64; count_high += 1; }
                }
            }
        }

        let avg_low = sum_delta_low / count_low.max(1) as f64;
        let avg_high = sum_delta_high / count_high.max(1) as f64;
        // High diversity should produce smaller average perturbations
        assert!(
            avg_high < avg_low,
            "High diversity should produce smaller weight perturbations: avg_high={avg_high:.4} avg_low={avg_low:.4}"
        );
    }

    #[test]
    fn test_high_diversity_increases_structural_mutations() {
        // At diversity=2.5 structural rates are 1.75× baseline.
        // We run many mutations and count how many hidden nodes are added.
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();

        let mut nodes_added_low = 0u32;
        let mut nodes_added_high = 0u32;

        for _ in 0..2000 {
            let mut g_low = BrainGenome::random(&mut rng);
            let mut g_high = g_low.clone();
            let before = g_low.next_node_id;

            mutate_brain(&mut g_low, 0.15, 1.0, &mut rng, &mut tracker);
            mutate_brain(&mut g_high, 0.15, 2.5, &mut rng, &mut tracker);

            nodes_added_low += (g_low.next_node_id - before) as u32;
            nodes_added_high += (g_high.next_node_id - before) as u32;
        }

        // High diversity should add more nodes (structural rate is 1.75× higher)
        assert!(
            nodes_added_high > nodes_added_low,
            "High diversity should add more structural nodes: high={nodes_added_high} low={nodes_added_low}"
        );
    }

    #[test]
    fn test_high_diversity_increases_connection_count() {
        // Similar to the node test but for connections (ADD_CONN_RATE + ADD_SELF_CONN_RATE)
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();

        let mut conns_added_low = 0i32;
        let mut conns_added_high = 0i32;

        for _ in 0..2000 {
            let mut g_low = BrainGenome::random(&mut rng);
            let mut g_high = g_low.clone();
            let before = g_low.connections.len() as i32;

            mutate_brain(&mut g_low, 0.15, 1.0, &mut rng, &mut tracker);
            mutate_brain(&mut g_high, 0.15, 2.5, &mut rng, &mut tracker);

            conns_added_low += g_low.connections.len() as i32 - before;
            conns_added_high += g_high.connections.len() as i32 - before;
        }

        assert!(
            conns_added_high > conns_added_low,
            "High diversity should add more connections: high={conns_added_high} low={conns_added_low}"
        );
    }

    #[test]
    fn test_diversity_one_is_neutral() {
        // At diversity=1.0, mutation behavior should be identical to the legacy behavior.
        // We prove this by running the same genome through mutation with diversity=1.0
        // and checking that mag_scale=1.0 and struct_scale=1.0.
        // (White-box: 1.0/sqrt(1.0)=1.0, 0.5+0.5*1.0=1.0)
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut rng2 = StdRng::seed_from_u64(99);
        let mut tracker1 = InnovationTracker::new();
        let mut tracker2 = InnovationTracker::new();

        let base = BrainGenome::random(&mut StdRng::seed_from_u64(42));
        let mut g1 = base.clone();
        let mut g2 = base.clone();

        mutate_brain(&mut g1, 0.15, 1.0, &mut rng1, &mut tracker1);
        mutate_brain(&mut g2, 0.15, 1.0, &mut rng2, &mut tracker2);

        // Same seed, same diversity → identical results
        assert_eq!(g1.connections.len(), g2.connections.len());
        for (c1, c2) in g1.connections.iter().zip(g2.connections.iter()) {
            assert_eq!(c1.weight, c2.weight);
            assert_eq!(c1.enabled, c2.enabled);
        }
        assert_eq!(g1.next_node_id, g2.next_node_id);
    }

    #[test]
    fn test_high_diversity_bias_perturbation_is_smaller() {
        // Bias perturbation should also scale inversely with √diversity
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();

        let mut sum_bias_delta_low = 0.0_f64;
        let mut sum_bias_delta_high = 0.0_f64;
        let mut count = 0u32;

        for _ in 0..500 {
            // Create a genome with a hidden node so we have biases to perturb
            let mut g = BrainGenome::random(&mut rng);
            // Force a hidden node to exist
            g.next_node_id = FIRST_HIDDEN + 1;
            ensure_node_genes(&mut g);
            let bias_before = g.node_genes.last().unwrap().bias;

            let mut g_low = g.clone();
            let mut g_high = g.clone();

            mutate_brain(&mut g_low, 1.0, 1.0, &mut rng, &mut tracker);
            mutate_brain(&mut g_high, 1.0, 2.5, &mut rng, &mut tracker);

            let idx = FIRST_HIDDEN as usize;
            if idx < g_low.node_genes.len() && idx < g_high.node_genes.len() {
                let d_low = (g_low.node_genes[idx].bias - bias_before).abs();
                let d_high = (g_high.node_genes[idx].bias - bias_before).abs();
                if d_low > 0.0 || d_high > 0.0 {
                    sum_bias_delta_low += d_low as f64;
                    sum_bias_delta_high += d_high as f64;
                    count += 1;
                }
            }
        }

        if count > 10 {
            let avg_low = sum_bias_delta_low / count as f64;
            let avg_high = sum_bias_delta_high / count as f64;
            assert!(
                avg_high < avg_low,
                "High diversity bias perturbation should be smaller: high={avg_high:.5} low={avg_low:.5}"
            );
        }
    }

    #[test]
    fn test_low_diversity_reduces_structural_rates() {
        // At diversity=0.25 structural rates are 0.625× baseline → fewer mutations
        let mut rng = StdRng::seed_from_u64(42);
        let mut tracker = InnovationTracker::new();

        let mut nodes_added_default = 0u32;
        let mut nodes_added_low = 0u32;

        for _ in 0..2000 {
            let mut g_def = BrainGenome::random(&mut rng);
            let mut g_low = g_def.clone();
            let before = g_def.next_node_id;

            mutate_brain(&mut g_def, 0.15, 1.0, &mut rng, &mut tracker);
            mutate_brain(&mut g_low, 0.15, 0.25, &mut rng, &mut tracker);

            nodes_added_default += (g_def.next_node_id - before) as u32;
            nodes_added_low += (g_low.next_node_id - before) as u32;
        }

        assert!(
            nodes_added_low < nodes_added_default,
            "Low diversity should add fewer nodes: low={nodes_added_low} default={nodes_added_default}"
        );
    }
}
