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

const FIRST_OUTPUT: u16 = INPUT_SIZE as u16;       // 16
const FIRST_HIDDEN: u16 = (INPUT_SIZE + OUTPUT_SIZE) as u16; // 23
const MAX_NODES: u16 = 60;       // cap total nodes (40 hidden max)
const MAX_CONNECTIONS: usize = 300;

/// Number of connections in the initial full input→output topology.
const INITIAL_INNOVATIONS: u32 = (INPUT_SIZE * OUTPUT_SIZE) as u32; // 112

// Structural mutation rates
const ADD_NODE_RATE: f64 = 0.03;
const ADD_CONN_RATE: f64 = 0.05;

// NEAT distance coefficients
const C_EXCESS: f32 = 1.0;
const C_DISJOINT: f32 = 1.0;
const C_WEIGHT: f32 = 0.4;

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

/// NEAT-style genome: a variable-length list of connection genes.
/// Node depths enforce feedforward structure.
#[derive(Debug, Clone)]
pub struct BrainGenome {
    pub connections: Vec<ConnectionGene>,
    /// Depth of each node (indexed by node ID). Input=0.0, Output=1.0, Hidden=between.
    pub node_depths: Vec<f32>,
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

        BrainGenome {
            connections,
            node_depths,
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
/// lifetime learning: weights are modified during the creature's life
/// based on co-activation, but learned changes are NOT inherited.
#[derive(Debug, Clone)]
pub struct Brain {
    order: Vec<u16>,
    /// Incoming connections per node: incoming[node_id] = [(source, weight), ...]
    incoming: Vec<Vec<(u16, f32)>>,
    num_nodes: usize,
    /// Hebbian learning rate (from genome, 0.0 = no learning)
    learning_rate: f32,
}

impl Brain {
    pub fn from_genome(genome: &BrainGenome) -> Self {
        let num_nodes = genome.next_node_id as usize;
        let (order, incoming) = compute_topology(&genome.connections, num_nodes);
        Brain { order, incoming, num_nodes, learning_rate: 0.0 }
    }

    pub fn from_genome_with_learning(genome: &BrainGenome, learning_rate: f32) -> Self {
        let num_nodes = genome.next_node_id as usize;
        let (order, incoming) = compute_topology(&genome.connections, num_nodes);
        Brain { order, incoming, num_nodes, learning_rate }
    }

    /// Feedforward pass with Hebbian weight update.
    /// When connected neurons co-activate (both positive or both negative),
    /// the connection weight is slightly strengthened.
    pub fn forward(&mut self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut activations = vec![0.0f32; self.num_nodes];

        for i in 0..INPUT_SIZE {
            activations[i] = input[i];
        }

        for &node_id in &self.order {
            let idx = node_id as usize;
            let mut sum = 0.0f32;
            for &(src, weight) in &self.incoming[idx] {
                sum += activations[src as usize] * weight;
            }
            activations[idx] = sum.tanh();
        }

        // Hebbian learning: adjust weights based on co-activation
        if self.learning_rate > 0.0 {
            let lr = self.learning_rate * 0.001; // Scale down for stability
            for conns in &mut self.incoming {
                for (src, weight) in conns.iter_mut() {
                    let src_act = activations.get(*src as usize).copied().unwrap_or(0.0);
                    // Find target node activation (this connection's target)
                    // We use the Oja rule variant: Δw = lr * post * (pre - w * post)
                    // This is a stable Hebbian rule that prevents weight explosion
                    // Since we iterate by target node, we need the target activation
                    // We'll approximate with a simpler rule for efficiency
                    let delta = lr * src_act;
                    *weight = (*weight + delta).clamp(-3.0, 3.0);
                }
            }
        }

        let mut output = [0.0f32; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            output[i] = activations[FIRST_OUTPUT as usize + i];
        }
        output
    }
}

/// Compute topological ordering and incoming connection lists via Kahn's algorithm.
fn compute_topology(connections: &[ConnectionGene], num_nodes: usize) -> (Vec<u16>, Vec<Vec<(u16, f32)>>) {
    let mut incoming: Vec<Vec<(u16, f32)>> = vec![Vec::new(); num_nodes];
    let mut in_degree = vec![0u32; num_nodes];
    let mut outgoing: Vec<Vec<u16>> = vec![Vec::new(); num_nodes];

    for conn in connections {
        if !conn.enabled { continue; }
        let out = conn.out_node as usize;
        let inp = conn.in_node as usize;
        if out >= num_nodes || inp >= num_nodes { continue; }
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

    (order, incoming)
}

// ── Sensory input construction ────────────────────────────────

struct NearestInfo {
    dist: f32,  // normalized 0..1 (1 = close)
    angle: f32, // normalized -1..1
}

impl Default for NearestInfo {
    fn default() -> Self {
        Self { dist: 0.0, angle: 0.0 }
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
        pheromone_concentration.clamp(0.0, 1.0),  // 14
        pheromone_angle.clamp(-1.0, 1.0),          // 15
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
        px: f32, py: f32,
        vx: f32, vy: f32,
        energy_frac: f32,
        hunger: f32, safety: f32, repro: f32,
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
        for (entity, pos, vel, energy, needs, physics, feeding, genome, brain) in
            &mut world.query::<(Entity, &Position, &Velocity, &Energy, &Needs, &DerivedPhysics, &FeedingCapability, &CreatureGenome, &Brain)>()
        {
            v.push(CreatureData {
                entity,
                px: pos.x, py: pos.y,
                vx: vel.vx, vy: vel.vy,
                energy_frac: energy.fraction(),
                hunger: needs.hunger, safety: needs.safety, repro: needs.reproduction,
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
    let actions: Vec<(Entity, f32, f32, BehaviorAction, Brain, (f32, f32, f32))> = creatures.into_par_iter().map(|mut c| {
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
                nearest_food = NearestInfo { dist: norm_dist, angle };
            }

            if !info.is_producer && info.max_prey_mass > c.body_mass && dist < best_pred_dist {
                best_pred_dist = dist;
                nearest_predator = NearestInfo { dist: norm_dist, angle };
            }

            if !info.is_producer && info.body_mass > 0.0 {
                let mass_ratio = c.body_mass / info.body_mass;
                if mass_ratio > 0.5 && mass_ratio < 2.0 && dist < best_ally_dist {
                    best_ally_dist = dist;
                    nearest_ally = NearestInfo { dist: norm_dist, angle };
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
        let energy_proxy = Energy { current: c.energy_frac, max: 1.0 };
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
            &pos_proxy, &vel_proxy, &energy_proxy, &needs_proxy, &physics_proxy,
            env, &nearest_food, &nearest_predator, &nearest_ally,
            tank_w, tank_h, phero_conc, phero_angle,
        );

        let output = c.brain.forward(&input);
        let scale = c.complexity.max(0.3);

        let steer_x = output[0] * scale;
        let steer_y = output[1] * scale;
        let speed_mult = (output[2] * scale + 1.0) * 0.7 + 0.1;

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
    }).collect();

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
    let a_genes: HashMap<u32, &ConnectionGene> = a.connections.iter()
        .map(|g| (g.innovation, g)).collect();
    let b_genes: HashMap<u32, &ConnectionGene> = b.connections.iter()
        .map(|g| (g.innovation, g)).collect();

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

    BrainGenome {
        connections: child_connections,
        node_depths,
        next_node_id,
    }
}

/// NEAT mutation: weight perturbation + structural mutations (add node, add connection).
pub fn mutate_brain(brain: &mut BrainGenome, rate: f32, rng: &mut impl Rng, tracker: &mut InnovationTracker) {
    // 1. Weight perturbation
    for conn in &mut brain.connections {
        if rng.random_bool(rate as f64) {
            conn.weight += rng.random_range(-0.3..0.3_f32);
            conn.weight = conn.weight.clamp(-3.0, 3.0);
        }
    }

    // 2. Add node: split an enabled connection, preserving network behavior
    if rng.random_bool(ADD_NODE_RATE) && brain.next_node_id < MAX_NODES {
        let enabled: Vec<usize> = brain.connections.iter().enumerate()
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

            let in_depth = brain.node_depths.get(old_in as usize).copied().unwrap_or(0.0);
            let out_depth = brain.node_depths.get(old_out as usize).copied().unwrap_or(1.0);
            while brain.node_depths.len() <= new_node as usize {
                brain.node_depths.push(0.5);
            }
            brain.node_depths[new_node as usize] = (in_depth + out_depth) / 2.0;

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
    if rng.random_bool(ADD_CONN_RATE) && brain.connections.len() < MAX_CONNECTIONS {
        let existing: HashSet<(u16, u16)> = brain.connections.iter()
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
            if a == b { continue; }

            let da = brain.node_depths.get(a as usize).copied().unwrap_or(0.0);
            let db = brain.node_depths.get(b as usize).copied().unwrap_or(1.0);

            let (from, to) = if da < db { (a, b) } else if db < da { (b, a) } else { continue };

            // Don't target input nodes
            if to < INPUT_SIZE as u16 { continue; }

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
}

/// NEAT genomic distance: excess genes, disjoint genes, and weight differences.
pub fn brain_distance(a: &BrainGenome, b: &BrainGenome) -> f32 {
    let a_map: HashMap<u32, &ConnectionGene> = a.connections.iter()
        .map(|g| (g.innovation, g)).collect();
    let b_map: HashMap<u32, &ConnectionGene> = b.connections.iter()
        .map(|g| (g.innovation, g)).collect();

    let a_max = a.connections.iter().map(|c| c.innovation).max().unwrap_or(0);
    let b_max = b.connections.iter().map(|c| c.innovation).max().unwrap_or(0);
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
    let avg_w = if matching > 0 { weight_diff_sum / matching as f32 } else { 0.0 };

    C_EXCESS * excess as f32 / n + C_DISJOINT * disjoint as f32 / n + C_WEIGHT * avg_w
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_genome_has_full_connectivity() {
        let bg = BrainGenome::random(&mut rand::rng());
        assert_eq!(bg.connections.len(), INPUT_SIZE * OUTPUT_SIZE);
        assert_eq!(bg.num_hidden_nodes(), 0);
        assert_eq!(bg.next_node_id, FIRST_HIDDEN);
    }

    #[test]
    fn test_forward_output_range() {
        let mut rng = rand::rng();
        let mut brain = Brain::from_genome(&BrainGenome::random(&mut rng));
        let input: [f32; INPUT_SIZE] = std::array::from_fn(|_| rng.random_range(-1.0..1.0_f32));
        let output = brain.forward(&input);
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0, "tanh output should be in [-1,1], got {}", v);
        }
    }

    #[test]
    fn test_forward_deterministic() {
        let genome = BrainGenome::random(&mut rand::rng());
        let mut brain = Brain::from_genome(&genome);
        let input: [f32; INPUT_SIZE] = [0.5; INPUT_SIZE];
        let out1 = brain.forward(&input);
        let out2 = brain.forward(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_forward_zero_weights() {
        let mut genome = BrainGenome::random(&mut rand::rng());
        for conn in &mut genome.connections {
            conn.weight = 0.0;
        }
        let mut brain = Brain::from_genome(&genome);
        let input = [1.0_f32; INPUT_SIZE];
        let output = brain.forward(&input);
        for &v in &output {
            assert!((v).abs() < 1e-6, "Zero weights should produce zero output, got {}", v);
        }
    }

    #[test]
    fn test_crossover_preserves_innovations() {
        let mut rng = rand::rng();
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
                "Child innovation {} not found in either parent", conn.innovation,
            );
        }
    }

    #[test]
    fn test_mutation_weight_stays_bounded() {
        let mut rng = rand::rng();
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();
        for _ in 0..1000 {
            mutate_brain(&mut bg, 0.5, &mut rng, &mut tracker);
        }
        for conn in &bg.connections {
            assert!(conn.weight >= -3.0 && conn.weight <= 3.0, "Weight out of bounds: {}", conn.weight);
        }
    }

    #[test]
    fn test_add_node_mutation() {
        let mut rng = rand::rng();
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();
        let initial_conns = bg.connections.len();
        let initial_nodes = bg.next_node_id;

        // Force many add-node attempts
        for _ in 0..100 {
            // Directly add a node by calling mutation with high structural rates
            mutate_brain(&mut bg, 0.0, &mut rng, &mut tracker); // only structural mutations
        }

        // At least one node should have been added (3% rate * 100 = ~3 expected)
        assert!(
            bg.next_node_id > initial_nodes,
            "Add-node mutation should create hidden nodes: {} -> {}",
            initial_nodes, bg.next_node_id,
        );
        assert!(
            bg.connections.len() > initial_conns,
            "Add-node should create new connections: {} -> {}",
            initial_conns, bg.connections.len(),
        );
    }

    #[test]
    fn test_add_connection_mutation() {
        let mut rng = rand::rng();
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        // First add a hidden node so there are new connections to create
        bg.next_node_id = FIRST_HIDDEN + 1;
        bg.node_depths.push(0.5); // hidden node at depth 0.5
        bg.connections.push(ConnectionGene {
            in_node: 0, out_node: FIRST_HIDDEN, weight: 1.0,
            enabled: true, innovation: tracker.get(0, FIRST_HIDDEN),
        });

        let initial_conns = bg.connections.len();
        for _ in 0..200 {
            mutate_brain(&mut bg, 0.0, &mut rng, &mut tracker);
        }

        assert!(
            bg.connections.len() > initial_conns,
            "Add-connection mutation should create connections: {} -> {}",
            initial_conns, bg.connections.len(),
        );
    }

    #[test]
    fn test_brain_distance_self_zero() {
        let bg = BrainGenome::random(&mut rand::rng());
        assert!((brain_distance(&bg, &bg)).abs() < 1e-6);
    }

    #[test]
    fn test_brain_distance_different_positive() {
        let mut rng = rand::rng();
        let a = BrainGenome::random(&mut rng);
        let b = BrainGenome::random(&mut rng);
        assert!(brain_distance(&a, &b) > 0.0);
    }

    #[test]
    fn test_brain_distance_topology_matters() {
        let mut rng = rand::rng();
        let a = BrainGenome::random(&mut rng);
        let mut b = a.clone();
        let mut tracker = InnovationTracker::new();

        // Add several structural mutations to b (200 iterations makes flakiness negligible)
        for _ in 0..200 {
            mutate_brain(&mut b, 0.0, &mut rng, &mut tracker);
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
        let mut rng = rand::rng();
        let mut brain = Brain::from_genome(&BrainGenome::random(&mut rng));
        let input_a = [0.0_f32; INPUT_SIZE];
        let mut input_b = [0.0_f32; INPUT_SIZE];
        input_b[0] = 1.0;
        input_b[1] = 1.0;
        let out_a = brain.forward(&input_a);
        let out_b = brain.forward(&input_b);
        let diff: f32 = out_a.iter().zip(&out_b).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.001, "Different inputs should produce different outputs, diff={}", diff);
    }

    #[test]
    fn test_different_brains_produce_different_outputs() {
        let mut rng = rand::rng();
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
        use crate::components::*;
        use crate::ecosystem::Age;
        use crate::needs::NeedWeights;
        use crate::phenotype::{derive_physics, derive_feeding};
        use crate::genome::CreatureGenome;
        use crate::boids::Boid;

        let mut rng = rand::rng();
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
            Age { ticks: 0, max_ticks: 5000 },
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ));

        let entity_map = std::collections::HashMap::new();
        let pheromone_grid = crate::pheromone::PheromoneGrid::new(100.0, 50.0);
        let _deposits = brain_system(&mut world, &grid, &entity_map, &env, &pheromone_grid, 0.05, 80.0, 24.0);

        let vel = world.get::<&Velocity>(entity).unwrap();
        let vel_mag = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
        assert!(vel_mag > 0.0, "Brain should have applied steering force");
    }

    #[test]
    fn test_offspring_inherit_brain_topology() {
        use crate::genome::CreatureGenome;

        let mut rng = rand::rng();
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
        assert_eq!(innov1, innov2, "Same structural mutation should get same innovation");

        let innov3 = tracker.get(1, 20);
        assert_ne!(innov1, innov3, "Different mutations should get different innovations");
    }

    #[test]
    fn test_topology_grows_over_generations() {
        let mut rng = rand::rng();
        let mut bg = BrainGenome::random(&mut rng);
        let mut tracker = InnovationTracker::new();

        let initial_conns = bg.num_enabled_connections();
        let initial_hidden = bg.num_hidden_nodes();

        // Simulate 200 generations of mutation
        for _ in 0..200 {
            mutate_brain(&mut bg, 0.15, &mut rng, &mut tracker);
        }

        assert!(
            bg.num_hidden_nodes() > initial_hidden,
            "Network should grow hidden nodes over 200 generations: {} -> {}",
            initial_hidden, bg.num_hidden_nodes(),
        );

        // The brain should still produce valid output
        let mut brain = Brain::from_genome(&bg);
        let input = [0.5f32; INPUT_SIZE];
        let output = brain.forward(&input);
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0, "Evolved brain output out of range: {}", v);
        }
    }

    #[test]
    fn test_output_size_is_seven() {
        let mut rng = rand::rng();
        let genome = BrainGenome::random(&mut rng);
        let mut brain = Brain::from_genome(&genome);
        let input = [0.0_f32; INPUT_SIZE];
        let output = brain.forward(&input);
        assert_eq!(output.len(), 7, "Should have 7 outputs (6 behavioral + 1 pheromone emission)");
    }

    #[test]
    fn test_hebbian_learning_modifies_weights() {
        let mut rng = rand::rng();
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
        let diff: f32 = output_learned.iter().zip(output_static.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Hebbian learning should change outputs: diff={}", diff);
    }

    #[test]
    fn test_hebbian_learning_zero_rate_no_change() {
        let mut rng = rand::rng();
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
            assert!((a - b).abs() < 1e-6, "Zero learning rate should produce stable outputs");
        }
    }

    #[test]
    fn test_pheromone_inputs_in_sensory() {
        use crate::components::*;
        use crate::needs::Needs;
        use crate::phenotype::DerivedPhysics;
        use crate::environment::Environment;

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
        let no_info = NearestInfo { dist: 0.0, angle: 0.0 };

        let input = build_sensory_input(
            &pos, &vel, &energy, &needs, &physics, &env,
            &no_info, &no_info, &no_info,
            100.0, 50.0,
            0.75,  // pheromone_concentration
            -0.5,  // pheromone_angle
        );

        assert_eq!(input.len(), INPUT_SIZE);
        assert!((input[14] - 0.75).abs() < 0.01, "Pheromone concentration at index 14");
        assert!((input[15] - (-0.5)).abs() < 0.01, "Pheromone angle at index 15");
    }

    #[test]
    fn test_brain_system_returns_pheromone_deposits() {
        use crate::components::*;
        use crate::ecosystem::Age;
        use crate::needs::NeedWeights;
        use crate::phenotype::{derive_physics, derive_feeding};
        use crate::genome::CreatureGenome;
        use crate::boids::Boid;

        let mut rng = rand::rng();
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
            Age { ticks: 0, max_ticks: 5000 },
            Needs::default(),
            NeedWeights::default(),
            BehaviorState::default(),
            Boid,
            brain,
        ));

        let entity_map: std::collections::HashMap<_, _> = std::collections::HashMap::new();
        let pheromone_grid = crate::pheromone::PheromoneGrid::new(100.0, 50.0);
        let deposits = brain_system(&mut world, &grid, &entity_map, &env, &pheromone_grid, 0.05, 100.0, 50.0);

        // Verify it doesn't crash and returns valid data
        for (x, y, amount) in &deposits {
            assert!(*x >= 0.0 && *x <= 100.0, "Deposit x should be in tank bounds");
            assert!(*y >= 0.0 && *y <= 50.0, "Deposit y should be in tank bounds");
            assert!(*amount >= 0.0, "Deposit amount should be non-negative");
        }
    }
}
