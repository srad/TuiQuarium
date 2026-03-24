//! Boids flocking algorithm (Craig Reynolds, 1986).
//! Three rules: separation, alignment, cohesion + wall avoidance.

use hecs::{Entity, World};

use crate::behavior::{BehaviorAction, BehaviorState};
use crate::components::{BoundingBox, Position, Velocity};
use crate::ecosystem::TrophicRole;
use crate::phenotype::DerivedPhysics;
use crate::spatial::SpatialGrid;

/// Parameters for the boids algorithm.
#[derive(Debug, Clone)]
pub struct BoidParams {
    pub separation_radius: f32,
    pub separation_weight: f32,
    pub alignment_radius: f32,
    pub alignment_weight: f32,
    pub cohesion_radius: f32,
    pub cohesion_weight: f32,
    pub wall_avoidance_dist: f32,
    pub wall_avoidance_weight: f32,
    pub max_speed: f32,
    pub max_force: f32,
    // New steering weights
    pub seek_weight: f32,
    pub flee_weight: f32,
}

impl Default for BoidParams {
    fn default() -> Self {
        Self {
            separation_radius: 5.0,
            separation_weight: 2.0,
            alignment_radius: 12.0,
            alignment_weight: 1.0,
            cohesion_radius: 15.0,
            cohesion_weight: 1.0,
            wall_avoidance_dist: 8.0,
            wall_avoidance_weight: 3.0,
            max_speed: 5.0,
            max_force: 1.5,
            seek_weight: 4.0,
            flee_weight: 6.0,
        }
    }
}

/// Marker component for entities that participate in boids flocking.
#[derive(Debug, Clone)]
pub struct Boid;

/// Run the boids flocking system + goal seeking.
/// Modifies velocities of all entities with a Boid component.
pub fn boids_system(
    world: &mut World,
    grid: &SpatialGrid,
    params: &BoidParams,
    tank_w: f32,
    tank_h: f32,
    dt: f32,
) {
    // Collect current state of boids
    let boid_states: Vec<(Entity, f32, f32, f32, f32, f32, BehaviorAction, TrophicRole, f32)> = {
        let mut states = Vec::new();
        for (entity, pos, vel, bbox, behavior, role, physics) in
            &mut world.query::<(
                Entity,
                &Position,
                &Velocity,
                &BoundingBox,
                &BehaviorState,
                &TrophicRole,
                &DerivedPhysics,
            )>()
        {
            if world.get::<&Boid>(entity).is_ok() {
                states.push((
                    entity,
                    pos.x,
                    pos.y,
                    vel.vx,
                    vel.vy,
                    bbox.w.max(bbox.h),
                    behavior.action,
                    *role,
                    physics.sensory_range,
                ));
            }
        }
        states
    };

    // Forces calculation
    let mut forces: Vec<(Entity, f32, f32)> = Vec::with_capacity(boid_states.len());

    for &(entity, px, py, vx, vy, size, action, role, sensory_range) in &boid_states {
        // Flocking accumulators
        let mut sep_x = 0.0_f32;
        let mut sep_y = 0.0_f32;
        let mut sep_count = 0;

        let mut align_vx = 0.0_f32;
        let mut align_vy = 0.0_f32;
        let mut align_count = 0;

        let mut coh_x = 0.0_f32;
        let mut coh_y = 0.0_f32;
        let mut coh_count = 0;

        // Steering accumulators
        let mut seek_target: Option<(f32, f32)> = None;
        let mut closest_target_dist = f32::MAX;
        
        let mut flee_target: Option<(f32, f32)> = None;
        let mut closest_threat_dist = f32::MAX;

        let neighbors = grid.neighbors(px, py, sensory_range.max(params.cohesion_radius));

        for &other in &neighbors {
            if other == entity {
                continue;
            }

            let other_pos = match world.get::<&Position>(other) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let ox = other_pos.x;
            let oy = other_pos.y;
            
            let dx = px - ox;
            let dy = py - oy;
            let dist = (dx * dx + dy * dy).sqrt().max(0.01);

            // 1. Goal Seeking Logic (Food/Prey)
            if action == BehaviorAction::Forage || action == BehaviorAction::Hunt {
                if let Ok(other_role) = world.get::<&TrophicRole>(other) {
                    let is_target = match action {
                        BehaviorAction::Forage => role.can_eat(*other_role), // Herbivore -> Producer
                        BehaviorAction::Hunt => role.can_eat(*other_role),   // Carnivore -> Herbivore
                        _ => false,
                    };

                    if is_target && dist < sensory_range && dist < closest_target_dist {
                        closest_target_dist = dist;
                        seek_target = Some((ox, oy));
                    }
                }
            }

            // 2. Flee Logic (Predators)
            if action == BehaviorAction::Flee {
                if let Ok(other_role) = world.get::<&TrophicRole>(other) {
                     // Check if other is a threat (Carnivore vs Herbivore)
                     let is_threat = other_role.can_eat(role);
                     if is_threat && dist < sensory_range && dist < closest_threat_dist {
                         closest_threat_dist = dist;
                         flee_target = Some((ox, oy));
                     }
                }
            }

            // 3. Flocking Logic (only with other boids)
            if world.get::<&Boid>(other).is_ok() {
                 let other_vel = match world.get::<&Velocity>(other) {
                    Ok(v) => v,
                    Err(_) => continue,
                 };
                // Separation
                let sep_dist = params.separation_radius + size; // Simplified size check
                if dist < sep_dist {
                    sep_x += dx / dist;
                    sep_y += dy / dist;
                    sep_count += 1;
                }
                // Alignment
                if dist < params.alignment_radius {
                    align_vx += other_vel.vx;
                    align_vy += other_vel.vy;
                    align_count += 1;
                }
                // Cohesion
                if dist < params.cohesion_radius {
                    coh_x += ox;
                    coh_y += oy;
                    coh_count += 1;
                }
            }
        }

        let mut force_x = 0.0_f32;
        let mut force_y = 0.0_f32;

        // Apply Steering (Seek)
        if let Some((tx, ty)) = seek_target {
            // Vector to target
            let dx = tx - px;
            let dy = ty - py;
            let dist = (dx * dx + dy * dy).sqrt().max(0.01);
            // Normalize and scale to max speed
            let desired_x = (dx / dist) * params.max_speed;
            let desired_y = (dy / dist) * params.max_speed;
            // Steering = Desired - Velocity
            force_x += (desired_x - vx) * params.seek_weight;
            force_y += (desired_y - vy) * params.seek_weight;
        }

        // Apply Steering (Flee)
        if let Some((tx, ty)) = flee_target {
            // Vector away from target
            let dx = px - tx;
            let dy = py - ty;
            let dist = (dx * dx + dy * dy).sqrt().max(0.01);
            let desired_x = (dx / dist) * params.max_speed;
            let desired_y = (dy / dist) * params.max_speed;
            force_x += (desired_x - vx) * params.flee_weight;
            force_y += (desired_y - vy) * params.flee_weight;
        }

        // Apply Flocking Forces
        if sep_count > 0 {
            force_x += (sep_x / sep_count as f32) * params.separation_weight;
            force_y += (sep_y / sep_count as f32) * params.separation_weight;
        }
        if align_count > 0 {
            let avg_vx = align_vx / align_count as f32;
            let avg_vy = align_vy / align_count as f32;
            force_x += (avg_vx - vx) * params.alignment_weight;
            force_y += (avg_vy - vy) * params.alignment_weight;
        }
        if coh_count > 0 {
            let center_x = coh_x / coh_count as f32;
            let center_y = coh_y / coh_count as f32;
            force_x += (center_x - px) * params.cohesion_weight * 0.01;
            force_y += (center_y - py) * params.cohesion_weight * 0.01;
        }

        // Wall avoidance
        let wd = params.wall_avoidance_dist;
        let ww = params.wall_avoidance_weight;
        if px < wd {
            force_x += (wd - px) / wd * ww;
        }
        if px > tank_w - wd {
            force_x -= (px - (tank_w - wd)) / wd * ww;
        }
        if py < wd {
            force_y += (wd - py) / wd * ww;
        }
        if py > tank_h - wd {
            force_y -= (py - (tank_h - wd)) / wd * ww;
        }

        // Clamp force
        let mag = (force_x * force_x + force_y * force_y).sqrt();
        if mag > params.max_force {
            force_x = force_x / mag * params.max_force;
            force_y = force_y / mag * params.max_force;
        }

        forces.push((entity, force_x, force_y));
    }

    // Apply forces to velocities
    for (entity, fx, fy) in forces {
        if let Ok(mut vel) = world.get::<&mut Velocity>(entity) {
            vel.vx += fx * dt;
            vel.vy += fy * dt;

            // Clamp speed
            let speed = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
            if speed > params.max_speed {
                vel.vx = vel.vx / speed * params.max_speed;
                vel.vy = vel.vy / speed * params.max_speed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::*;
    use crate::behavior::BehaviorState;
    use crate::ecosystem::TrophicRole;
    use crate::phenotype::DerivedPhysics;

    fn spawn_boid(world: &mut World, x: f32, y: f32, vx: f32, vy: f32) -> Entity {
        let frame = AsciiFrame::from_rows(vec!["<>"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        
        // Dummy physics
        let physics = DerivedPhysics {
            max_speed: 5.0,
            acceleration: 1.0,
            turn_radius: 1.0,
            drag_coefficient: 0.1,
            body_mass: 1.0,
            max_energy: 100.0,
            base_metabolism: 1.0,
            visual_profile: 1.0,
            camouflage: 0.0,
            sensory_range: 20.0,
        };

        world.spawn((
            Position { x, y },
            Velocity { vx, vy },
            BoundingBox { w: 2.0, h: 1.0 },
            appearance,
            AnimationState::new(0.2),
            Boid,
            BehaviorState::default(),
            TrophicRole::Herbivore,
            physics,
        ))
    }


    #[test]
    fn test_separation_pushes_apart() {
        let mut world = World::new();
        let e1 = spawn_boid(&mut world, 10.0, 10.0, 0.0, 0.0);
        let e2 = spawn_boid(&mut world, 11.0, 10.0, 0.0, 0.0);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let params = BoidParams {
            separation_weight: 5.0,
            alignment_weight: 0.0,
            cohesion_weight: 0.0,
            wall_avoidance_weight: 0.0,
            ..Default::default()
        };

        boids_system(&mut world, &grid, &params, 100.0, 100.0, 1.0);

        let v1 = world.get::<&Velocity>(e1).unwrap();
        let v2 = world.get::<&Velocity>(e2).unwrap();
        // e1 should be pushed left (negative vx), e2 right (positive vx)
        assert!(v1.vx < 0.0, "e1 should be pushed left, got {}", v1.vx);
        assert!(v2.vx > 0.0, "e2 should be pushed right, got {}", v2.vx);
    }

    #[test]
    fn test_alignment_matches_direction() {
        let mut world = World::new();
        let e1 = spawn_boid(&mut world, 10.0, 10.0, 0.0, 0.0);
        let _e2 = spawn_boid(&mut world, 15.0, 10.0, 3.0, 0.0);
        let _e3 = spawn_boid(&mut world, 13.0, 12.0, 3.0, 0.0);

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let params = BoidParams {
            separation_weight: 0.0,
            alignment_weight: 2.0,
            cohesion_weight: 0.0,
            wall_avoidance_weight: 0.0,
            ..Default::default()
        };

        boids_system(&mut world, &grid, &params, 100.0, 100.0, 1.0);

        let v1 = world.get::<&Velocity>(e1).unwrap();
        assert!(v1.vx > 0.0, "e1 should align rightward, got {}", v1.vx);
    }

    #[test]
    fn test_cohesion_pulls_toward_center() {
        let mut world = World::new();
        let e1 = spawn_boid(&mut world, 5.0, 50.0, 0.0, 0.0);
        let _e2 = spawn_boid(&mut world, 15.0, 50.0, 0.0, 0.0);
        let _e3 = spawn_boid(&mut world, 15.0, 55.0, 0.0, 0.0);

        let mut grid = SpatialGrid::new(20.0);
        grid.rebuild(&world);

        let params = BoidParams {
            separation_weight: 0.0,
            alignment_weight: 0.0,
            cohesion_weight: 5.0,
            wall_avoidance_weight: 0.0,
            ..Default::default()
        };

        boids_system(&mut world, &grid, &params, 100.0, 100.0, 1.0);

        let v1 = world.get::<&Velocity>(e1).unwrap();
        assert!(v1.vx > 0.0, "e1 should be pulled right toward group center, got {}", v1.vx);
    }
}
