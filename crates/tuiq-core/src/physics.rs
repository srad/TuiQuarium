use hecs::World;

use crate::components::{Appearance, BoundingBox, Direction, Position, Velocity};
use crate::phenotype::DerivedPhysics;

/// Update positions from velocities and bounce off tank walls.
pub fn physics_system(world: &mut World, dt: f32, tank_w: f32, tank_h: f32) {
    for (pos, vel, bbox, appearance, physics) in world.query_mut::<(
        &mut Position,
        &mut Velocity,
        &BoundingBox,
        &mut Appearance,
        Option<&DerivedPhysics>,
    )>() {
        // Integrate position
        pos.x += vel.vx * dt;
        pos.y += vel.vy * dt;

        // Bounce off left/right walls
        if pos.x < 0.0 {
            pos.x = 0.0;
            vel.vx = vel.vx.abs();
        } else if pos.x + bbox.w > tank_w {
            pos.x = tank_w - bbox.w;
            vel.vx = -vel.vx.abs();
        }

        // Bounce off top/bottom walls (leave room for substrate)
        let substrate_h = 2.0;
        let is_passive = physics.map(|p| p.max_speed < 0.1).unwrap_or(false);

        if pos.y < 0.0 {
            pos.y = 0.0;
            vel.vy = vel.vy.abs();
        } else if pos.y + bbox.h > tank_h - substrate_h {
            pos.y = tank_h - substrate_h - bbox.h;
            if is_passive {
                vel.vy = 0.0; // Stop at bottom
                vel.vx = 0.0; // Friction stops horizontal movement too
            } else {
                vel.vy = -vel.vy.abs(); // Bounce
            }
        }

        // Update facing direction based on velocity
        if vel.vx > 0.1 {
            appearance.facing = Direction::Right;
        } else if vel.vx < -0.1 {
            appearance.facing = Direction::Left;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::*;

    fn spawn_fish(world: &mut World, x: f32, y: f32, vx: f32, vy: f32) -> hecs::Entity {
        let frame = AsciiFrame::from_rows(vec![r"  /\  ", r" /o \==>", r" \  /==>", r"  \/  "]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame]],
            facing: Direction::Right,
            color_index: 0,
        };
        world.spawn((
            Position { x, y },
            Velocity { vx, vy },
            BoundingBox { w: 8.0, h: 4.0 },
            appearance,
        ))
    }

    #[test]
    fn test_position_integration() {
        let mut world = World::new();
        let e = spawn_fish(&mut world, 10.0, 5.0, 2.0, 1.0);
        physics_system(&mut world, 1.0, 100.0, 30.0);
        let pos = world.get::<&Position>(e).unwrap();
        assert!((pos.x - 12.0).abs() < 0.01);
        assert!((pos.y - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_bounce_right_wall() {
        let mut world = World::new();
        let e = spawn_fish(&mut world, 95.0, 5.0, 10.0, 0.0);
        // Tank width 100, fish width 8, so max x = 92
        physics_system(&mut world, 1.0, 100.0, 30.0);
        let vel = world.get::<&Velocity>(e).unwrap();
        assert!(vel.vx < 0.0, "Should bounce off right wall");
    }

    #[test]
    fn test_bounce_left_wall() {
        let mut world = World::new();
        let e = spawn_fish(&mut world, 1.0, 5.0, -5.0, 0.0);
        physics_system(&mut world, 1.0, 100.0, 30.0);
        let vel = world.get::<&Velocity>(e).unwrap();
        assert!(vel.vx > 0.0, "Should bounce off left wall");
    }

    #[test]
    fn test_facing_updates() {
        let mut world = World::new();
        let e = spawn_fish(&mut world, 50.0, 5.0, -5.0, 0.0);
        physics_system(&mut world, 1.0, 100.0, 30.0);
        let app = world.get::<&Appearance>(e).unwrap();
        assert_eq!(app.facing, Direction::Left);
    }
}
