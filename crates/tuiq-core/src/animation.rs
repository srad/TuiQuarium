use hecs::World;

use crate::components::{AnimationState, Appearance};

/// Advance animation frame timers and cycle through frames.
pub fn animation_system(world: &mut World, dt: f32) {
    for (anim, appearance) in world.query_mut::<(&mut AnimationState, &Appearance)>() {
        let action_idx = anim.current_action as usize;
        if action_idx >= appearance.frame_sets.len() {
            continue;
        }
        let frame_count = appearance.frame_sets[action_idx].len();
        if frame_count == 0 {
            continue;
        }

        anim.frame_timer += dt;
        if anim.frame_timer >= anim.frame_duration {
            anim.frame_timer -= anim.frame_duration;
            anim.frame_index = (anim.frame_index + 1) % frame_count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::*;

    #[test]
    fn test_frame_cycling() {
        let mut world = World::new();
        let frame0 = AsciiFrame::from_rows(vec!["frame0"]);
        let frame1 = AsciiFrame::from_rows(vec!["frame1"]);
        let frame2 = AsciiFrame::from_rows(vec!["frame2"]);
        let appearance = Appearance {
            frame_sets: vec![vec![frame0, frame1, frame2]],
            facing: Direction::Right,
            color_index: 0,
        };
        let anim = AnimationState::new(0.2); // 5 fps
        let e = world.spawn((appearance, anim));

        // After 0.19s, still on frame 0
        animation_system(&mut world, 0.19);
        assert_eq!(world.get::<&AnimationState>(e).unwrap().frame_index, 0);

        // After another 0.02s (total 0.21s), should advance to frame 1
        animation_system(&mut world, 0.02);
        assert_eq!(world.get::<&AnimationState>(e).unwrap().frame_index, 1);

        // Advance past frame 2 and wrap to frame 0
        animation_system(&mut world, 0.2);
        assert_eq!(world.get::<&AnimationState>(e).unwrap().frame_index, 2);
        animation_system(&mut world, 0.2);
        assert_eq!(world.get::<&AnimationState>(e).unwrap().frame_index, 0);
    }
}
