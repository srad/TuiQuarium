//! Spatial hash grid for efficient neighbor queries.

use std::collections::HashMap;

use hecs::{Entity, World};

use crate::components::Position;

/// A spatial hash grid that buckets entities by grid cell.
/// Stores entity positions alongside IDs for distance-filtered queries.
pub struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<(Entity, f32, f32)>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_key(&self, x: f32, y: f32) -> (i32, i32) {
        (
            (x / self.cell_size).floor() as i32,
            (y / self.cell_size).floor() as i32,
        )
    }

    /// Rebuild the grid from the current world state.
    pub fn rebuild(&mut self, world: &World) {
        for v in self.cells.values_mut() {
            v.clear();
        }
        for (entity, pos) in &mut world.query::<(Entity, &Position)>() {
            let key = self.cell_key(pos.x, pos.y);
            self.cells
                .entry(key)
                .or_default()
                .push((entity, pos.x, pos.y));
        }
    }

    /// Find all entities within `radius` of the given position.
    /// Filters by actual distance² — only returns entities truly within radius.
    pub fn neighbors(&self, x: f32, y: f32, radius: f32) -> Vec<Entity> {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let center = self.cell_key(x, y);
        let radius_sq = radius * radius;

        // Estimate capacity: ~4 entities per cell on average
        let estimated = ((2 * r_cells + 1) * (2 * r_cells + 1) * 4) as usize;
        let mut result = Vec::with_capacity(estimated.min(256));

        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                let key = (center.0 + dx, center.1 + dy);
                if let Some(entities) = self.cells.get(&key) {
                    for &(entity, ex, ey) in entities {
                        let ddx = ex - x;
                        let ddy = ey - y;
                        if ddx * ddx + ddy * ddy <= radius_sq {
                            result.push(entity);
                        }
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Position;

    #[test]
    fn test_neighbors_within_radius() {
        let mut world = World::new();
        let e1 = world.spawn((Position { x: 5.0, y: 5.0 },));
        let e2 = world.spawn((Position { x: 7.0, y: 5.0 },));
        let _e3 = world.spawn((Position { x: 50.0, y: 50.0 },));

        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);

        let nearby = grid.neighbors(5.0, 5.0, 10.0);
        assert!(nearby.contains(&e1));
        assert!(nearby.contains(&e2));
        assert_eq!(nearby.len(), 2);
    }

    #[test]
    fn test_empty_grid() {
        let world = World::new();
        let mut grid = SpatialGrid::new(10.0);
        grid.rebuild(&world);
        let nearby = grid.neighbors(0.0, 0.0, 10.0);
        assert!(nearby.is_empty());
    }

    #[test]
    fn test_distance_filtering() {
        let mut world = World::new();
        let e_close = world.spawn((Position { x: 10.0, y: 10.0 },));
        let _e_far = world.spawn((Position { x: 18.0, y: 10.0 },));

        let mut grid = SpatialGrid::new(6.0);
        grid.rebuild(&world);

        // Radius 5: should only get e_close (distance 0), not e_far (distance 8)
        let nearby = grid.neighbors(10.0, 10.0, 5.0);
        assert!(nearby.contains(&e_close));
        assert_eq!(nearby.len(), 1, "Should filter out entity at distance 8");
    }
}
