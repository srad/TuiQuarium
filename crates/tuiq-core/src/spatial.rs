//! Spatial hash grid for efficient neighbor queries.

use std::collections::HashMap;

use hecs::{Entity, World};

use crate::components::Position;

/// A spatial hash grid that buckets entities by grid cell.
pub struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<Entity>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_key(&self, x: f32, y: f32) -> (i32, i32) {
        ((x / self.cell_size).floor() as i32, (y / self.cell_size).floor() as i32)
    }

    /// Rebuild the grid from the current world state.
    pub fn rebuild(&mut self, world: &World) {
        self.cells.clear();
        for (entity, pos) in &mut world.query::<(Entity, &Position)>() {
            let key = self.cell_key(pos.x, pos.y);
            self.cells.entry(key).or_default().push(entity);
        }
    }

    /// Find all entities within `radius` of the given position.
    pub fn neighbors(&self, x: f32, y: f32, radius: f32) -> Vec<Entity> {
        let mut result = Vec::new();
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let center = self.cell_key(x, y);

        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                let key = (center.0 + dx, center.1 + dy);
                if let Some(entities) = self.cells.get(&key) {
                    result.extend_from_slice(entities);
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
        // e3 is far away, shouldn't be in neighbors
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
}
