/// Pheromone concentration grid for chemical signaling.
/// Creatures deposit pheromones; concentrations decay and diffuse over time.

pub struct PheromoneGrid {
    width: usize,  // number of cells in X
    height: usize, // number of cells in Y
    cell_size: f32,
    cells: Vec<f32>,
}

const CELL_SIZE: f32 = 4.0;
const DECAY_RATE: f32 = 0.95; // per tick (at 20Hz → ~3.5s half-life)
const DIFFUSION_RATE: f32 = 0.05; // fraction spread to each neighbor per tick

impl PheromoneGrid {
    pub fn new(tank_width: f32, tank_height: f32) -> Self {
        let w = (tank_width / CELL_SIZE).ceil() as usize;
        let h = (tank_height / CELL_SIZE).ceil() as usize;
        let w = w.max(1);
        let h = h.max(1);
        Self {
            width: w,
            height: h,
            cell_size: CELL_SIZE,
            cells: vec![0.0; w * h],
        }
    }

    fn cell_index(&self, cx: usize, cy: usize) -> usize {
        cy * self.width + cx
    }

    fn to_cell(&self, x: f32, y: f32) -> (usize, usize) {
        let cx = ((x / self.cell_size).floor() as usize).min(self.width.saturating_sub(1));
        let cy = ((y / self.cell_size).floor() as usize).min(self.height.saturating_sub(1));
        (cx, cy)
    }

    /// Deposit pheromone at a world position.
    pub fn deposit(&mut self, x: f32, y: f32, amount: f32) {
        if amount <= 0.0 {
            return;
        }
        let (cx, cy) = self.to_cell(x, y);
        let idx = self.cell_index(cx, cy);
        self.cells[idx] = (self.cells[idx] + amount).min(1.0);
    }

    /// Sample pheromone concentration at a world position (0.0–1.0).
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let (cx, cy) = self.to_cell(x, y);
        self.cells[self.cell_index(cx, cy)]
    }

    /// Compute the gradient direction at a world position.
    /// Returns (gx, gy) as a normalized direction toward increasing concentration,
    /// or (0, 0) if flat. Values suitable for atan2 → angle computation.
    pub fn gradient(&self, x: f32, y: f32) -> (f32, f32) {
        let (cx, cy) = self.to_cell(x, y);

        // Sample neighbors (clamp at boundaries)
        let left = if cx > 0 {
            self.cells[self.cell_index(cx - 1, cy)]
        } else {
            0.0
        };
        let right = if cx + 1 < self.width {
            self.cells[self.cell_index(cx + 1, cy)]
        } else {
            0.0
        };
        let up = if cy > 0 {
            self.cells[self.cell_index(cx, cy - 1)]
        } else {
            0.0
        };
        let down = if cy + 1 < self.height {
            self.cells[self.cell_index(cx, cy + 1)]
        } else {
            0.0
        };

        let gx = right - left;
        let gy = down - up;

        (gx, gy)
    }

    /// Decay all concentrations and diffuse to neighbors. Call once per tick.
    pub fn tick(&mut self) {
        let len = self.cells.len();
        let w = self.width;
        let h = self.height;

        // 1. Diffusion: compute diffusion deltas into a buffer
        let mut deltas = vec![0.0_f32; len];
        for cy in 0..h {
            for cx in 0..w {
                let idx = cy * w + cx;
                let val = self.cells[idx];
                if val < 0.001 {
                    continue;
                } // skip near-zero cells

                let spread = val * DIFFUSION_RATE;
                let mut neighbor_count = 0u32;

                if cx > 0 {
                    neighbor_count += 1;
                }
                if cx + 1 < w {
                    neighbor_count += 1;
                }
                if cy > 0 {
                    neighbor_count += 1;
                }
                if cy + 1 < h {
                    neighbor_count += 1;
                }

                if neighbor_count == 0 {
                    continue;
                }

                let per_neighbor = spread / neighbor_count as f32;
                deltas[idx] -= spread;

                if cx > 0 {
                    deltas[cy * w + (cx - 1)] += per_neighbor;
                }
                if cx + 1 < w {
                    deltas[cy * w + (cx + 1)] += per_neighbor;
                }
                if cy > 0 {
                    deltas[(cy - 1) * w + cx] += per_neighbor;
                }
                if cy + 1 < h {
                    deltas[(cy + 1) * w + cx] += per_neighbor;
                }
            }
        }

        // 2. Apply diffusion + decay
        for i in 0..len {
            self.cells[i] = ((self.cells[i] + deltas[i]) * DECAY_RATE).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_grid_is_zero() {
        let grid = PheromoneGrid::new(40.0, 30.0);
        assert!(grid.cells.iter().all(|&v| v == 0.0));
        assert_eq!(grid.width, 10);
        assert_eq!(grid.height, 8);
    }

    #[test]
    fn test_deposit_and_sample() {
        let mut grid = PheromoneGrid::new(40.0, 30.0);
        grid.deposit(10.0, 10.0, 0.5);
        let val = grid.sample(10.0, 10.0);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_deposit_clamps_to_one() {
        let mut grid = PheromoneGrid::new(40.0, 30.0);
        grid.deposit(5.0, 5.0, 0.8);
        grid.deposit(5.0, 5.0, 0.8);
        assert!(grid.sample(5.0, 5.0) <= 1.0);
    }

    #[test]
    fn test_decay_reduces_concentration() {
        let mut grid = PheromoneGrid::new(40.0, 30.0);
        grid.deposit(10.0, 10.0, 1.0);
        let before = grid.sample(10.0, 10.0);
        grid.tick();
        let after = grid.sample(10.0, 10.0);
        assert!(
            after < before,
            "Concentration should decay: {} -> {}",
            before,
            after
        );
    }

    #[test]
    fn test_diffusion_spreads_to_neighbors() {
        let mut grid = PheromoneGrid::new(40.0, 30.0);
        // Deposit in center cell
        let cx = grid.width / 2;
        let cy = grid.height / 2;
        let x = cx as f32 * CELL_SIZE + 1.0;
        let y = cy as f32 * CELL_SIZE + 1.0;
        grid.deposit(x, y, 1.0);

        // Neighbors should be zero before tick
        let right_x = (cx + 1) as f32 * CELL_SIZE + 1.0;
        assert_eq!(grid.sample(right_x, y), 0.0);

        grid.tick();

        // After tick, neighbor should have some concentration
        let neighbor_val = grid.sample(right_x, y);
        assert!(neighbor_val > 0.0, "Diffusion should spread to neighbors");
    }

    #[test]
    fn test_gradient_points_toward_source() {
        let mut grid = PheromoneGrid::new(40.0, 30.0);
        // Deposit to the right
        let cx = grid.width / 2;
        let cy = grid.height / 2;
        let right_x = (cx + 1) as f32 * CELL_SIZE + 1.0;
        let center_y = cy as f32 * CELL_SIZE + 1.0;
        grid.deposit(right_x, center_y, 1.0);

        // Gradient at center should point right (positive gx)
        let center_x = cx as f32 * CELL_SIZE + 1.0;
        let (gx, _gy) = grid.gradient(center_x, center_y);
        assert!(
            gx > 0.0,
            "Gradient should point toward higher concentration (right)"
        );
    }

    #[test]
    fn test_negative_deposit_ignored() {
        let mut grid = PheromoneGrid::new(40.0, 30.0);
        grid.deposit(5.0, 5.0, -0.5);
        assert_eq!(grid.sample(5.0, 5.0), 0.0);
    }

    #[test]
    fn test_sample_out_of_bounds_clamps() {
        let grid = PheromoneGrid::new(40.0, 30.0);
        // Should not panic, just clamp
        let _ = grid.sample(100.0, 100.0);
        let _ = grid.sample(-5.0, -5.0);
    }
}
