/// Position in the tank (float for smooth simulation, truncated to cells for rendering).
#[derive(Debug, Clone)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

/// Velocity in cells per second.
#[derive(Debug, Clone)]
pub struct Velocity {
    pub vx: f32,
    pub vy: f32,
}

/// Bounding box for collision and rendering (width, height in cells).
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub w: f32,
    pub h: f32,
}

/// Direction the creature is facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

/// Visual appearance: pre-rendered ASCII art frames indexed by animation action.
#[derive(Debug, Clone)]
pub struct Appearance {
    /// Animation frames per action. Index by [AnimAction as usize][frame_index].
    pub frame_sets: Vec<Vec<AsciiFrame>>,
    pub facing: Direction,
    /// Color index for rendering (renderer maps this to actual color).
    pub color_index: u8,
}

/// A single frame of multi-line ASCII art.
#[derive(Debug, Clone)]
pub struct AsciiFrame {
    /// Each row as a string.
    pub rows: Vec<String>,
    pub width: u16,
    pub height: u16,
}

impl AsciiFrame {
    pub fn from_rows(rows: Vec<&str>) -> Self {
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u16;
        let height = rows.len() as u16;
        let rows = rows.into_iter().map(|s| s.to_string()).collect();
        Self { rows, width, height }
    }

    /// Return a horizontally flipped version of this frame.
    pub fn flip_horizontal(&self) -> Self {
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|row| {
                let padded: String = format!("{:<width$}", row, width = self.width as usize);
                padded
                    .chars()
                    .rev()
                    .map(|c| match c {
                        '>' => '<',
                        '<' => '>',
                        '/' => '\\',
                        '\\' => '/',
                        '(' => ')',
                        ')' => '(',
                        '{' => '}',
                        '}' => '{',
                        '[' => ']',
                        ']' => '[',
                        other => other,
                    })
                    .collect()
            })
            .collect();
        Self {
            rows,
            width: self.width,
            height: self.height,
        }
    }
}

/// Animation actions (used as indices into Appearance.frame_sets).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum AnimAction {
    Swim = 0,
    Idle = 1,
}

pub const ANIM_ACTION_COUNT: usize = 2;

/// Per-entity animation state.
#[derive(Debug, Clone)]
pub struct AnimationState {
    pub current_action: AnimAction,
    pub frame_index: usize,
    pub frame_timer: f32,
    /// Seconds per frame.
    pub frame_duration: f32,
}

impl AnimationState {
    pub fn new(frame_duration: f32) -> Self {
        Self {
            current_action: AnimAction::Swim,
            frame_index: 0,
            frame_timer: 0.0,
            frame_duration,
        }
    }
}
