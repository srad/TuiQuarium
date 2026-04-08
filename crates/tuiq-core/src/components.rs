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
        Self {
            rows,
            width,
            height,
        }
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

/// Producer colony growth stage — derived from reserve status, biomass fill,
/// and accumulated stress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProducerStage {
    Cell,
    Patch,
    Mature,
    Broadcasting,
    Collapsing,
}

/// Tracks how long a producer colony has been alive (in simulation seconds).
#[derive(Debug, Clone)]
pub struct ProducerAge {
    pub seconds: f32,
}

/// Marker for producer lineages that should remain visually rooted to the substrate.
#[derive(Debug, Clone, Copy, Default)]
pub struct RootedMacrophyte;

/// Dynamic consumer life-history state.
///
/// Research note: aquatic consumer reproduction depends on maturation delay and
/// sustained energetic surplus rather than a simple elapsed-time timer, so the
/// model tracks condition, maturation, and reproductive reserves explicitly.
#[derive(Debug, Clone)]
pub struct ConsumerState {
    pub reserve_buffer: f32,
    pub maturity_progress: f32,
    pub reproductive_buffer: f32,
    pub brood_cooldown: f32,
    pub recent_assimilation: f32,
}

impl Default for ConsumerState {
    fn default() -> Self {
        Self {
            reserve_buffer: 0.2,
            maturity_progress: 0.0,
            reproductive_buffer: 0.0,
            brood_cooldown: 0.0,
            recent_assimilation: 0.0,
        }
    }
}

impl ConsumerState {
    pub fn is_adult(&self) -> bool {
        self.maturity_progress >= 1.0
    }
}

/// How a producer offspring was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProducerOrigin {
    Broadcast,
    Fragment,
}

/// Dynamic producer state beyond the genome.
///
/// `Energy` stores the reserve carbohydrate pool. This component stores
/// slower-changing biomass pools and local ecological pressure.
#[derive(Debug, Clone, Default)]
pub struct ProducerState {
    /// Long-lived attachment or support biomass.
    pub structural_biomass: f32,
    /// Actively photosynthetic biomass exposed to light and grazers.
    pub leaf_biomass: f32,
    /// Dormant storage biomass available for recovery after stress.
    pub belowground_reserve: f32,
    /// Regenerative potential available for regrowth and local fragmentation.
    pub meristem_bank: f32,
    /// Attached or suspended fouling that shades active biomass.
    pub epiphyte_load: f32,
    /// Cooldown for broad dispersal propagules.
    pub seed_cooldown: f32,
    /// Cooldown for local fragmentation.
    pub clonal_cooldown: f32,
    pub stress_time: f32,
    pub propagule_kind: Option<ProducerOrigin>,
}

impl ProducerState {
    pub fn total_biomass(&self) -> f32 {
        self.structural_biomass + self.leaf_biomass + self.belowground_reserve
    }

    pub fn aboveground_biomass(&self) -> f32 {
        self.structural_biomass + self.leaf_biomass
    }
}
