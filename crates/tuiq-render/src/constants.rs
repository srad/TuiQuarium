//! Tunable rendering and recording constants.
//!
//! Adjust these to trade quality vs. file size / performance.

/// Font size (px) used when saving PNG screenshots.
pub const SCREENSHOT_FONT_SIZE: f32 = 20.0;

/// Font size (px) used when encoding GIF frames.
/// Lower → smaller files; 20 = same as screenshots.
pub const GIF_FONT_SIZE: f32 = 16.0;

/// GIF recording frame rate (frames per second).
pub const GIF_FPS: u32 = 10;

/// Duration (seconds) a flash message stays visible.
pub const FLASH_DURATION_SECS: f32 = 3.0;
