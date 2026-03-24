//! Pre-crafted ASCII art templates for each creature type.
//! Each template provides animation frames for different actions.

use tuiq_core::components::AsciiFrame;

/// Creature template identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CreatureTemplate {
    SmallFish,
    TropicalFish,
    Angelfish,
    Jellyfish,
    Crab,
    Seaweed,
}

/// Returns (swim_frames, idle_frames) for each creature template.
/// All frames face RIGHT. Left-facing is done by flipping at render time.
pub fn get_template_frames(template: CreatureTemplate) -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    match template {
        CreatureTemplate::SmallFish => small_fish_frames(),
        CreatureTemplate::TropicalFish => tropical_fish_frames(),
        CreatureTemplate::Angelfish => angelfish_frames(),
        CreatureTemplate::Jellyfish => jellyfish_frames(),
        CreatureTemplate::Crab => crab_frames(),
        CreatureTemplate::Seaweed => seaweed_frames(),
    }
}

fn small_fish_frames() -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let swim = vec![
        AsciiFrame::from_rows(vec![
            r"  ,__,  ",
            r" / o  \>",
            r" \_   />",
            r"   '--' ",
        ]),
        AsciiFrame::from_rows(vec![
            r"  ,__,   ",
            r" / o  \>>",
            r" \_   /  ",
            r"   '--'  ",
        ]),
        AsciiFrame::from_rows(vec![
            r"  ,__,  ",
            r" / o  \>",
            r" \_   / ",
            r"   '--'>",
        ]),
    ];
    let idle = vec![
        AsciiFrame::from_rows(vec![
            r"  ,__,  ",
            r" / o  \>",
            r" \_   />",
            r"   '--' ",
        ]),
        AsciiFrame::from_rows(vec![
            r"   ,__,  ",
            r"  / o  \>",
            r"  \_   />",
            r"    '--' ",
        ]),
    ];
    (swim, idle)
}

fn tropical_fish_frames() -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let swim = vec![
        AsciiFrame::from_rows(vec![
            r"       /\          ",
            r"      /  \         ",
            r"  ___/  o \________",
            r" /  ~  ~  ~ ~  ===<",
            r" \___  ~ ~ ~______/",
            r"     \    /        ",
            r"      \  /         ",
            r"       \/          ",
        ]),
        AsciiFrame::from_rows(vec![
            r"       /\           ",
            r"      /  \          ",
            r"  ___/  o \_________",
            r" /  ~  ~  ~ ~  ====<",
            r" \___  ~ ~ ~_______/",
            r"     \    /         ",
            r"      \  /          ",
            r"       \/           ",
        ]),
        AsciiFrame::from_rows(vec![
            r"       /\         ",
            r"      /  \        ",
            r"  ___/  o \_______",
            r" /  ~  ~  ~ ~  ==<",
            r" \___  ~ ~ ~_____/",
            r"     \    /       ",
            r"      \  /        ",
            r"       \/         ",
        ]),
    ];
    let idle = vec![
        AsciiFrame::from_rows(vec![
            r"       /\         ",
            r"      /  \        ",
            r"  ___/  o \___    ",
            r" /  ~  ~  ~ ~ \=>",
            r" \___  ~ ~ ~__/  ",
            r"     \    /      ",
            r"      \  /       ",
            r"       \/        ",
        ]),
        AsciiFrame::from_rows(vec![
            r"        /\         ",
            r"       /  \        ",
            r"   ___/  o \___    ",
            r"  /  ~  ~  ~ ~ \=>",
            r"  \___  ~ ~ ~__/  ",
            r"      \    /      ",
            r"       \  /       ",
            r"        \/        ",
        ]),
    ];
    (swim, idle)
}

fn angelfish_frames() -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let swim = vec![
        AsciiFrame::from_rows(vec![
            r"     /\      ",
            r"    / .\     ",
            r"   / o  \    ",
            r"  /  ~~  \}=>",
            r"  \  ~~  /}=>",
            r"   \    /    ",
            r"    \  /     ",
            r"     \/      ",
        ]),
        AsciiFrame::from_rows(vec![
            r"     /\       ",
            r"    / .\      ",
            r"   / o  \     ",
            r"  /  ~~  \}==>",
            r"  \  ~~  /}==>",
            r"   \    /     ",
            r"    \  /      ",
            r"     \/       ",
        ]),
        AsciiFrame::from_rows(vec![
            r"     /\     ",
            r"    / .\    ",
            r"   / o  \   ",
            r"  /  ~~  \}>",
            r"  \  ~~  /}>",
            r"   \    /   ",
            r"    \  /    ",
            r"     \/     ",
        ]),
    ];
    let idle = vec![
        AsciiFrame::from_rows(vec![
            r"     /\     ",
            r"    / .\    ",
            r"   / o  \   ",
            r"  /  ~~  \}>",
            r"  \  ~~  /}>",
            r"   \    /   ",
            r"    \  /    ",
            r"     \/     ",
        ]),
        AsciiFrame::from_rows(vec![
            r"      /\     ",
            r"     / .\    ",
            r"    / o  \   ",
            r"   /  ~~  \}>",
            r"   \  ~~  /}>",
            r"    \    /   ",
            r"     \  /    ",
            r"      \/     ",
        ]),
    ];
    (swim, idle)
}

fn jellyfish_frames() -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let swim = vec![
        AsciiFrame::from_rows(vec![
            r"   .---.   ",
            r"  / o o \  ",
            r" (  ___  ) ",
            r"  '-----'  ",
            r"   |\ /|   ",
            r"   | X |   ",
            r"   |/ \|   ",
            r"   /   \   ",
        ]),
        AsciiFrame::from_rows(vec![
            r"   .---.   ",
            r"  / o o \  ",
            r" (  ___  ) ",
            r"  '-----'  ",
            r"   |/ \|   ",
            r"   | X |   ",
            r"   |\ /|   ",
            r"   \   /   ",
        ]),
        AsciiFrame::from_rows(vec![
            r"   .---.   ",
            r"  / o o \  ",
            r" (  ___  ) ",
            r"  '-----'  ",
            r"   || ||   ",
            r"   || ||   ",
            r"   || ||   ",
            r"   |   |   ",
        ]),
    ];
    let idle = vec![
        AsciiFrame::from_rows(vec![
            r"   .---.   ",
            r"  / o o \  ",
            r" (       ) ",
            r"  '-----'  ",
            r"   |   |   ",
            r"   |   |   ",
            r"   |   |   ",
            r"    \ /    ",
        ]),
        AsciiFrame::from_rows(vec![
            r"   .___.   ",
            r"  ( o o )  ",
            r" (       ) ",
            r"  '-----'  ",
            r"   |   |   ",
            r"    | |    ",
            r"    | |    ",
            r"     V     ",
        ]),
    ];
    (swim, idle)
}

fn crab_frames() -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    let swim = vec![
        AsciiFrame::from_rows(vec![
            r" _         _ ",
            r"| |  .-.  | |",
            r"| | ( o ) | |",
            r" \_|-----|_/ ",
            r"   | | | |   ",
            r"  _/ | | \_  ",
            r" /   | |   \ ",
        ]),
        AsciiFrame::from_rows(vec![
            r"  _       _  ",
            r" \ | .-. | / ",
            r"  || ( o ) ||",
            r"  \_|-----|_/",
            r"    | | | |  ",
            r"   / | | \   ",
            r"  /  | |  \  ",
        ]),
    ];
    let idle = vec![
        AsciiFrame::from_rows(vec![
            r" _         _ ",
            r"| |  .-.  | |",
            r"| | ( o ) | |",
            r" \_|-----|_/ ",
            r"   | | | |   ",
            r"  _/ | | \_  ",
            r" /   | |   \ ",
        ]),
    ];
    (swim, idle)
}

fn seaweed_frames() -> (Vec<AsciiFrame>, Vec<AsciiFrame>) {
    // Seaweed sways — same frames for "swim" and "idle"
    let sway = vec![
        AsciiFrame::from_rows(vec![
            r"    )  ",
            r"   (   ",
            r"    )  ",
            r"   ( ) ",
            r"    )) ",
            r"   ((  ",
            r"  _))_ ",
            r" |||||)",
        ]),
        AsciiFrame::from_rows(vec![
            r"   (   ",
            r"    )  ",
            r"   (   ",
            r"  ( )  ",
            r"  ((   ",
            r"   ))  ",
            r"  _((_ ",
            r" (|||||",
        ]),
        AsciiFrame::from_rows(vec![
            r"    )  ",
            r"    )  ",
            r"   (   ",
            r"   ( ) ",
            r"   ))  ",
            r"   ((  ",
            r"  _)(_ ",
            r" |||||)",
        ]),
    ];
    (sway.clone(), sway)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_templates_have_frames() {
        let templates = [
            CreatureTemplate::SmallFish,
            CreatureTemplate::TropicalFish,
            CreatureTemplate::Angelfish,
            CreatureTemplate::Jellyfish,
            CreatureTemplate::Crab,
            CreatureTemplate::Seaweed,
        ];
        for t in templates {
            let (swim, idle) = get_template_frames(t);
            assert!(!swim.is_empty(), "{t:?} has no swim frames");
            assert!(!idle.is_empty(), "{t:?} has no idle frames");
            for f in &swim {
                assert!(f.width > 0 && f.height > 0, "{t:?} swim frame has zero size");
            }
            for f in &idle {
                assert!(f.width > 0 && f.height > 0, "{t:?} idle frame has zero size");
            }
        }
    }

    #[test]
    fn test_frames_consistent_dimensions() {
        let templates = [
            CreatureTemplate::SmallFish,
            CreatureTemplate::TropicalFish,
            CreatureTemplate::Angelfish,
            CreatureTemplate::Jellyfish,
            CreatureTemplate::Crab,
            CreatureTemplate::Seaweed,
        ];
        for t in templates {
            let (swim, idle) = get_template_frames(t);
            // All swim frames should have the same height
            let swim_h = swim[0].height;
            for f in &swim {
                assert_eq!(f.height, swim_h, "{t:?} swim frames have inconsistent height");
            }
            let idle_h = idle[0].height;
            for f in &idle {
                assert_eq!(f.height, idle_h, "{t:?} idle frames have inconsistent height");
            }
        }
    }
}
