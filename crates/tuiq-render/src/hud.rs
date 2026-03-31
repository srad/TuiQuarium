//! HUD overlay: population stats, time, environment info, controls hint.

use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};
use tuiq_core::environment::{Environment, EventKind};
use tuiq_core::{DailyEcologySample, EcologyDiagnostics, SimStats};

fn total_producer_biomass(stats: &SimStats) -> f32 {
    stats.producer_leaf_biomass
        + stats.producer_structural_biomass
        + stats.producer_belowground_reserve
}

fn status_line(stats: &SimStats, env: &Environment, paused: bool, speed: f32, diversity: f32) -> String {
    let time_str = format_time(env.time_of_day);
    let day_night = if env.is_night() { "Night " } else { "Day   " };
    let temp_str = format!("{:>3.0}C", env.temperature);
    let light_str = format!("{:>3.0}%", env.light_level * 100.0);

    let event_str = match &env.active_event {
        Some(e) => match e.kind {
            EventKind::AlgaeBloom => format!(" | ALGAE {:>3.0}s", e.remaining),
            EventKind::FeedingFrenzy => format!(" | FRENZY {:>3.0}s", e.remaining),
            EventKind::ColdSnap => format!(" | COLD {:>3.0}s", e.remaining),
            EventKind::Earthquake => format!(" | QUAKE {:>3.0}s", e.remaining),
        },
        None => String::new(),
    };

    let pause_str = if paused { " [PAUSED]" } else { "" };
    let speed_str = format!("{:>5.1}x", speed);

    format!(
        " Pop:{:>3} | Gen:{:>4} | Cx:{:.2} | Sp:{:>2} | Div:{:.2} | Day {:>4} {} {} {} | Light:{} | {}{}{} ",
        stats.creature_count,
        stats.max_generation,
        stats.avg_complexity,
        stats.species_count,
        diversity,
        stats.elapsed_days + 1,
        time_str,
        day_night,
        temp_str,
        light_str,
        speed_str,
        pause_str,
        event_str,
    )
}

fn ecology_line(stats: &SimStats) -> String {
    format!(
        " C B/D:{:>3}/{:<3} | Prod B/D:{:>4}/{:<4} | A/J:{:>3}/{:<3} | ProdBio:{:>7.1} | NPP:{:>5.2} ",
        stats.creature_births,
        stats.creature_deaths,
        stats.producer_births,
        stats.producer_deaths,
        stats.adult_count,
        stats.juvenile_count,
        total_producer_biomass(stats),
        stats.rolling_producer_npp,
    )
}

fn controls_line() -> &'static str {
    " q:Quit  Space:Pause  L/R:Speed  U/D:Diversity  r:Reset  f:Feed  d:Diag  p:Screenshot  g:Record  ?:Help "
}

fn sparkline(values: &[f32], target_len: usize) -> String {
    const CHARS: &[u8] = b".:-=+*#%@";

    if target_len == 0 {
        return String::new();
    }

    if values.is_empty() {
        return ".".repeat(target_len.min(8).max(1));
    }

    let sample_count = values.len().min(target_len.max(1));
    let mut sampled = Vec::with_capacity(sample_count);
    for idx in 0..sample_count {
        let start = idx * values.len() / sample_count;
        let end = ((idx + 1) * values.len() / sample_count)
            .max(start + 1)
            .min(values.len());
        let slice = &values[start..end];
        sampled.push(slice.iter().sum::<f32>() / slice.len() as f32);
    }

    let min = sampled.iter().copied().fold(f32::INFINITY, f32::min);
    let max = sampled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).abs();

    if span < 1e-6 {
        let fill = if max.abs() < 1e-6 { '.' } else { '=' };
        return std::iter::repeat(fill).take(sampled.len()).collect();
    }

    sampled
        .into_iter()
        .map(|value| {
            let normalized = ((value - min) / span).clamp(0.0, 1.0);
            let idx = (normalized * (CHARS.len() - 1) as f32).round() as usize;
            CHARS[idx] as char
        })
        .collect()
}

fn history_series<F>(history: &[DailyEcologySample], current: f32, selector: F) -> Vec<f32>
where
    F: Fn(&DailyEcologySample) -> f32,
{
    let mut values: Vec<f32> = history.iter().map(selector).collect();
    values.push(current);
    values
}

fn diagnostics_lines(diagnostics: &EcologyDiagnostics, width: usize) -> Vec<String> {
    let instant = &diagnostics.instant;
    let history = diagnostics.daily_history.as_slice();
    let spark_width = width.saturating_sub(12).clamp(8, 24);

    let producer_series = history_series(history, instant.producer_total_biomass, |sample| {
        sample.producer_total_biomass
    });
    let consumer_series = history_series(history, instant.consumer_biomass, |sample| {
        sample.consumer_biomass
    });
    let balance_series = history_series(
        history,
        instant.rolling_producer_npp - instant.rolling_consumer_intake,
        |sample| sample.rolling_producer_npp - sample.rolling_consumer_intake,
    );

    vec![
        format!(
            " ProdTot:{:.1} | Active:{:.1} | Cons:{:.1} | C:P:{:.2} ",
            instant.producer_total_biomass,
            instant.producer_active_biomass,
            instant.consumer_biomass,
            instant.consumer_to_producer_biomass_ratio,
        ),
        format!(
            " NPP:{:.2} | Intake:{:.2} | Maint:{:.2} | I:N:{:.2} | M:I:{:.2} ",
            instant.rolling_producer_npp,
            instant.rolling_consumer_intake,
            instant.rolling_consumer_maintenance,
            instant.intake_to_npp_ratio,
            instant.maintenance_to_intake_ratio,
        ),
        format!(
            " dN:{:.1} dP:{:.1} | sN:{:.1} sP:{:.1} | Phy:{:.2} ",
            instant.dissolved_n,
            instant.dissolved_p,
            instant.sediment_n,
            instant.sediment_p,
            instant.phytoplankton_load,
        ),
        format!(" ProdHist {}", sparkline(&producer_series, spark_width)),
        format!(" ConsHist {}", sparkline(&consumer_series, spark_width)),
        format!(" BalHist  {}", sparkline(&balance_series, spark_width)),
    ]
}

/// Render the HUD overlay at the bottom of the screen.
pub fn render_hud(
    frame: &mut Frame,
    area: Rect,
    stats: &SimStats,
    diagnostics: Option<&EcologyDiagnostics>,
    env: &Environment,
    paused: bool,
    speed: f32,
    diversity: f32,
    is_recording: bool,
    recording_secs: u32,
) {
    if area.height < 1 {
        return;
    }

    let diagnostics_rows = area.height.saturating_sub(3) as usize;
    if let Some(diagnostics) = diagnostics {
        let lines = diagnostics_lines(diagnostics, area.width as usize);
        let visible_rows = diagnostics_rows.min(lines.len());
        for row in 0..visible_rows {
            let y = area.y + row as u16;
            let line_area = Rect::new(area.x, y, area.width, 1);
            let widget = Paragraph::new(Line::from(vec![Span::styled(
                lines[row].clone(),
                Style::default().fg(Color::DarkGray),
            )]))
            .alignment(Alignment::Left);
            frame.render_widget(widget, line_area);
        }
    }

    let rec_suffix = if is_recording {
        format!(" ● REC {:02}:{:02}", recording_secs / 60, recording_secs % 60)
    } else {
        String::new()
    };

    let status_text = status_line(stats, env, paused, speed, diversity);
    let ecology_text = ecology_line(stats);
    let controls_text = controls_line().to_string();

    let footer_start = area.y + area.height.saturating_sub(3);

    // Row 0: Status line (white) + optional REC indicator (red).
    if area.height >= 1 {
        let y = footer_start;
        let line_area = Rect::new(area.x, y, area.width, 1);
        let mut spans = vec![Span::styled(
            status_text,
            Style::default().fg(Color::White),
        )];
        if is_recording {
            spans.push(Span::styled(
                rec_suffix,
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        let widget = Paragraph::new(Line::from(spans)).alignment(Alignment::Left);
        frame.render_widget(widget, line_area);
    }

    // Row 1: Ecology line.
    if area.height >= 2 {
        let y = footer_start + 1;
        let line_area = Rect::new(area.x, y, area.width, 1);
        let widget = Paragraph::new(Line::from(vec![Span::styled(
            ecology_text,
            Style::default().fg(Color::Gray),
        )]))
        .alignment(Alignment::Left);
        frame.render_widget(widget, line_area);
    }

    // Row 2: Controls hint.
    if area.height >= 3 {
        let y = footer_start + 2;
        let line_area = Rect::new(area.x, y, area.width, 1);
        let widget = Paragraph::new(Line::from(vec![Span::styled(
            controls_text,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        )]))
        .alignment(Alignment::Center);
        frame.render_widget(widget, line_area);
    }
}

/// Format time_of_day (0-24 float) as HH:MM string.
fn format_time(hours: f32) -> String {
    let h = hours as u32 % 24;
    let m = ((hours - hours.floor()) * 60.0) as u32;
    format!("{:02}:{:02}", h, m)
}

/// Compute a centered popup rectangle inside `area`.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_width = area.width.min(area.width * percent_x / 100).max(40);
    let popup_height = area.height.min(area.height * percent_y / 100).max(20);
    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    Rect::new(x, y, popup_width, popup_height)
}

/// Render a help popup overlay explaining all HUD abbreviations.
pub fn render_help_popup(frame: &mut Frame, area: Rect) {
    let popup_area = centered_rect(75, 85, area);

    let header = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let key_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let desc = Style::default().fg(Color::White);
    let dim = Style::default().fg(Color::DarkGray);

    let lines = vec![
        Line::from(vec![Span::styled("  Controls", header)]),
        Line::from(vec![
            Span::styled("  q", key_style),
            Span::styled("       Quit the simulation", desc),
        ]),
        Line::from(vec![
            Span::styled("  Space", key_style),
            Span::styled("   Pause / resume", desc),
        ]),
        Line::from(vec![
            Span::styled("  L/R", key_style),
            Span::styled("     Decrease / increase simulation speed", desc),
        ]),
        Line::from(vec![
            Span::styled("  U/D", key_style),
            Span::styled("     Increase / decrease diversity coefficient", desc),
        ]),
        Line::from(vec![
            Span::styled("  r", key_style),
            Span::styled("       Reset speed and diversity to defaults", desc),
        ]),
        Line::from(vec![
            Span::styled("  f", key_style),
            Span::styled("       Feed - scatter extra food", desc),
        ]),
        Line::from(vec![
            Span::styled("  d", key_style),
            Span::styled("       Toggle diagnostics panel", desc),
        ]),
        Line::from(vec![
            Span::styled("  ?/h", key_style),
            Span::styled("     Toggle this help screen", desc),
        ]),
        Line::from(vec![
            Span::styled("  p", key_style),
            Span::styled("       Save PNG screenshot", desc),
        ]),
        Line::from(vec![
            Span::styled("  g", key_style),
            Span::styled("       Toggle GIF recording", desc),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![Span::styled("  Status Bar", header)]),
        Line::from(vec![
            Span::styled("  Pop", key_style),
            Span::styled("     Total creature population", desc),
        ]),
        Line::from(vec![
            Span::styled("  Gen", key_style),
            Span::styled("     Highest generation number reached", desc),
        ]),
        Line::from(vec![
            Span::styled("  Cx", key_style),
            Span::styled("      Average neural complexity (0.0-1.0)", desc),
        ]),
        Line::from(vec![
            Span::styled("  Sp", key_style),
            Span::styled("      Estimated species count (by genome clustering)", desc),
        ]),
        Line::from(vec![
            Span::styled("  Div", key_style),
            Span::styled("     Diversity coefficient - mutation strength/variety", desc),
        ]),
        Line::from(vec![
            Span::styled("  Day/Night", key_style),
            Span::styled(" In-game time, temperature, light level", desc),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![Span::styled("  Ecology Line", header)]),
        Line::from(vec![
            Span::styled("  C B/D", key_style),
            Span::styled("   Creature births / deaths (cumulative)", desc),
        ]),
        Line::from(vec![
            Span::styled("  Prod B/D", key_style),
            Span::styled("Producer (plant) births / deaths", desc),
        ]),
        Line::from(vec![
            Span::styled("  A/J", key_style),
            Span::styled("     Adult / juvenile creature count", desc),
        ]),
        Line::from(vec![
            Span::styled("  ProdBio", key_style),
            Span::styled(" Total producer biomass", desc),
        ]),
        Line::from(vec![
            Span::styled("  NPP", key_style),
            Span::styled("     Net Primary Productivity (plant growth rate)", desc),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![Span::styled("  Diagnostics Panel (d)", header)]),
        Line::from(vec![
            Span::styled("  ProdTot", key_style),
            Span::styled(" Total producer biomass  ", desc),
            Span::styled("Active", key_style),
            Span::styled("  Active (non-dormant) biomass", desc),
        ]),
        Line::from(vec![
            Span::styled("  Cons", key_style),
            Span::styled("    Consumer biomass         ", desc),
            Span::styled("C:P", key_style),
            Span::styled("     Consumer-to-producer ratio", desc),
        ]),
        Line::from(vec![
            Span::styled("  Intake", key_style),
            Span::styled("  Consumer energy intake     ", desc),
            Span::styled("Maint", key_style),
            Span::styled("   Consumer maintenance cost", desc),
        ]),
        Line::from(vec![
            Span::styled("  I:N", key_style),
            Span::styled("     Intake-to-NPP ratio     ", desc),
            Span::styled("M:I", key_style),
            Span::styled("     Maintenance-to-intake ratio", desc),
        ]),
        Line::from(vec![
            Span::styled("  dN/dP", key_style),
            Span::styled("   Dissolved nitrogen/phosphorus  ", desc),
            Span::styled("sN/sP", key_style),
            Span::styled("   Sediment N/P", desc),
        ]),
        Line::from(vec![
            Span::styled("  Phy", key_style),
            Span::styled("     Phytoplankton load (water clarity)", desc),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![Span::styled("  Sparklines", header)]),
        Line::from(vec![
            Span::styled("  ProdHist", key_style),
            Span::styled(" Producer biomass over time", desc),
        ]),
        Line::from(vec![
            Span::styled("  ConsHist", key_style),
            Span::styled(" Consumer biomass over time", desc),
        ]),
        Line::from(vec![
            Span::styled("  BalHist", key_style),
            Span::styled("  Balance (C:P ratio) over time", desc),
        ]),
        Line::from(vec![
            Span::styled("  Chars: ", dim),
            Span::styled(". : - = + * # % @", key_style),
            Span::styled("  (low -> high)", dim),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![Span::styled(
            "  Press ? or h to close",
            dim,
        )]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Help ")
        .title_alignment(Alignment::Center)
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    frame.render_widget(Clear, popup_area);
    frame.render_widget(paragraph, popup_area);
}

/// Render a small transient notification popup (centered near the top).
pub fn render_flash_popup(frame: &mut Frame, area: Rect, message: &str) {
    let msg_width = message.len() as u16 + 4; // 2 border + 2 padding
    let popup_w = msg_width.min(area.width);
    let popup_h = 3u16.min(area.height);
    let x = area.x + area.width.saturating_sub(popup_w) / 2;
    let y = area.y + 2.min(area.height.saturating_sub(popup_h));
    let popup_area = Rect::new(x, y, popup_w, popup_h);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green))
        .style(Style::default().bg(Color::Black));

    let text = Paragraph::new(Line::from(Span::styled(
        message,
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )))
    .block(block)
    .alignment(Alignment::Center);

    frame.render_widget(Clear, popup_area);
    frame.render_widget(text, popup_area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(0.0), "00:00");
        assert_eq!(format_time(12.5), "12:30");
        assert_eq!(format_time(23.75), "23:45");
    }

    #[test]
    fn test_ecology_line_shows_split_counters() {
        let stats = SimStats {
            creature_births: 2,
            creature_deaths: 1,
            producer_births: 7,
            producer_deaths: 3,
            adult_count: 4,
            juvenile_count: 6,
            producer_leaf_biomass: 8.0,
            producer_structural_biomass: 4.0,
            producer_belowground_reserve: 3.0,
            rolling_producer_npp: 0.42,
            ..SimStats::default()
        };
        let line = ecology_line(&stats);
        assert!(line.contains("C B/D:  2/1"));
        assert!(line.contains("Prod B/D:   7/3"));
        assert!(line.contains("A/J:  4/6"));
        assert!(line.contains("ProdBio:   15.0"));
    }

    #[test]
    fn test_diagnostics_lines_include_history_labels() {
        let diagnostics = EcologyDiagnostics {
            instant: tuiq_core::EcologyInstant {
                producer_total_biomass: 12.0,
                producer_active_biomass: 5.0,
                consumer_biomass: 2.0,
                rolling_producer_npp: 0.5,
                rolling_consumer_intake: 0.3,
                rolling_consumer_maintenance: 0.2,
                dissolved_n: 50.0,
                dissolved_p: 8.0,
                sediment_n: 180.0,
                sediment_p: 36.0,
                phytoplankton_load: 0.15,
                consumer_to_producer_biomass_ratio: 2.0 / 12.0,
                intake_to_npp_ratio: 0.3 / 0.5,
                maintenance_to_intake_ratio: 0.2 / 0.3,
                ..tuiq_core::EcologyInstant::default()
            },
            daily_history: vec![
                DailyEcologySample {
                    day: 1,
                    producer_total_biomass: 10.0,
                    consumer_biomass: 1.5,
                    rolling_producer_npp: 0.4,
                    rolling_consumer_intake: 0.2,
                    rolling_consumer_maintenance: 0.1,
                    dissolved_n: 48.0,
                    dissolved_p: 7.5,
                    phytoplankton_load: 0.12,
                    ..DailyEcologySample::default()
                },
                DailyEcologySample {
                    day: 2,
                    producer_total_biomass: 11.0,
                    consumer_biomass: 1.8,
                    rolling_producer_npp: 0.45,
                    rolling_consumer_intake: 0.25,
                    rolling_consumer_maintenance: 0.15,
                    dissolved_n: 49.0,
                    dissolved_p: 7.8,
                    phytoplankton_load: 0.13,
                    ..DailyEcologySample::default()
                },
            ],
        };

        let lines = diagnostics_lines(&diagnostics, 80);
        assert!(lines[0].contains("ProdTot:12.0"));
        assert!(lines[1].contains("Intake:0.30"));
        assert!(lines[2].contains("Phy:0.15"));
        assert!(lines[3].contains("ProdHist"));
        assert!(lines[4].contains("ConsHist"));
        assert!(lines[5].contains("BalHist"));
    }

    #[test]
    fn test_status_line_includes_time_and_speed() {
        let mut env = Environment::default();
        env.time_of_day = 8.5;
        env.temperature = 24.0;
        env.light_level = 0.75;
        let stats = SimStats {
            creature_count: 5,
            max_generation: 2,
            avg_complexity: 0.33,
            species_count: 3,
            elapsed_days: 4,
            ..SimStats::default()
        };
        let line = status_line(&stats, &env, false, 2.0, 1.0);
        assert!(line.contains("Pop:  5"));
        assert!(line.contains("Day    5 08:30 Day"));
        assert!(line.contains("2.0x"));
    }

    #[test]
    fn test_centered_rect_produces_valid_area() {
        let area = Rect::new(0, 0, 120, 40);
        let popup = centered_rect(75, 85, area);
        assert!(popup.width <= area.width);
        assert!(popup.height <= area.height);
        assert!(popup.x >= area.x);
        assert!(popup.y >= area.y);
        assert!(popup.right() <= area.right());
        assert!(popup.bottom() <= area.bottom());
    }

    #[test]
    fn test_centered_rect_small_terminal() {
        let area = Rect::new(0, 0, 30, 10);
        let popup = centered_rect(75, 85, area);
        assert!(popup.width >= 30);
        assert!(popup.height >= 10);
    }

    #[test]
    fn test_controls_line_includes_help_key() {
        let line = controls_line();
        assert!(line.contains("?:Help"));
    }
}
