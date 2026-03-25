//! HUD overlay: population stats, time, environment info, controls hint.

use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};
use tuiq_core::environment::{Environment, EventKind};
use tuiq_core::SimStats;

/// Render the HUD overlay at the bottom of the screen.
pub fn render_hud(
    frame: &mut Frame,
    area: Rect,
    stats: &SimStats,
    env: &Environment,
    paused: bool,
    speed: f32,
) {
    if area.height < 1 {
        return;
    }

    let time_str = format_time(env.time_of_day);
    let day_night = if env.is_night() { "Night" } else { "Day" };
    let temp_str = format!("{:.0}°C", env.temperature);
    let light_str = format!("{:.0}%", env.light_level * 100.0);

    let event_str = match &env.active_event {
        Some(e) => match e.kind {
            EventKind::AlgaeBloom => format!(" | ALGAE BLOOM {:.0}s", e.remaining),
            EventKind::FeedingFrenzy => format!(" | FEEDING FRENZY {:.0}s", e.remaining),
            EventKind::ColdSnap => format!(" | COLD SNAP {:.0}s", e.remaining),
            EventKind::Earthquake => format!(" | EARTHQUAKE {:.0}s", e.remaining),
        },
        None => String::new(),
    };

    let pause_str = if paused { " [PAUSED]" } else { "" };
    let speed_str = format!("{:.1}x", speed);

    let status_line = format!(
        " Pop: {} | Gen: {} | Cx: {:.2} | Sp: {} | Born: {} Died: {} | Day {} {} {} | {}  {} | {}{}{} ",
        stats.creature_count,
        stats.max_generation,
        stats.avg_complexity,
        stats.species_count,
        stats.births,
        stats.deaths,
        stats.elapsed_days + 1,
        time_str,
        day_night,
        temp_str,
        light_str,
        speed_str,
        pause_str,
        event_str,
    );

    let controls = " q:Quit  Space:Pause  +/-:Speed  f:Feed ";

    // Status line at bottom-1
    let status_y = area.y + area.height.saturating_sub(2);
    let status_area = Rect::new(area.x, status_y, area.width, 1);

    let status = Paragraph::new(Line::from(vec![
        Span::styled(status_line, Style::default().fg(Color::White)),
    ]))
    .alignment(Alignment::Left);

    frame.render_widget(status, status_area);

    // Controls hint at very bottom
    let controls_y = area.y + area.height.saturating_sub(1);
    let controls_area = Rect::new(area.x, controls_y, area.width, 1);

    let controls_widget = Paragraph::new(Line::from(vec![
        Span::styled(
            controls,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ),
    ]))
    .alignment(Alignment::Center);

    frame.render_widget(controls_widget, controls_area);
}

/// Format time_of_day (0-24 float) as HH:MM string.
fn format_time(hours: f32) -> String {
    let h = hours as u32 % 24;
    let m = ((hours - hours.floor()) * 60.0) as u32;
    format!("{:02}:{:02}", h, m)
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
}
