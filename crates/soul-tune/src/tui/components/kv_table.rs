use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::scroll::ScrollState;

pub struct KvRow {
    pub key: String,
    pub value: String,
}

pub struct KvGroup {
    pub title: String,
    pub rows: Vec<KvRow>,
}

/// Render grouped key-value table with ScrollState.
/// Uses ratatui Constraint system for layout.
pub fn render_kv_table(frame: &mut Frame, area: Rect, groups: &[KvGroup], scroll: &ScrollState) {
    // Build flat display text lines first
    let mut lines: Vec<String> = Vec::new();
    for group in groups {
        lines.push(format!(" {} ", group.title));
        for row in &group.rows {
            lines.push(format!("  {}: {}", row.key, row.value));
        }
        lines.push(String::new());
    }

    let visible = area.height as usize;
    let end = (scroll.offset + visible).min(lines.len());
    for (i, y_offset) in (scroll.offset..end).enumerate() {
        if i >= visible {
            break;
        }
        let y = area.y + i as u16;
        let line = &lines[y_offset];
        let is_title = line.starts_with(' ') && !line.starts_with("  ");
        let style = if is_title {
            ratatui::style::Style::default().fg(Color::Yellow).bold()
        } else {
            ratatui::style::Style::default()
        };
        frame.render_widget(
            Paragraph::new(line.as_str()).style(style),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}
