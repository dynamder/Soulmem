use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use std::fmt::Display;

use super::scroll::ScrollState;

/// Render a scrollable list of items with cursor highlight.
/// Uses ratatui Constraint system for layout.
pub fn render_list<T: Display>(
    frame: &mut Frame,
    area: Rect,
    items: &[T],
    scroll: &ScrollState,
    highlight_prefix: &str,
    highlight_style: Style,
) {
    let n = items.len().min(area.height as usize);
    let line_rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Length(1); n])
        .split(Rect::new(area.x, area.y, area.width, n as u16));

    for (i, line_rect) in line_rects.iter().enumerate() {
        let actual_idx = scroll.offset + i;
        if actual_idx >= items.len() {
            break;
        }
        let is_cursor = actual_idx == scroll.cursor;
        let prefix = if is_cursor { highlight_prefix } else { "  " };
        let style = if is_cursor {
            highlight_style
        } else {
            Style::default()
        };
        frame.render_widget(
            Paragraph::new(format!("{}{}", prefix, items[actual_idx])).style(style),
            *line_rect,
        );
    }
}

/// Convenience: render a scrollable list with default styling (yellow highlight, "▸ " prefix).
pub fn render_simple_list<T: Display>(
    frame: &mut Frame,
    area: Rect,
    items: &[T],
    scroll: &ScrollState,
) {
    render_list(
        frame,
        area,
        items,
        scroll,
        "▸ ",
        Style::default().fg(Color::Yellow).bg(Color::DarkGray),
    );
}
