use ratatui::layout::Rect;
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::state::params::ParamRow;

pub fn render_editable_table(
    frame: &mut Frame,
    area: Rect,
    rows: &[ParamRow],
    textareas: &[TextArea],
    selected: usize,
    editing: Option<usize>,
    _scroll: usize,
) {
    let block = Block::bordered();
    let inner = block.inner(area);
    block.render(area, frame.buffer_mut());

    // Header
    let header = "  参数名              当前值              描述";
    let header_style = Style::default().fg(Color::Cyan).bold();
    frame.render_widget(
        Paragraph::new(header).style(header_style),
        Rect::new(inner.x, inner.y, inner.width, 1),
    );

    let line_style = Style::default().fg(Color::DarkGray);
    frame.render_widget(
        Paragraph::new("  ").style(line_style),
        Rect::new(inner.x, inner.y + 1, inner.width, 1),
    );

    for (i, row) in rows.iter().enumerate() {
        let y = inner.y + 2 + i as u16;
        if y >= inner.y + inner.height {
            break;
        }

        let is_selected = i == selected;
        let is_editing = editing == Some(i);

        let prefix = if is_selected { "▶ " } else { "  " };
        let value_display = if is_editing { "[EDITING]" } else { &row.value };

        let line = format!(
            "{}{:<20} {:<18} {}",
            prefix, row.name, value_display, row.description
        );
        let style = if is_selected {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(line).style(style),
            Rect::new(inner.x, y, inner.width, 1),
        );

        if is_editing {
            if let Some(ta) = textareas.get(i) {
                let edit_area =
                    Rect::new(inner.x + 23, y, inner.width.saturating_sub(24).min(18), 1);
                frame.render_widget(ta, edit_area);
            }
        }
    }
}
