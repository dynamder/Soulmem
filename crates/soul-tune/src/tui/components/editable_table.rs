use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use super::scroll::ScrollState;
use crate::state::params::ParamRow;

pub fn render_editable_table(
    frame: &mut Frame,
    area: Rect,
    rows: &[ParamRow],
    textareas: &[TextArea],
    selected: usize,
    editing: Option<usize>,
    scroll: &ScrollState,
) {
    let block = Block::bordered();
    let inner = block.inner(area);
    block.render(area, frame.buffer_mut());

    let header_hints: [Constraint; 6] = [
        Constraint::Length(1),
        Constraint::Length(20),
        Constraint::Length(1),
        Constraint::Length(18),
        Constraint::Length(1),
        Constraint::Fill(1),
    ];

    // Header row
    let h_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(header_hints)
        .split(Rect::new(inner.x, inner.y, inner.width, 1));
    frame.render_widget(Paragraph::new("参数名").bold().fg(Color::Cyan), h_row[1]);
    frame.render_widget(Paragraph::new("当前值").bold().fg(Color::Cyan), h_row[3]);
    frame.render_widget(Paragraph::new("描述").bold().fg(Color::Cyan), h_row[5]);

    // Data rows
    let visible = inner.height.saturating_sub(1) as usize;
    let end = (scroll.offset + visible).min(rows.len());
    for (disp_i, actual_idx) in (scroll.offset..end).enumerate() {
        let row = &rows[actual_idx];
        let y = inner.y + 1 + disp_i as u16;
        if y >= inner.y + inner.height {
            break;
        }

        let is_selected = actual_idx == selected;
        let is_editing = editing == Some(actual_idx);
        let bg_style = if is_selected {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        };

        let row_rect = Rect::new(inner.x, y, inner.width, 1);
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(header_hints)
            .split(row_rect);

        // Cursor marker
        frame.render_widget(
            Paragraph::new(if is_selected { "▶" } else { "" }).style(bg_style),
            cols[0],
        );
        // Name
        frame.render_widget(Paragraph::new(row.name.as_str()).style(bg_style), cols[1]);
        // Spacer
        frame.render_widget(Paragraph::new(" ").style(bg_style), cols[2]);
        // Value (or editing indicator)
        if is_editing {
            frame.render_widget(
                Paragraph::new("[EDITING]").style(bg_style.fg(Color::Yellow)),
                cols[3],
            );
            if let Some(ta) = textareas.get(actual_idx) {
                frame.render_widget(ta, cols[3]);
            }
        } else {
            frame.render_widget(Paragraph::new(row.value.as_str()).style(bg_style), cols[3]);
        }
        // Spacer
        frame.render_widget(Paragraph::new(" ").style(bg_style), cols[4]);
        // Description
        frame.render_widget(
            Paragraph::new(row.description.as_str()).style(bg_style),
            cols[5],
        );
    }
}
