use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use super::scroll::ScrollState;

/// A generic scrollable table widget.
/// Uses ratatui Constraint system for column widths.
pub struct TableWidget<'a, T> {
    pub items: &'a [T],
    pub scroll: &'a ScrollState,
    pub columns: &'a [Constraint],
    pub headers: &'a [&'a str],
    pub highlight_style: Style,
    pub default_style: Style,
    pub row_renderer: Box<dyn Fn(&T, &[Rect], &mut Frame, bool)>,
}

impl<'a, T> TableWidget<'a, T> {
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered();
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let col_count = self.columns.len();
        let header_rects = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(self.columns)
            .split(Rect::new(inner.x, inner.y, inner.width, 1));

        for (i, hdr) in self.headers.iter().enumerate() {
            if i >= col_count {
                break;
            }
            frame.render_widget(Paragraph::new(*hdr).bold().fg(Color::Cyan), header_rects[i]);
        }

        let data_rows = self
            .items
            .len()
            .min(inner.height.saturating_sub(1) as usize);
        let row_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1); data_rows])
            .split(Rect::new(
                inner.x,
                inner.y + 1,
                inner.width,
                inner.height.saturating_sub(1),
            ));

        for (disp_i, row_rect) in row_rects.iter().enumerate() {
            let actual_idx = self.scroll.offset + disp_i;
            if actual_idx >= self.items.len() {
                break;
            }
            let is_cursor = actual_idx == self.scroll.cursor;
            let item = &self.items[actual_idx];

            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(self.columns)
                .split(*row_rect);

            (self.row_renderer)(item, &cols, frame, is_cursor);
        }
    }
}
