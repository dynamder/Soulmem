use ratatui::layout::Rect;
use ratatui::prelude::{Stylize, Widget};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use super::scroll::ScrollState;

/// Render a scrollable paragraph from a list of lines.
/// Uses Paragraph::scroll() with the scroll state as (offset, 0).
pub fn render_scrollable_paragraph(
    frame: &mut Frame,
    area: Rect,
    lines: &[Line],
    scroll: &ScrollState,
    block_title: &str,
    block_fg: ratatui::style::Color,
) {
    let block = Block::bordered().title(block_title).fg(block_fg);
    let inner = block.inner(area);
    block.render(area, frame.buffer_mut());
    frame.render_widget(
        Paragraph::new(Text::from(lines.to_vec()))
            .wrap(Wrap { trim: false })
            .scroll((scroll.offset as u16, 0)),
        inner,
    );
}

/// Render a scrollable paragraph without a border block.
pub fn render_scrollable_paragraph_raw(
    frame: &mut Frame,
    area: Rect,
    lines: &[Line],
    scroll: &ScrollState,
) {
    frame.render_widget(
        Paragraph::new(Text::from(lines.to_vec()))
            .wrap(Wrap { trim: false })
            .scroll((scroll.offset as u16, 0)),
        area,
    );
}
