use ratatui::layout::Rect;
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

pub fn render_command_input(frame: &mut Frame, area: Rect, textarea: &TextArea) {
    let block = Block::bordered().title(" 命令输入 ").fg(Color::Green);
    let inner = block.inner(area);
    block.render(area, frame.buffer_mut());

    let prefix = ":";
    let prefix_width = prefix.len() as u16;
    let input_area = Rect::new(
        inner.x + prefix_width,
        inner.y,
        inner.width.saturating_sub(prefix_width),
        inner.height,
    );

    frame.render_widget(
        Paragraph::new(prefix).fg(Color::Yellow).bold(),
        Rect::new(inner.x, inner.y, prefix_width, inner.height),
    );

    frame.render_widget(textarea, input_area);
}
