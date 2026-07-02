use ratatui::layout::Rect;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

pub fn render_status_bar(frame: &mut Frame, area: Rect, hints: &[(String, String)]) {
    let text: String = hints
        .iter()
        .map(|(key, desc)| format!(" {} {} ", key, desc))
        .collect::<Vec<_>>()
        .join("  ");

    frame.render_widget(Paragraph::new(text).fg(Color::DarkGray), area);
}
