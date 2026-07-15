use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Tabs;
use ratatui::Frame;

pub fn render_tabs(frame: &mut Frame, area: Rect, titles: &[&str], active: usize) {
    let tabs = Tabs::new(titles.to_vec())
        .select(active)
        .highlight_style(Style::default().fg(Color::Yellow));
    frame.render_widget(tabs, area);
}
