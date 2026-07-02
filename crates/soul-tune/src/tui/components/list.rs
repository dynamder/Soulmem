use ratatui::layout::Rect;
use ratatui::style::{Color, Style, Stylize};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

pub fn render_list(
    frame: &mut Frame,
    area: Rect,
    items: &[String],
    selected: usize,
    scroll: usize,
) {
    let visible_items: Vec<&String> = items.iter().skip(scroll).collect();
    for (i, item) in visible_items.iter().enumerate() {
        let y = area.y + i as u16;
        if y >= area.y + area.height {
            break;
        }
        let actual_idx = scroll + i;
        let line = if actual_idx == selected {
            format!("▸ {}", item)
        } else {
            format!("  {}", item)
        };
        let style = if actual_idx == selected {
            Style::default().fg(Color::Yellow).bg(Color::DarkGray)
        } else {
            Style::default()
        };
        frame.render_widget(
            Paragraph::new(line).style(style),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}
