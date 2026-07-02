use ratatui::layout::Rect;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::Paragraph;
#![allow(dead_code)]

use ratatui::Frame;

pub struct KvRow {
    pub key: String,
    pub value: String,
}

pub struct KvGroup {
    pub title: String,
    pub rows: Vec<KvRow>,
}

pub fn render_kv_table(frame: &mut Frame, area: Rect, groups: &[KvGroup], scroll: &mut usize) {
    let mut y = area.y;

    for group in groups {
        if y > area.y + area.height {
            break;
        }
        let title = format!(" {} ", group.title);
        frame.render_widget(
            Paragraph::new(title).fg(Color::Yellow).bold(),
            Rect::new(area.x, y, area.width, 1),
        );
        y += 1;

        for row in &group.rows {
            if y > area.y + area.height {
                break;
            }
            if *scroll > 0 {
                *scroll -= 1;
                continue;
            }
            let line = format!("  {}: {}", row.key, row.value);
            frame.render_widget(
                Paragraph::new(line),
                Rect::new(area.x + 1, y, area.width - 2, 1),
            );
            y += 1;
        }
        y += 1;
    }
}
