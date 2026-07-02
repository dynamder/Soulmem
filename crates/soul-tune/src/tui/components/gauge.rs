use ratatui::layout::Rect;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge};
#![allow(dead_code)]

use ratatui::Frame;

pub fn render_gauge(frame: &mut Frame, area: Rect, title: &str, current: usize, total: usize) {
    let ratio = if total > 0 {
        current as f64 / total as f64
    } else {
        0.0
    };
    let block = Block::bordered()
        .title(format!(
            " {}: {}/{} ({:.0}%) ",
            title,
            current,
            total,
            ratio * 100.0
        ))
        .fg(Color::Cyan);
    let gauge = Gauge::default().block(block).ratio(ratio).fg(Color::Cyan);
    frame.render_widget(gauge, area);
}
