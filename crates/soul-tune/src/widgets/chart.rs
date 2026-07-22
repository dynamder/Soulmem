#![allow(dead_code)]

use ratatui::layout::Rect;
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType};
use ratatui::Frame;

pub fn render_chart(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    data: Vec<(f64, f64)>,
    x_label: &str,
    y_label: &str,
) {
    if data.is_empty() {
        let block = Block::bordered()
            .title(format!(" {}", title))
            .fg(Color::Yellow);
        block.render(area, frame.buffer_mut());
        return;
    }

    let min_x = data.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let max_x = data
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let max_y = data
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);

    let x_bounds = if (max_x - min_x).abs() < 1e-6 {
        [min_x - 1.0, min_x + 1.0]
    } else {
        [
            min_x - (max_x - min_x) * 0.05,
            max_x + (max_x - min_x) * 0.05,
        ]
    };
    let y_bounds = if (max_y - min_y).abs() < 1e-6 {
        [min_y - 1.0, min_y + 1.0]
    } else {
        [
            min_y - (max_y - min_y) * 0.05,
            max_y + (max_y - min_y) * 0.05,
        ]
    };

    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .data(&data)
        .style(Style::default().fg(Color::Cyan));

    let chart = Chart::new(vec![dataset])
        .block(
            Block::bordered()
                .title(format!(" {} ", title))
                .fg(Color::Yellow),
        )
        .x_axis(Axis::default().title(x_label).bounds(x_bounds))
        .y_axis(Axis::default().title(y_label).bounds(y_bounds));

    frame.render_widget(chart, area);
}
