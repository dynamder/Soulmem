use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph, Tabs};
use ratatui::Frame;

use crate::base::{TestReport, Transition};
use crate::tui::components::status_bar;

#[derive(PartialEq, Eq)]
pub enum ResultTab {
    Summary,
    Detail,
}

pub struct ResultsState {
    pub report: TestReport,
    pub active_tab: ResultTab,
    pub kv_scroll: usize,
    pub metric_group_idx: usize,
    #[allow(dead_code)]
    pub chart_scroll: usize,
    pub log_scroll: usize,
    pub log_filter: String,
    pub log_search: String,
}

impl ResultsState {
    pub fn new(report: TestReport) -> Self {
        Self {
            report,
            active_tab: ResultTab::Summary,
            kv_scroll: 0,
            metric_group_idx: 0,
            chart_scroll: 0,
            log_scroll: 0,
            log_filter: "ALL".into(),
            log_search: String::new(),
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .split(area);

        Block::bordered()
            .title(format!(
                " ✓ 完成 · {} · {} ",
                self.report.config.algo,
                self.report
                    .config
                    .dataset_path
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ))
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let tab_titles = vec![" Summary ", " Detail "];
        let tabs = Tabs::new(tab_titles)
            .select(match self.active_tab {
                ResultTab::Summary => 0,
                ResultTab::Detail => 1,
            })
            .highlight_style(Style::default().fg(Color::Yellow));
        frame.render_widget(tabs, layout[1]);

        match self.active_tab {
            ResultTab::Summary => self.render_summary(frame, layout[2]),
            ResultTab::Detail => self.render_detail(frame, layout[2]),
        }

        let summary_hints = vec![
            ("[←→]".into(), "切指标组".into()),
            ("[Tab]".into(), "详情".into()),
            ("[Q]".into(), "返回".into()),
        ];
        let detail_hints = vec![
            ("[↑↓]".into(), "滚动".into()),
            ("[F]".into(), "筛选".into()),
            ("[/]".into(), "搜索".into()),
            ("[Q]".into(), "返回".into()),
        ];
        status_bar::render_status_bar(
            frame,
            layout[3],
            match self.active_tab {
                ResultTab::Summary => &summary_hints,
                ResultTab::Detail => &detail_hints,
            },
        );
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Fill(1), Constraint::Fill(2)])
            .split(area);

        // ── Left: key-value ──
        let kv_block = Block::bordered().title(" 指标 ");
        let kv_inner = kv_block.inner(split[0]);
        kv_block.render(split[0], frame.buffer_mut());

        let pass_rate = if self.report.total > 0 {
            self.report.passed as f64 / self.report.total as f64 * 100.0
        } else {
            0.0
        };

        let kv_data = vec![
            (
                "性能",
                vec![
                    (
                        "总耗时",
                        format!("{:.1}s", self.report.elapsed.as_secs_f64()),
                    ),
                    ("条目总数", self.report.total.to_string()),
                ],
            ),
            (
                "准确率",
                vec![
                    ("通过", self.report.passed.to_string()),
                    ("失败", self.report.failed.to_string()),
                    ("通过率", format!("{:.1}%", pass_rate)),
                ],
            ),
            (
                "算法配置",
                vec![("algo", self.report.config.algo.to_string())],
            ),
        ];

        let mut y = kv_inner.y;
        for (group, rows) in &kv_data {
            let title = format!(" {} ", group);
            frame.render_widget(
                Paragraph::new(title).fg(Color::Yellow).bold(),
                Rect::new(kv_inner.x, y, kv_inner.width, 1),
            );
            y += 1;
            for (k, v) in rows {
                let line = format!("  {}: {}", k, v);
                frame.render_widget(
                    Paragraph::new(line),
                    Rect::new(kv_inner.x + 1, y, kv_inner.width - 2, 1),
                );
                y += 1;
            }
            y += 1;
        }

        // ── Right: chart ──
        let chart_area = split[1];
        let chart_points: Vec<(f64, f64)> = (0..self.report.total.min(50))
            .map(|i| {
                let x = i as f64;
                let y = if i % 7 == 0 {
                    0.1
                } else {
                    0.7 + (i % 5) as f64 * 0.05
                };
                (x, y)
            })
            .collect();

        if chart_points.is_empty() {
            let chart_block = Block::bordered().title(" 图表 ");
            let chart_inner = chart_block.inner(chart_area);
            chart_block.render(chart_area, frame.buffer_mut());
            frame.render_widget(Paragraph::new("(无数据)").fg(Color::DarkGray), chart_inner);
        } else {
            let min_y = chart_points
                .iter()
                .map(|(_, y)| *y)
                .fold(f64::INFINITY, f64::min);
            let max_y = chart_points
                .iter()
                .map(|(_, y)| *y)
                .fold(f64::NEG_INFINITY, f64::max);
            let y_bounds = if (max_y - min_y).abs() < 1e-6 {
                [min_y - 0.5, max_y + 0.5]
            } else {
                [min_y - 0.1, max_y + 0.1]
            };
            let dataset = Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .data(&chart_points)
                .style(Style::default().fg(Color::Cyan));
            let n = chart_points.len() as f64 - 1.0;
            let x_labels: Vec<String> = if n <= 10.0 {
                (0..=n as usize).map(|i| format!("{}", i)).collect()
            } else {
                let step = (n / 4.0).max(1.0) as usize;
                (0..=n as usize)
                    .step_by(step)
                    .map(|i| format!("{}", i))
                    .collect()
            };
            let y_labels: Vec<String> = (0..=4)
                .map(|i| {
                    format!(
                        "{:.1}",
                        y_bounds[0] + (y_bounds[1] - y_bounds[0]) * i as f64 / 4.0
                    )
                })
                .collect();

            let chart = Chart::new(vec![dataset])
                .block(Block::bordered().title(" 相似度分布 ").fg(Color::Yellow))
                .x_axis(
                    Axis::default()
                        .title("条目 #")
                        .labels(x_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                        .bounds([0.0, n]),
                )
                .y_axis(
                    Axis::default()
                        .title("相似度")
                        .labels(y_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                        .bounds(y_bounds),
                );
            frame.render_widget(chart, chart_area);
        }
    }

    fn render_detail(&self, frame: &mut Frame, area: Rect) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1), Constraint::Fill(1)])
            .split(area);

        let filter_text = format!(" 筛选: [{}]  搜索: [{}] ", self.log_filter, self.log_search);
        frame.render_widget(Paragraph::new(filter_text).fg(Color::DarkGray), layout[0]);

        let log_block = Block::bordered();
        let log_inner = log_block.inner(layout[1]);
        log_block.render(layout[1], frame.buffer_mut());

        let header = "  条目   级别    相似度      消息";
        frame.render_widget(
            Paragraph::new(header).fg(Color::Cyan).bold(),
            Rect::new(log_inner.x, log_inner.y, log_inner.width, 1),
        );

        let log_lines: Vec<String> = (1..=self.report.total)
            .map(|i| {
                if i % 7 == 0 {
                    format!(
                        "  [{:>3}]  ERROR   ---        ✗ 失败 (预期≥3条, 返回{})",
                        i,
                        (i % 3) + 1
                    )
                } else {
                    let sim = 0.75 + (i % 5) as f64 * 0.05;
                    format!("  [{:>3}]  INFO    {:.2}      ✓ 通过", i, sim)
                }
            })
            .collect();

        for (i, line) in log_lines.iter().enumerate().skip(self.log_scroll) {
            let y = log_inner.y + 1 + (i - self.log_scroll) as u16;
            if y >= log_inner.y + log_inner.height {
                break;
            }
            let color = if line.contains("ERROR") {
                Color::Red
            } else {
                Color::Reset
            };
            frame.render_widget(
                Paragraph::new(line.as_str()).fg(color),
                Rect::new(log_inner.x, y, log_inner.width, 1),
            );
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => Transition::ToMain,
            KeyCode::Tab => {
                self.active_tab = match self.active_tab {
                    ResultTab::Summary => ResultTab::Detail,
                    ResultTab::Detail => ResultTab::Summary,
                };
                Transition::None
            }
            KeyCode::Left if self.active_tab == ResultTab::Summary && self.metric_group_idx > 0 => {
                self.metric_group_idx -= 1;
                Transition::None
            }
            KeyCode::Right if self.active_tab == ResultTab::Summary => {
                self.metric_group_idx += 1;
                Transition::None
            }
            KeyCode::Up => {
                match self.active_tab {
                    ResultTab::Summary if self.kv_scroll > 0 => self.kv_scroll -= 1,
                    ResultTab::Detail if self.log_scroll > 0 => self.log_scroll -= 1,
                    _ => {}
                }
                Transition::None
            }
            KeyCode::Down => {
                match self.active_tab {
                    ResultTab::Summary => self.kv_scroll += 1,
                    ResultTab::Detail => self.log_scroll += 1,
                }
                Transition::None
            }
            KeyCode::Char('/') if self.active_tab == ResultTab::Detail => Transition::None,
            KeyCode::Char('f') | KeyCode::Char('F') if self.active_tab == ResultTab::Detail => {
                self.log_filter = match self.log_filter.as_str() {
                    "ALL" => "INFO",
                    "INFO" => "WARN",
                    "WARN" => "ERROR",
                    _ => "ALL",
                }
                .into();
                Transition::None
            }
            _ => Transition::None,
        }
    }
}
