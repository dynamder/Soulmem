//! 遗忘测试观测状态页 —— 集成进主 TUI，供人类测试员逐节点观测遗忘结果。
//!
//! 入口：主应用命令面板 `test forget`（或 `forget`）→ 选图 → 跑完全管线 →
//! `app.rs` 在 `ToTestResults` 时按算法分派到本页（而非通用 Results 页）。
//!
//! 界面：
//! - 顶部：用例切换（←/→），显示通过率
//! - 左栏：节点列表（类型 / id / 缺失度 / 动作，按动作着色）
//! - 右栏：选中节点的详情 —— **图原文** / 缺失度前后 / 遮罩输入 / LLM 原始回复
//! - 底部：按键提示
//!
//! 按键：↑/↓ 选择节点，PgUp/PgDn 滚动详情，←/→ 切换用例，q/ESC 返回主菜单。

use ratatui::crossterm::event::{KeyCode, KeyEvent, MouseEvent, MouseEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::prelude::{Color, Line, Modifier, Style, Stylize};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use crate::base::{TestReport, Transition};
use crate::component::{Component, ComponentEvent};
use crate::engine::forget::{ForgetCaseData, NodeForgetStat};
use crate::engine::suite::TestCaseOutcome;

pub struct ForgetObserverState {
    report: TestReport,
    case_idx: usize,
    node_idx: usize,
    detail_scroll: u16,
}

impl ForgetObserverState {
    pub fn new(report: TestReport) -> Self {
        Self {
            report,
            case_idx: 0,
            node_idx: 0,
            detail_scroll: 0,
        }
    }

    fn case_name(&self) -> String {
        self.outcomes()
            .get(self.case_idx)
            .map(|o| o.case_name.clone())
            .unwrap_or_default()
    }

    fn outcomes(&self) -> &[TestCaseOutcome] {
        &self.report.suite_report.outcomes
    }

    fn case_data(&self) -> Option<&ForgetCaseData> {
        self.outcomes()
            .get(self.case_idx)
            .and_then(|o| o.data.downcast_ref::<ForgetCaseData>())
    }

    fn nodes(&self) -> Vec<&NodeForgetStat> {
        self.case_data()
            .map(|d| d.nodes.iter().collect())
            .unwrap_or_default()
    }
}

/// 按动作着色：NoAction 灰 / MaskOnly 黄 / Revised 绿(有效) 红(无效)
fn action_style(action: &str, effective: bool) -> Style {
    match action {
        "Revised" => {
            if effective {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            }
        }
        "MaskOnly" => Style::default().fg(Color::Yellow),
        _ => Style::default().fg(Color::Gray),
    }
}

fn node_item_line(node: &NodeForgetStat) -> Line<'static> {
    let short_id: String = node.id.chars().take(8).collect();
    let eff = if node.action == "Revised" {
        if node.effective {
            " [有效]"
        } else {
            " [无效!]"
        }
    } else {
        ""
    };
    Line::from(format!(
        "{} [{}] md {:.2}→{:.2} {}{}",
        node.type_name, short_id, node.md_before, node.md_after, node.action, eff
    ))
    .style(action_style(node.action, node.effective))
}

/// 按宽度折行（中文按字符计）
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for ch in text.chars() {
        if line.chars().count() >= width {
            out.push(line.clone());
            line.clear();
        }
        line.push(ch);
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// 详情面板：图原文 / 缺失度 / 遮罩输入 / LLM 原始回复
fn detail_lines(node: &NodeForgetStat) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let short_id: String = node.id.chars().take(8).collect();

    lines.push(
        Line::from(format!(
            "{} [{}]  md {:.3} → {:.3}",
            node.type_name, short_id, node.md_before, node.md_after
        ))
        .style(Style::default().add_modifier(Modifier::BOLD)),
    );

    if !node.original.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from("── 图原文 ──").style(Style::default().fg(Color::Cyan)));
        for l in wrap_text(&node.original, 80) {
            lines.push(Line::from(l));
        }
    }
    if let Some((m, t)) = node.mask {
        if t > 0 {
            lines.push(Line::from(""));
            lines.push(
                Line::from(format!(
                    "── 遮罩 {}/{} = {:.0}% ──",
                    m,
                    t,
                    m as f32 / t as f32 * 100.0
                ))
                .style(Style::default().fg(Color::Yellow)),
            );
        }
    }
    if let Some(mt) = &node.masked_text {
        lines.push(Line::from(""));
        lines.push(Line::from("── 遮罩输入 ──").style(Style::default().fg(Color::Yellow)));
        for l in wrap_text(mt, 80) {
            lines.push(Line::from(l));
        }
    }
    if let Some(reply) = &node.llm_reply {
        lines.push(Line::from(""));
        lines.push(
            Line::from("── LLM 原始回复 ──").style(Style::default().fg(if node.effective {
                Color::Green
            } else {
                Color::Red
            })),
        );
        for l in wrap_text(reply, 80) {
            lines.push(Line::from(l));
        }
    }
    lines
}

impl Component for ForgetObserverState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        let key = match event {
            ComponentEvent::Key(key) => key,
            ComponentEvent::Mouse(_) | ComponentEvent::Tick => return Transition::None,
        };
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => Transition::ToMain,
            KeyCode::Left | KeyCode::Char('h') => {
                if !self.outcomes().is_empty() {
                    self.case_idx = (self.case_idx + self.outcomes().len() - 1) % self.outcomes().len();
                }
                self.node_idx = 0;
                self.detail_scroll = 0;
                Transition::None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if !self.outcomes().is_empty() {
                    self.case_idx = (self.case_idx + 1) % self.outcomes().len();
                }
                self.node_idx = 0;
                self.detail_scroll = 0;
                Transition::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.node_idx > 0 {
                    self.node_idx -= 1;
                    self.detail_scroll = 0;
                }
                Transition::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let n = self.nodes().len();
                if n > 0 && self.node_idx + 1 < n {
                    self.node_idx += 1;
                    self.detail_scroll = 0;
                }
                Transition::None
            }
            KeyCode::PageDown => {
                self.detail_scroll = self.detail_scroll.saturating_add(5);
                Transition::None
            }
            KeyCode::PageUp => {
                self.detail_scroll = self.detail_scroll.saturating_sub(5);
                Transition::None
            }
            _ => Transition::None,
        }
    }

    fn view(&self, frame: &mut Frame) {
        let passed = self.report.passed;
        let total = self.report.total;

        // ── 布局 ──
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(frame.area());

        // ── 顶部：用例 tabs + 通过率 ──
        let titles: Vec<Line> = self
            .outcomes()
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let name = o
                    .case_name
                    .strip_prefix("forget/full/")
                    .unwrap_or(&o.case_name)
                    .to_string();
                let marker = if o.passed { "●" } else { "✗" };
                Line::from(format!(" {} {} ", marker, name))
            })
            .collect();
        let title_block = Block::default().borders(Borders::ALL).title(format!(
            " 遗忘测试观测 | 通过 {}/{} ({:.0}%) ",
            passed,
            total,
            if total > 0 {
                passed as f64 / total as f64 * 100.0
            } else {
                0.0
            }
        ));
        frame.render_widget(
            Tabs::new(titles)
                .select(self.case_idx)
                .block(title_block)
                .style(Style::default().fg(Color::Gray))
                .highlight_style(
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            chunks[0],
        );

        // ── 中部：左节点列表 | 右详情 ──
        let mid = Layout::horizontal([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(chunks[1]);

        let nodes = self.nodes();

        // 左栏：节点列表
        let list_items: Vec<ListItem> = nodes.iter().map(|n| ListItem::new(node_item_line(n))).collect();
        let mut list_state = ListState::default();
        if !nodes.is_empty() {
            list_state.select(Some(self.node_idx.min(nodes.len() - 1)));
        }
        let list = List::new(list_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(
                        " 节点 ({}/{}) ",
                        if nodes.is_empty() { 0 } else { self.node_idx + 1 },
                        nodes.len()
                    )),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            );
        frame.render_stateful_widget(list, mid[0], &mut list_state);

        // 右栏：详情
        let detail_block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" 详情: {} ", self.case_name()));
        if let Some(node) = nodes.get(self.node_idx.min(nodes.len().saturating_sub(1))) {
            frame.render_widget(
                Paragraph::new(detail_lines(node))
                    .block(detail_block)
                    .wrap(Wrap { trim: false })
                    .scroll((self.detail_scroll, 0)),
                mid[1],
            );
        } else if let Some(d) = self.case_data() {
            // activation / incremental 等无节点列表的用例：展示指标 + 明细文本
            let mut lines: Vec<Line> = Vec::new();
            for (group, label, value) in &d.metrics {
                lines.push(Line::from(format!("{} | {}: {}", group, label, value)));
            }
            lines.push(Line::from(""));
            for dl in &d.detail_lines {
                lines.push(Line::from(dl.as_str()));
            }
            frame.render_widget(
                Paragraph::new(lines)
                    .block(detail_block)
                    .wrap(Wrap { trim: false })
                    .scroll((self.detail_scroll, 0)),
                mid[1],
            );
        } else {
            frame.render_widget(detail_block, mid[1]);
        }

        // ── 底部：按键提示 ──
        let hint = " ↑/↓ 节点  ←/→ 用例  PgUp/PgDn 详情滚动  q/ESC 返回 ";
        frame.render_widget(
            Paragraph::new(Line::from(hint).style(Style::default().fg(Color::DarkGray))),
            chunks[2],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{AlgoType, ForgetMode, TestConfig};
    use crate::engine::suite::SuiteReport;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Duration;

    fn dummy_report() -> TestReport {
        TestReport {
            config: TestConfig {
                algo: AlgoType::Forget(ForgetMode::Pipeline),
                dataset_path: PathBuf::new(),
                params: HashMap::new(),
            },
            total: 0,
            passed: 0,
            failed: 0,
            elapsed: Duration::ZERO,
            suite_report: SuiteReport {
                metrics: vec![],
                detail_header: String::new(),
                detail_rows: vec![],
                outcomes: vec![],
            },
            error: None,
        }
    }

    #[test]
    fn test_wrap_text_chinese() {
        let wrapped = wrap_text("一二三四五六七八九十", 4);
        assert_eq!(wrapped, vec!["一二三四", "五六七八", "九十"]);
    }

    #[test]
    fn test_action_style() {
        assert_eq!(action_style("Revised", true), Style::default().fg(Color::Green));
        assert_eq!(action_style("Revised", false), Style::default().fg(Color::Red));
        assert_eq!(action_style("MaskOnly", false), Style::default().fg(Color::Yellow));
    }

    #[test]
    fn test_esc_returns_to_main() {
        let mut state = ForgetObserverState::new(dummy_report());
        let t = state.handle_event(ComponentEvent::Key(KeyEvent::from(KeyCode::Esc)));
        assert!(matches!(t, Transition::ToMain));
    }

    #[test]
    fn test_mouse_ignored() {
        let mut state = ForgetObserverState::new(dummy_report());
        let t = state.handle_event(ComponentEvent::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 0,
            row: 0,
            modifiers: ratatui::crossterm::event::KeyModifiers::NONE,
        }));
        assert!(matches!(t, Transition::None));
    }
}
