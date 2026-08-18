//! 遗忘测试观测 TUI —— 供人类测试员逐节点观测遗忘结果。
//!
//! 用法：`soul-tune forget-tui <graph.json>`
//!
//! 加载指定 fixture 图（自动启用 llama-server，环境变量约定同 playtest），
//! 运行全管线全部用例后进入观测界面：
//!
//! - 顶部：用例切换（←/→），显示通过率
//! - 左栏：节点列表（类型 / id / 缺失度 / 动作，按动作着色）
//! - 右栏：选中节点的详情 —— **图原文** / 缺失度前后 / 遮罩输入 / LLM 原始回复
//! - 底部：按键提示
//!
//! 按键：↑/↓ 选择节点，PgUp/PgDn 滚动详情，←/→ 切换用例，q/ESC 退出。

use std::io;
use std::path::Path;

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers, KeyEventKind};
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::crossterm::ExecutableCommand;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::prelude::{Color, Line, Modifier, Style, Stylize};
use ratatui::widgets::{
    Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap,
};
use ratatui::{Frame, Terminal};

use crate::engine::forget::{ForgetCaseData, ForgetPipelineSuite, NodeForgetStat};
use crate::engine::suite::{TestCaseOutcome, TestSuite};

/// 观测界面状态
struct ObserverState {
    outcomes: Vec<TestCaseOutcome>,
    case_idx: usize,
    node_idx: usize,
    detail_scroll: u16,
}

impl ObserverState {
    fn case_name(&self) -> String {
        self.outcomes
            .get(self.case_idx)
            .map(|o| o.case_name.clone())
            .unwrap_or_default()
    }

    fn case_data(&self) -> Option<&ForgetCaseData> {
        self.outcomes
            .get(self.case_idx)
            .and_then(|o| o.data.downcast_ref::<ForgetCaseData>())
    }

    fn nodes(&self) -> Vec<&NodeForgetStat> {
        self.case_data().map(|d| d.nodes.iter().collect()).unwrap_or_default()
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

/// 节点列表项文本
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

/// 详情面板：图原文 / 缺失度 / 遮罩输入 / LLM 原始回复
fn detail_lines(data: &ForgetCaseData, node: &NodeForgetStat) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let short_id: String = node.id.chars().take(8).collect();

    lines.push(Line::from(format!(
        "{} [{}]  md {:.3} → {:.3}",
        node.type_name, short_id, node.md_before, node.md_after
    ))
    .style(Style::default().add_modifier(Modifier::BOLD)));

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
            lines.push(Line::from(format!(
                "── 遮罩 {}/{} = {:.0}% ──",
                m,
                t,
                m as f32 / t as f32 * 100.0
            ))
            .style(Style::default().fg(Color::Yellow)));
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
            Line::from("── LLM 原始回复 ──")
                .style(Style::default().fg(if node.effective { Color::Green } else { Color::Red })),
        );
        for l in wrap_text(reply, 80) {
            lines.push(Line::from(l));
        }
    }
    lines
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

fn draw(frame: &mut Frame, state: &ObserverState, passed: usize, total: usize) {
    // ── 布局 ──
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(frame.area());

    // ── 顶部：用例 tabs + 通过率 ──
    let titles: Vec<Line> = state
        .outcomes
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
    let title_block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " 遗忘测试观测 | 通过 {}/{} ({:.0}%) | 图: {} ",
            passed,
            total,
            if total > 0 {
                passed as f64 / total as f64 * 100.0
            } else {
                0.0
            },
            state.case_data().map(|d| "").unwrap_or("")
        ));
    frame.render_widget(
        Tabs::new(titles)
            .select(state.case_idx)
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

    let data = state.case_data();
    let nodes = state.nodes();

    // 左栏：节点列表
    let list_items: Vec<ListItem> = nodes.iter().map(|n| ListItem::new(node_item_line(n))).collect();
    let mut list_state = ListState::default();
    if !nodes.is_empty() {
        list_state.select(Some(state.node_idx.min(nodes.len() - 1)));
    }
    let list = List::new(list_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    " 节点 ({}/{}) ",
                    if nodes.is_empty() { 0 } else { state.node_idx + 1 },
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
        .title(format!(" 详情: {} ", state.case_name()));
    if let (Some(data), Some(node)) = (data, nodes.get(state.node_idx.min(nodes.len().saturating_sub(1)))) {
        let lines = detail_lines(data, node);
        frame.render_widget(
            Paragraph::new(lines)
                .block(detail_block)
                .wrap(Wrap { trim: false })
                .scroll((state.detail_scroll, 0)),
            mid[1],
        );
    } else {
        // activation / incremental 等无节点列表的用例：展示 metrics + 明细文本
        let mut lines: Vec<Line> = Vec::new();
        if let Some(d) = data {
            for (group, label, value) in &d.metrics {
                lines.push(Line::from(format!("{} | {}: {}", group, label, value)));
            }
            lines.push(Line::from(""));
            for dl in &d.detail_lines {
                lines.push(Line::from(dl.as_str()));
            }
        }
        frame.render_widget(
            Paragraph::new(lines)
                .block(detail_block)
                .wrap(Wrap { trim: false })
                .scroll((state.detail_scroll, 0)),
            mid[1],
        );
    }

    // ── 底部：按键提示 ──
    let hint = " ↑/↓ 节点  ←/→ 用例  PgUp/PgDn 详情滚动  q/ESC 退出 ";
    frame.render_widget(
        Paragraph::new(Line::from(hint).style(Style::default().fg(Color::DarkGray))),
        chunks[2],
    );
}

fn handle_event(state: &mut ObserverState, key: KeyCode) -> bool {
    match key {
        KeyCode::Char('q') | KeyCode::Esc => return true,
        KeyCode::Left | KeyCode::Char('h') => {
            if !state.outcomes.is_empty() {
                state.case_idx = (state.case_idx + state.outcomes.len() - 1) % state.outcomes.len();
            }
            state.node_idx = 0;
            state.detail_scroll = 0;
        }
        KeyCode::Right | KeyCode::Char('l') => {
            if !state.outcomes.is_empty() {
                state.case_idx = (state.case_idx + 1) % state.outcomes.len();
            }
            state.node_idx = 0;
            state.detail_scroll = 0;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            if state.node_idx > 0 {
                state.node_idx -= 1;
                state.detail_scroll = 0;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let n = state.nodes().len();
            if n > 0 && state.node_idx + 1 < n {
                state.node_idx += 1;
                state.detail_scroll = 0;
            }
        }
        KeyCode::PageDown => {
            state.detail_scroll = state.detail_scroll.saturating_add(5);
        }
        KeyCode::PageUp => {
            state.detail_scroll = state.detail_scroll.saturating_sub(5);
        }
        _ => {}
    }
    false
}

/// 观测 TUI 主入口：`soul-tune forget-tui <graph.json>`
pub fn run(graph_path: &Path) -> color_eyre::Result<()> {
    let suite = ForgetPipelineSuite::load(graph_path)
        .map_err(|e| color_eyre::eyre::eyre!("加载图失败: {}", e))?;
    let total = suite.case_count();
    let outcomes: Vec<TestCaseOutcome> = (0..total).map(|i| suite.run_case(i)).collect();
    let passed = outcomes.iter().filter(|o| o.passed).count();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut state = ObserverState {
        outcomes,
        case_idx: 0,
        node_idx: 0,
        detail_scroll: 0,
    };

    let result = (|| -> color_eyre::Result<()> {
        loop {
            terminal.draw(|frame| draw(frame, &state, passed, total))?;
            if event::poll(std::time::Duration::from_millis(200))? {
                let ev = event::read()?;
                if let Event::Key(key) = ev {
                    if key.kind == KeyEventKind::Press {
                        let quit = handle_event(&mut state, key.code);
                        if quit {
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    })();

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_text_chinese() {
        let wrapped = wrap_text("一二三四五六七八九十", 4);
        assert_eq!(wrapped, vec!["一二三四", "五六七八", "九十"]);
    }

    #[test]
    fn test_action_style() {
        assert_eq!(
            action_style("Revised", true),
            Style::default().fg(Color::Green)
        );
        assert_eq!(
            action_style("Revised", false),
            Style::default().fg(Color::Red)
        );
        assert_eq!(
            action_style("MaskOnly", false),
            Style::default().fg(Color::Yellow)
        );
    }
}
