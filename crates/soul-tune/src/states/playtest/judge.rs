use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::engine::playtest::repair::split_response;
use crate::engine::playtest::{HitStage, PlayTestResult, PlayTurnResult};
use crate::widgets::expandable::ExpandableList;
use crate::widgets::scroll::display_width;
use crate::widgets::scroll::ScrollState;
use crate::widgets::status_bar;

fn wrapped_line_count(text: &str, col_width: u16) -> usize {
    if col_width < 2 {
        return text.lines().count();
    }
    let cw = col_width as usize;
    text.lines()
        .map(|line| {
            let w = display_width(line);
            if w == 0 {
                1
            } else {
                (w + cw - 1) / cw
            }
        })
        .sum()
}

#[derive(PartialEq, Eq)]
enum DebugView {
    Hidden,
    Retrieval,
    Queries,
}

#[derive(PartialEq, Eq)]
enum JudgePhase {
    Voting,
    Reveal,
}

pub struct PlayTestJudgeState {
    result: PlayTestResult,
    current_turn: usize,
    phase: JudgePhase,
    debug_view: DebugView,
    think_fold_a: ExpandableList,
    think_fold_b: ExpandableList,
    scroll_a: ScrollState,
    scroll_b: ScrollState,
    json_scroll: ScrollState,
}

impl PlayTestJudgeState {
    pub fn new(result: PlayTestResult) -> Self {
        let n = result.turns.len();
        Self {
            result,
            current_turn: 0,
            phase: JudgePhase::Voting,
            debug_view: DebugView::Hidden,
            think_fold_a: ExpandableList::new(n),
            think_fold_b: ExpandableList::new(n),
            scroll_a: ScrollState::new(),
            scroll_b: ScrollState::new(),
            json_scroll: ScrollState::new(),
        }
    }

    fn current_turn(&self) -> &PlayTurnResult {
        &self.result.turns[self.current_turn]
    }

    fn resp_a(&self) -> (Option<String>, String) {
        let turn = self.current_turn();
        let val = if turn.swap {
            turn.fullpipeline_response.as_deref()
        } else {
            turn.embedding_response.as_deref()
        };
        match val {
            Some(s) if !s.is_empty() => split_response(s),
            _ => (None, "(无)".to_string()),
        }
    }

    fn resp_b(&self) -> (Option<String>, String) {
        let turn = self.current_turn();
        let val = if turn.swap {
            turn.embedding_response.as_deref()
        } else {
            turn.fullpipeline_response.as_deref()
        };
        match val {
            Some(s) if !s.is_empty() => split_response(s),
            _ => (None, "(无)".to_string()),
        }
    }

    fn label_a(&self) -> &str {
        if self.phase == JudgePhase::Reveal {
            let turn = self.current_turn();
            let real = if turn.swap {
                "FullPipeline"
            } else {
                "Embedding"
            };
            let picked = if turn.human_pick == Some(0) {
                " ✓"
            } else {
                ""
            };
            return Box::leak(format!("{} {}", real, picked).into_boxed_str());
        }
        "响应 A"
    }

    fn label_b(&self) -> &str {
        if self.phase == JudgePhase::Reveal {
            let turn = self.current_turn();
            let real = if turn.swap {
                "Embedding"
            } else {
                "FullPipeline"
            };
            let picked = if turn.human_pick == Some(1) {
                " ✓"
            } else {
                ""
            };
            return Box::leak(format!("{} {}", real, picked).into_boxed_str());
        }
        "响应 B"
    }

    fn think_a_expanded(&self) -> bool {
        self.think_fold_a.is_expanded(self.current_turn)
    }

    fn think_b_expanded(&self) -> bool {
        self.think_fold_b.is_expanded(self.current_turn)
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Min(3),
                Constraint::Fill(4),
                Constraint::Length(1),
            ])
            .split(area);

        let n = self.result.turns.len();
        let title = format!(
            " 角色扮演 · 第 {}/{} 轮 — {} ",
            self.current_turn + 1,
            n,
            match self.phase {
                JudgePhase::Voting => "选择更好的响应",
                JudgePhase::Reveal => "结果详情",
            }
        );
        Block::bordered()
            .title(title)
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        self.render_info_area(frame, layout[1]);

        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(layout[2]);

        self.render_panel_a(frame, split[0]);
        self.render_panel_b(frame, split[1]);

        match self.debug_view {
            DebugView::Retrieval => self.render_trace_detail(frame),
            DebugView::Queries => self.render_query_detail(frame),
            DebugView::Hidden => {}
        }

        let hints = match self.phase {
            JudgePhase::Voting => {
                let mut h = vec![
                    ("[1]".into(), "选 A".into()),
                    ("[2]".into(), "选 B".into()),
                    ("[S]".into(), "跳过".into()),
                ];
                h.push(("[↑↓]".into(), "A 滚动".into()));
                h.push(("[Ctrl+↑↓]".into(), "B 滚动".into()));
                h.push(("[D]".into(), "详览".into()));
                h.push(("[Q]".into(), "返回".into()));
                h
            }
            JudgePhase::Reveal => {
                let mut h = vec![
                    ("[↑↓]".into(), "A 滚动".into()),
                    ("[Ctrl+↑↓]".into(), "B 滚动".into()),
                    ("[D]".into(), "切换详览".into()),
                ];
                h.push(("[T]".into(), "折叠 A 深思".into()));
                h.push(("[Ctrl+T]".into(), "折叠 B 深思".into()));
                h.push(("[Enter]".into(), "下一轮".into()));
                h.push(("[Q]".into(), "返回列表".into()));
                h
            }
        };
        status_bar::render_status_bar(frame, layout[3], &hints);
    }

    fn render_info_area(&self, frame: &mut Frame, area: Rect) {
        let turn = self.current_turn();
        let mut info_lines: Vec<Line> = Vec::new();

        if let Some(ref role) = self.result.human_role {
            info_lines.push(Line::from(Span::styled(
                format!("角色: {}", role),
                Style::new().cyan(),
            )));
        }
        info_lines.push(Line::from(Span::raw(format!(
            "用户: {}",
            turn.user_message
        ))));

        if self.phase == JudgePhase::Reveal && !turn.generated_queries_json.is_empty() {
            info_lines.push(Line::from(Span::styled(
                format!(
                    "Query: {}",
                    &turn
                        .generated_queries_json
                        .chars()
                        .take(80)
                        .collect::<String>()
                ),
                Style::new().dark_gray(),
            )));
        }

        if let Some(ref err) = turn.error {
            info_lines.push(Line::from(Span::styled(
                format!("错误: {}", err),
                Style::new().red(),
            )));
        }

        frame.render_widget(Paragraph::new(Text::from(info_lines)), area);
    }

    fn build_panel_lines(
        &self,
        think: &Option<String>,
        body: &str,
        expanded: bool,
        is_panel_a: bool,
    ) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        if let Some(t) = think {
            if expanded {
                lines.push(Line::from(Span::styled("深思:", Style::new().yellow())));
                for l in t.lines() {
                    lines.push(Line::from(Span::raw(format!("  {}", l))));
                }
            } else {
                let preview: String = t.chars().take(60).collect();
                let hint = if is_panel_a {
                    "[T] 展开"
                } else {
                    "[Ctrl+T] 展开"
                };
                lines.push(Line::from(Span::styled(
                    format!("深思: {}...  {}", preview, hint),
                    Style::new().yellow(),
                )));
            }
        }
        for l in body.lines() {
            lines.push(Line::from(Span::raw(l.to_string())));
        }
        lines
    }

    fn render_panel(
        &self,
        frame: &mut Frame,
        area: Rect,
        label: &str,
        color: Color,
        scroll: &ScrollState,
        expanded: bool,
        is_panel_a: bool,
        resp: &(Option<String>, String),
    ) {
        let block = Block::bordered().title(label).fg(color);
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let (think, body) = resp;
        let lines = self.build_panel_lines(think, body, expanded, is_panel_a);

        let (content_rect, bar_rect) = ScrollState::split_area(inner);
        let total_wrapped = wrapped_line_count(
            &lines
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
            content_rect.width,
        );
        let max_offset = total_wrapped.saturating_sub(content_rect.height as usize);
        let clamped_offset = scroll.offset.min(max_offset);

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((clamped_offset as u16, 0)),
            content_rect,
        );
        ScrollState::render_scrollbar(
            frame,
            bar_rect,
            total_wrapped,
            content_rect.height,
            clamped_offset,
        );
    }

    fn render_panel_a(&self, frame: &mut Frame, area: Rect) {
        let color = if self.phase == JudgePhase::Reveal && self.current_turn().human_pick == Some(0)
        {
            Color::Green
        } else {
            Color::Reset
        };
        self.render_panel(
            frame,
            area,
            self.label_a(),
            color,
            &self.scroll_a,
            self.think_a_expanded(),
            true,
            &self.resp_a(),
        );
    }

    fn render_panel_b(&self, frame: &mut Frame, area: Rect) {
        let color = if self.phase == JudgePhase::Reveal && self.current_turn().human_pick == Some(1)
        {
            Color::Green
        } else {
            Color::Reset
        };
        self.render_panel(
            frame,
            area,
            self.label_b(),
            color,
            &self.scroll_b,
            self.think_b_expanded(),
            false,
            &self.resp_b(),
        );
    }

    fn render_trace_detail(&self, frame: &mut Frame) {
        let area = frame.area();
        let bottom_area = Rect::new(
            area.x,
            area.y + area.height / 2,
            area.width,
            area.height / 2 - 4,
        );
        let block = Block::bordered().title(" 检索轨迹 ").fg(Color::Yellow);
        let inner = block.inner(bottom_area);
        block.render(bottom_area, frame.buffer_mut());

        let turn = self.current_turn();
        let mut lines: Vec<Line> = Vec::new();

        if let Some(ref emb_trace) = turn.embedding_trace {
            lines.push(Line::from(Span::styled(
                format!(
                    " Embedding: {:.2}s | {} 个节点",
                    emb_trace.total_elapsed.as_secs_f64(),
                    emb_trace.merged_nodes.len()
                ),
                Style::new().cyan(),
            )));
            for n in emb_trace.merged_nodes.iter().take(5) {
                let stage_mark = match n.stage {
                    HitStage::Similarity => "S",
                    HitStage::Ppr => "P",
                    HitStage::Action => "A",
                    HitStage::Both => "B",
                };
                lines.push(Line::from(Span::raw(format!(
                    "   [{:.2}] {:<20} [{}]",
                    n.score, n.name, stage_mark
                ))));
            }
        }

        if let Some(ref full_trace) = turn.fullpipeline_trace {
            lines.push(Line::from(Span::styled(
                format!(
                    " FullPipeline: {:.2}s | {} 个节点 (sim={} ppr={} act={})",
                    full_trace.total_elapsed.as_secs_f64(),
                    full_trace.merged_nodes.len(),
                    full_trace
                        .merged_nodes
                        .iter()
                        .filter(|n| matches!(n.stage, HitStage::Similarity | HitStage::Both))
                        .count(),
                    full_trace
                        .merged_nodes
                        .iter()
                        .filter(|n| matches!(n.stage, HitStage::Ppr | HitStage::Both))
                        .count(),
                    full_trace
                        .merged_nodes
                        .iter()
                        .filter(|n| matches!(n.stage, HitStage::Action | HitStage::Both))
                        .count()
                ),
                Style::new().green(),
            )));
            for n in full_trace.merged_nodes.iter().take(8) {
                let stage_mark = match n.stage {
                    HitStage::Similarity => "S",
                    HitStage::Ppr => "P",
                    HitStage::Action => "A",
                    HitStage::Both => "B",
                };
                lines.push(Line::from(Span::raw(format!(
                    "   [{:.2}] {:<20} [{}]",
                    n.score, n.name, stage_mark
                ))));
            }
        }

        let line_count = lines.len();
        let (content_rect, bar_rect) = ScrollState::split_area(inner);
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((self.scroll_a.offset as u16, 0)),
            content_rect,
        );
        ScrollState::render_scrollbar(
            frame,
            bar_rect,
            line_count,
            content_rect.height,
            self.scroll_a.offset,
        );
    }

    fn render_query_detail(&self, frame: &mut Frame) {
        let area = frame.area();
        let bottom_area = Rect::new(
            area.x,
            area.y + area.height / 2,
            area.width,
            area.height / 2 - 4,
        );
        let block = Block::bordered().title(" 生成查询 ").fg(Color::Cyan);
        let inner = block.inner(bottom_area);
        block.render(bottom_area, frame.buffer_mut());

        let turn = self.current_turn();
        let pretty = if let Ok(val) =
            serde_json::from_str::<serde_json::Value>(&turn.generated_queries_json)
        {
            serde_json::to_string_pretty(&val)
                .unwrap_or_else(|_| turn.generated_queries_json.clone())
        } else {
            turn.generated_queries_json.clone()
        };

        let lines: Vec<Line> = pretty
            .lines()
            .map(|l| Line::from(Span::raw(l.to_string())))
            .collect();

        let (content_rect, bar_rect) = ScrollState::split_area(inner);
        let total_wrapped = wrapped_line_count(&pretty, content_rect.width);
        let max_offset = total_wrapped.saturating_sub(content_rect.height as usize);
        let clamped = self.json_scroll.offset.min(max_offset);

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((clamped as u16, 0)),
            content_rect,
        );
        ScrollState::render_scrollbar(frame, bar_rect, total_wrapped, content_rect.height, clamped);
    }

    fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match self.phase {
            JudgePhase::Voting => match key.code {
                KeyCode::Char('1') => {
                    let turn = &mut self.result.turns[self.current_turn];
                    turn.human_pick = Some(0);
                    self.phase = JudgePhase::Reveal;
                    Transition::None
                }
                KeyCode::Char('2') => {
                    let turn = &mut self.result.turns[self.current_turn];
                    turn.human_pick = Some(1);
                    self.phase = JudgePhase::Reveal;
                    Transition::None
                }
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    let turn = &mut self.result.turns[self.current_turn];
                    turn.human_pick = None;
                    self.phase = JudgePhase::Reveal;
                    Transition::None
                }
                KeyCode::Char('d') | KeyCode::Char('D') => {
                    self.debug_view = match self.debug_view {
                        DebugView::Hidden => DebugView::Retrieval,
                        DebugView::Retrieval => DebugView::Queries,
                        DebugView::Queries => DebugView::Hidden,
                    };
                    Transition::None
                }
                KeyCode::Char('q') | KeyCode::Char('Q') => Transition::ToMain,
                KeyCode::Up => {
                    if self.debug_view == DebugView::Queries {
                        self.json_scroll.scroll_up();
                    } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_b.scroll_up();
                    } else {
                        self.scroll_a.scroll_up();
                    }
                    Transition::None
                }
                KeyCode::Down => {
                    if self.debug_view == DebugView::Queries {
                        self.json_scroll.scroll_down();
                    } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_b.scroll_down();
                    } else {
                        self.scroll_a.scroll_down();
                    }
                    Transition::None
                }
                _ => Transition::None,
            },
            JudgePhase::Reveal => match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => Transition::ToMain,
                KeyCode::Char('d') | KeyCode::Char('D') => {
                    self.debug_view = match self.debug_view {
                        DebugView::Hidden => DebugView::Retrieval,
                        DebugView::Retrieval => DebugView::Queries,
                        DebugView::Queries => DebugView::Hidden,
                    };
                    Transition::None
                }
                KeyCode::Char('t') | KeyCode::Char('T') => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.think_fold_b.toggle(self.current_turn);
                    } else {
                        self.think_fold_a.toggle(self.current_turn);
                    }
                    Transition::None
                }
                KeyCode::Enter => {
                    if self.current_turn + 1 < self.result.turns.len() {
                        self.current_turn += 1;
                        self.phase = JudgePhase::Voting;
                        self.scroll_a.reset();
                        self.scroll_b.reset();
                        Transition::None
                    } else {
                        Transition::ToMain
                    }
                }
                KeyCode::Up => {
                    if self.debug_view == DebugView::Queries {
                        self.json_scroll.scroll_up();
                    } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_b.scroll_up();
                    } else {
                        self.scroll_a.scroll_up();
                    }
                    Transition::None
                }
                KeyCode::Down => {
                    if self.debug_view == DebugView::Queries {
                        self.json_scroll.scroll_down();
                    } else if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.scroll_b.scroll_down();
                    } else {
                        self.scroll_a.scroll_down();
                    }
                    Transition::None
                }
                _ => Transition::None,
            },
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollDown => self.scroll_a.scroll_down(),
            MouseEventKind::ScrollUp => self.scroll_a.scroll_up(),
            _ => {}
        }
    }
}

impl Component for PlayTestJudgeState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => self.handle_key(key),
            ComponentEvent::Mouse(mouse) => {
                self.handle_mouse(mouse);
                Transition::None
            }
            ComponentEvent::Tick => Transition::None,
        }
    }
    fn view(&self, frame: &mut Frame) {
        self.render(frame);
    }
}
