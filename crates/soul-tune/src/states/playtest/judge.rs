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
use crate::engine::playtest::{HitStage, PlayRunSnapshot, PlayTestResult, PlayTurnResult};
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
    AllRevealed,
}

pub struct PlayTestJudgeState {
    result: PlayTestResult,
    current_turn: usize,
    current_run: usize,
    phase: JudgePhase,
    debug_view: DebugView,
    think_fold_a: ExpandableList,
    think_fold_b: ExpandableList,
    scroll_a: ScrollState,
    scroll_b: ScrollState,
    json_scroll: ScrollState,
    reveal_cursor: usize,
}

impl PlayTestJudgeState {
    pub fn new(result: PlayTestResult) -> Self {
        let total_slots: usize = result.turns.iter().map(|t| t.runs.len()).sum();
        Self {
            result,
            current_turn: 0,
            current_run: 0,
            phase: JudgePhase::Voting,
            debug_view: DebugView::Hidden,
            think_fold_a: ExpandableList::new(total_slots),
            think_fold_b: ExpandableList::new(total_slots),
            scroll_a: ScrollState::new(),
            scroll_b: ScrollState::new(),
            json_scroll: ScrollState::new(),
            reveal_cursor: 0,
        }
    }

    fn current_turn(&self) -> &PlayTurnResult {
        &self.result.turns[self.current_turn]
    }

    fn current_runs(&self) -> &[PlayRunSnapshot] {
        &self.current_turn().runs
    }

    fn display_run_idx(&self) -> usize {
        let max_idx = self.current_runs().len().saturating_sub(1);
        match self.phase {
            JudgePhase::Voting => self.current_run.min(max_idx),
            JudgePhase::AllRevealed => self.reveal_cursor.min(max_idx),
        }
    }

    fn display_run(&self) -> &PlayRunSnapshot {
        //runs为空时兜底（正常路径已保证至少1条，防御外部构造的异常结果）
        static EMPTY: PlayRunSnapshot = PlayRunSnapshot {
            embedding_response: None,
            fullpipeline_response: None,
            swap: false,
            human_pick: None,
            error: None,
        };
        match self.current_runs().get(self.display_run_idx()) {
            Some(run) => run,
            None => &EMPTY,
        }
    }

    fn flat_display_idx(&self) -> usize {
        self.result.turns[..self.current_turn]
            .iter()
            .map(|t| t.runs.len())
            .sum::<usize>()
            + self.display_run_idx()
    }

    fn resp_a(&self) -> (Option<String>, String) {
        let run = self.display_run();
        let val = if run.swap {
            run.fullpipeline_response.as_deref()
        } else {
            run.embedding_response.as_deref()
        };
        match val {
            Some(s) if !s.is_empty() => split_response(s),
            _ => (None, "(无)".to_string()),
        }
    }

    fn resp_b(&self) -> (Option<String>, String) {
        let run = self.display_run();
        let val = if run.swap {
            run.embedding_response.as_deref()
        } else {
            run.fullpipeline_response.as_deref()
        };
        match val {
            Some(s) if !s.is_empty() => split_response(s),
            _ => (None, "(无)".to_string()),
        }
    }

    fn label_a(&self) -> String {
        if self.phase == JudgePhase::AllRevealed {
            let run = self.display_run();
            let real = if run.swap {
                "FullPipeline"
            } else {
                "Embedding"
            };
            let picked = if run.human_pick == Some(0) {
                " ✓"
            } else {
                ""
            };
            format!("{} {}", real, picked)
        } else {
            "响应 A".to_string()
        }
    }

    fn label_b(&self) -> String {
        if self.phase == JudgePhase::AllRevealed {
            let run = self.display_run();
            let real = if run.swap {
                "Embedding"
            } else {
                "FullPipeline"
            };
            let picked = if run.human_pick == Some(1) {
                " ✓"
            } else {
                ""
            };
            format!("{} {}", real, picked)
        } else {
            "响应 B".to_string()
        }
    }

    fn think_a_expanded(&self) -> bool {
        self.think_fold_a.is_expanded(self.flat_display_idx())
    }

    fn think_b_expanded(&self) -> bool {
        self.think_fold_b.is_expanded(self.flat_display_idx())
    }

    fn count_votes(&self) -> (usize, usize, usize) {
        let turn = self.current_turn();
        let mut emb = 0usize;
        let mut full = 0usize;
        let mut skipped = 0usize;
        for run in &turn.runs {
            match run.human_pick {
                Some(0) => {
                    if run.swap {
                        full += 1;
                    } else {
                        emb += 1;
                    }
                }
                Some(1) => {
                    if run.swap {
                        emb += 1;
                    } else {
                        full += 1;
                    }
                }
                None => skipped += 1,
                _ => skipped += 1,
            }
        }
        (emb, full, skipped)
    }

    fn advance_vote(&mut self) {
        //按当前回合实际runs数量推进，避免查询生成失败时runs不足导致越界
        let run_count = self.current_runs().len().max(1);
        if self.current_run + 1 < run_count {
            self.current_run += 1;
            self.scroll_a.reset();
            self.scroll_b.reset();
        } else {
            self.phase = JudgePhase::AllRevealed;
            self.reveal_cursor = 0;
            self.scroll_a.reset();
            self.scroll_b.reset();
        }
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
                JudgePhase::Voting => {
                    format!("第 {} 次投票", self.current_run + 1)
                }
                JudgePhase::AllRevealed => "投票结果".to_string(),
            }
        );
        Block::bordered()
            .title(title)
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        self.render_info_area(frame, layout[1]);

        if self.phase == JudgePhase::AllRevealed {
            self.render_reveal_summary(frame, layout[1]);
        }

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
                h.push(("[T]".into(), "展开 A 深思".into()));
                h.push(("[Ctrl+T]".into(), "展开 B 深思".into()));
                h.push(("[D]".into(), "详览".into()));
                h.push(("[Q]".into(), "返回".into()));
                h
            }
            JudgePhase::AllRevealed => {
                let mut h = vec![
                    ("[←→]".into(), "切换查看".into()),
                    ("[↑↓]".into(), "A 滚动".into()),
                    ("[Ctrl+↑↓]".into(), "B 滚动".into()),
                ];
                h.push(("[T]".into(), "折叠 A 深思".into()));
                h.push(("[Ctrl+T]".into(), "折叠 B 深思".into()));
                h.push(("[D]".into(), "切换详览".into()));
                h.push(("[Enter]".into(), "下一轮".into()));
                h.push(("[Q]".into(), "返回列表".into()));
                h
            }
        };
        status_bar::render_status_bar(frame, layout[3], &hints);
    }

    fn render_reveal_summary(&self, frame: &mut Frame, _area: Rect) {
        let (emb, full, skipped) = self.count_votes();
        let emb_bar = "█".repeat(emb).to_string();
        let full_bar = "█".repeat(full).to_string();

        let summary = format!(
            "Embedding: {} {}胜  FullPipeline: {} {}胜  跳过: {}",
            emb_bar, emb, full_bar, full, skipped
        );
        let cursor_info = format!(
            " {} 第 {} 次 {}",
            if self.reveal_cursor > 0 { "◄" } else { " " },
            self.reveal_cursor + 1,
            if self.reveal_cursor + 1 < self.current_runs().len() {
                "►"
            } else {
                " "
            }
        );

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(summary, Style::new().fg(Color::Yellow)),
                Span::raw("   "),
                Span::styled(cursor_info, Style::new().fg(Color::DarkGray)),
            ])),
            Rect {
                x: _area.x,
                y: _area.y.saturating_sub(1),
                width: _area.width,
                height: 1,
            },
        );
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

        if self.phase == JudgePhase::AllRevealed && !turn.generated_queries_json.is_empty() {
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

        let run = self.display_run();
        if let Some(ref err) = run.error {
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
        let run = self.display_run();
        let color = if self.phase == JudgePhase::AllRevealed && run.human_pick == Some(0) {
            Color::Green
        } else {
            Color::Reset
        };
        self.render_panel(
            frame,
            area,
            &self.label_a(),
            color,
            &self.scroll_a,
            self.think_a_expanded(),
            true,
            &self.resp_a(),
        );
    }

    fn render_panel_b(&self, frame: &mut Frame, area: Rect) {
        let run = self.display_run();
        let color = if self.phase == JudgePhase::AllRevealed && run.human_pick == Some(1) {
            Color::Green
        } else {
            Color::Reset
        };
        self.render_panel(
            frame,
            area,
            &self.label_b(),
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
            area.height.saturating_div(2).saturating_sub(4),
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
            area.height.saturating_div(2).saturating_sub(4),
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
        //runs为空时（防御异常数据）禁用投票等需要索引的操作
        if self.current_runs().is_empty() {
            return match key.code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => Transition::ToMain,
                _ => Transition::None,
            };
        }
        match self.phase {
            JudgePhase::Voting => match key.code {
                KeyCode::Char('1') => {
                    let idx = self.display_run_idx();
                    self.result.turns[self.current_turn].runs[idx].human_pick = Some(0);
                    self.advance_vote();
                    Transition::None
                }
                KeyCode::Char('2') => {
                    let idx = self.display_run_idx();
                    self.result.turns[self.current_turn].runs[idx].human_pick = Some(1);
                    self.advance_vote();
                    Transition::None
                }
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    let idx = self.display_run_idx();
                    self.result.turns[self.current_turn].runs[idx].human_pick = None;
                    self.advance_vote();
                    Transition::None
                }
                KeyCode::Char('t') | KeyCode::Char('T') => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.think_fold_b.toggle(self.flat_display_idx());
                    } else {
                        self.think_fold_a.toggle(self.flat_display_idx());
                    }
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
            JudgePhase::AllRevealed => match key.code {
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
                        self.think_fold_b.toggle(self.flat_display_idx());
                    } else {
                        self.think_fold_a.toggle(self.flat_display_idx());
                    }
                    Transition::None
                }
                KeyCode::Left => {
                    if self.reveal_cursor > 0 {
                        self.reveal_cursor -= 1;
                        self.scroll_a.reset();
                        self.scroll_b.reset();
                    }
                    Transition::None
                }
                KeyCode::Right => {
                    if self.reveal_cursor + 1 < self.current_runs().len() {
                        self.reveal_cursor += 1;
                        self.scroll_a.reset();
                        self.scroll_b.reset();
                    }
                    Transition::None
                }
                KeyCode::Enter => {
                    if self.current_turn + 1 < self.result.turns.len() {
                        self.current_turn += 1;
                        self.current_run = 0;
                        self.reveal_cursor = 0;
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
