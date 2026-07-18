use ratatui::crossterm::event::{KeyCode, KeyEvent, MouseEvent, MouseEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::eval::playtest::{HitStage, PlayTestResult, PlayTurnResult};
use crate::tui::components::scroll_container::ScrollContainer;
use crate::tui::components::status_bar;

#[derive(PartialEq, Eq)]
enum JudgePhase {
    Voting,
    Reveal,
}

pub struct PlayTestJudgeState {
    result: PlayTestResult,
    current_turn: usize,
    phase: JudgePhase,
    debug_view: bool,
    scroll: ScrollContainer,
}

impl PlayTestJudgeState {
    pub fn new(result: PlayTestResult) -> Self {
        Self {
            result,
            current_turn: 0,
            phase: JudgePhase::Voting,
            debug_view: false,
            scroll: ScrollContainer::new(),
        }
    }

    fn current_turn(&self) -> &PlayTurnResult {
        &self.result.turns[self.current_turn]
    }

    fn resp_a(&self) -> &str {
        let turn = self.current_turn();
        if turn.swap {
            turn.fullpipeline_response.as_deref().unwrap_or("(无)")
        } else {
            turn.embedding_response.as_deref().unwrap_or("(无)")
        }
    }

    fn resp_b(&self) -> &str {
        let turn = self.current_turn();
        if turn.swap {
            turn.embedding_response.as_deref().unwrap_or("(无)")
        } else {
            turn.fullpipeline_response.as_deref().unwrap_or("(无)")
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
            // Mark which one the user picked
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

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Fill(1),
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

        // User message and query info
        let turn = self.current_turn();
        let info_lines = vec![Line::from(Span::raw(format!(
            "用户: {}",
            turn.user_message
        )))];
        let info_text =
            if self.phase == JudgePhase::Reveal && !turn.generated_queries_json.is_empty() {
                Text::from(vec![
                    Line::from(Span::raw(format!("用户: {}", turn.user_message))),
                    Line::from(Span::styled(
                        format!(
                            "Query: {}",
                            &turn
                                .generated_queries_json
                                .chars()
                                .take(80)
                                .collect::<String>()
                        ),
                        Style::new().dark_gray(),
                    )),
                ])
            } else {
                Text::from(info_lines)
            };
        frame.render_widget(Paragraph::new(info_text), layout[1]);

        // Side-by-side responses
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(layout[2]);

        if self.phase == JudgePhase::Voting {
            self.render_response_panel(frame, split[0], self.label_a(), self.resp_a(), false);
            self.render_response_panel(frame, split[1], self.label_b(), self.resp_b(), false);
        } else {
            self.render_reveal(frame, split[0], split[1]);
        }

        let hints = match self.phase {
            JudgePhase::Voting => {
                vec![
                    ("[1]".into(), "选 A".into()),
                    ("[2]".into(), "选 B".into()),
                    ("[S]".into(), "跳过".into()),
                    ("[D]".into(), "详情".into()),
                    ("[Q]".into(), "返回".into()),
                ]
            }
            JudgePhase::Reveal => {
                vec![
                    ("[↑↓]".into(), "滚动".into()),
                    ("[Enter]".into(), "下一轮".into()),
                    ("[Q]".into(), "返回列表".into()),
                ]
            }
        };
        status_bar::render_status_bar(frame, layout[3], &hints);
    }

    fn render_response_panel(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        content: &str,
        highlight: bool,
    ) {
        let color = if highlight {
            Color::Green
        } else {
            Color::Reset
        };
        let block = Block::bordered().title(title).fg(color);
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());
        frame.render_widget(Paragraph::new(content).wrap(Wrap { trim: false }), inner);
    }

    fn render_reveal(&self, frame: &mut Frame, left: Rect, right: Rect) {
        let turn = self.current_turn();

        // Show which was picked
        let hl_a = turn.human_pick == Some(0);
        let hl_b = turn.human_pick == Some(1);
        self.render_response_panel(frame, left, self.label_a(), self.resp_a(), hl_a);
        self.render_response_panel(frame, right, self.label_b(), self.resp_b(), hl_b);

        // Show retrieval trace details below
        if self.debug_view {
            self.render_trace_detail(frame);
        }
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
        let (content_rect, bar_rect) = ScrollContainer::split_area(inner);
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((self.scroll.offset as u16, 0)),
            content_rect,
        );
        ScrollContainer::render_scrollbar(
            frame,
            bar_rect,
            line_count,
            content_rect.height,
            self.scroll.offset,
        );
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
                    self.debug_view = !self.debug_view;
                    Transition::None
                }
                KeyCode::Char('q') | KeyCode::Char('Q') => Transition::ToMain,
                _ => Transition::None,
            },
            JudgePhase::Reveal => match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => Transition::ToMain,
                KeyCode::Enter => {
                    if self.current_turn + 1 < self.result.turns.len() {
                        self.current_turn += 1;
                        self.phase = JudgePhase::Voting;
                        self.scroll.reset();
                        Transition::None
                    } else {
                        Transition::ToMain
                    }
                }
                KeyCode::Up => {
                    self.scroll.scroll_up();
                    Transition::None
                }
                KeyCode::Down => {
                    self.scroll.scroll_down();
                    Transition::None
                }
                _ => Transition::None,
            },
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if self.phase == JudgePhase::Reveal {
            match mouse.kind {
                MouseEventKind::ScrollDown => self.scroll.scroll_down(),
                MouseEventKind::ScrollUp => self.scroll.scroll_up(),
                _ => {}
            }
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
