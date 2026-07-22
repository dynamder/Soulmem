use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::widgets::status_bar;

pub struct SelectAlgoState;

impl Component for SelectAlgoState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => self.handle_key(key),
            _ => Transition::None,
        }
    }
    fn view(&self, frame: &mut Frame) {
        self.render(frame);
    }
}

impl SelectAlgoState {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .split(area);

        Block::bordered()
            .title(" 比对测试 · 选择算法 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let content = vec![
            "",
            "  对比测试: Embedding vs FullPipeline 的指标提升",
            "",
            "  选择要测试的算法:",
            "    [R]  Retrieve          检索（Embedding ↔ Full Pipeline）",
            "    [C]  Consolidate       巩固（未实现）",
            "    [F]  Forget            遗忘（未实现）",
            "",
            "  [Esc] 返回",
        ]
        .join("\n");
        frame.render_widget(Paragraph::new(content), layout[1]);

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[R]".into(), "检索对比".into()),
                ("[Esc]".into(), "返回".into()),
            ],
        );
    }

    pub fn handle_key(&self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('r') | KeyCode::Char('R') => Transition::ToSelectCompareDataset,
            KeyCode::Esc => Transition::ToMain,
            _ => Transition::None,
        }
    }
}
