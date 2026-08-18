//! 遗忘测试模式选择页（Mask / Revise / Pipeline 三阶段）

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::base::{AlgoType, ForgetMode, Transition};
use crate::component::{Component, ComponentEvent};
use crate::widgets::status_bar;

pub struct ForgetModeSelectState;

impl Component for ForgetModeSelectState {
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

impl ForgetModeSelectState {
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
            .title(" 选择遗忘测试模式 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let content = vec![
            "",
            "  选择遗忘测试模式:",
            "    [M]  Mask       只验证遮罩（纯算法，无 LLM）",
            "    [R]  Revise     验证遮罩补全（llama-server）",
            "    [F]  Full       全管线（衰减+遮罩+补全+边衰减）",
            "",
            "  [Esc] 返回",
        ]
        .join("\n");
        frame.render_widget(Paragraph::new(content), layout[1]);

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[M]".into(), "遮罩".into()),
                ("[R]".into(), "补全".into()),
                ("[F]".into(), "全管线".into()),
                ("[Esc]".into(), "返回".into()),
            ],
        );
    }

    pub fn handle_key(&self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('m') | KeyCode::Char('M') => {
                Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Mask))
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Revise))
            }
            KeyCode::Char('f') | KeyCode::Char('F') => {
                Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Pipeline))
            }
            KeyCode::Esc => Transition::ToMain,
            _ => Transition::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_keys() {
        let s = ForgetModeSelectState::new();
        assert!(matches!(
            s.handle_key(KeyEvent::from(KeyCode::Char('m'))),
            Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Mask))
        ));
        assert!(matches!(
            s.handle_key(KeyEvent::from(KeyCode::Char('r'))),
            Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Revise))
        ));
        assert!(matches!(
            s.handle_key(KeyEvent::from(KeyCode::Char('f'))),
            Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Pipeline))
        ));
        assert!(matches!(
            s.handle_key(KeyEvent::from(KeyCode::Esc)),
            Transition::ToMain
        ));
    }
}
