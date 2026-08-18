use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::widgets::status_bar;

pub struct MainState;

impl Component for MainState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => handle_key(key),
            _ => Transition::None,
        }
    }
    fn view(&self, frame: &mut Frame) {
        render(frame);
    }
}

pub fn render(frame: &mut Frame) {
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
        .title(" Soul-Tune · 记忆算法测试框架 ")
        .fg(Color::Cyan)
        .render(layout[0], frame.buffer_mut());

    let content = vec![
        "",
        "  快捷键:",
        "    [R]  Retrieve             检索测试（选择模式）",
        "    [D]  Diff/对比            比对 Embedding vs Full Pipeline",
        "    [P]  PlayTest             角色扮演测试",
        "    [C]  Consolidate          巩固（未实现）",
        "    [F]  Forget               遗忘测试（选择模式后进入）",
        "    [I]  Inspect              直接检视测试数据",
        "    [B]  Batch                批量运行",
        "    [Q]  退出",
        "",
        "  输入 `:` 进入命令模式",
    ]
    .join("\n");
    frame.render_widget(Paragraph::new(content), layout[1]);

    status_bar::render_status_bar(
        frame,
        layout[2],
        &[
            ("[R]".into(), "检索".into()),
            ("[D]".into(), "对比".into()),
            ("[P]".into(), "扮演".into()),
            ("[F]".into(), "遗忘".into()),
            ("[I]".into(), "检视".into()),
            ("[B]".into(), "批量".into()),
            ("[:]".into(), "命令".into()),
            ("[Q]".into(), "退出".into()),
        ],
    );
}

pub fn handle_key(key: KeyEvent) -> Transition {
    match key.code {
        KeyCode::Char('r') | KeyCode::Char('R') => Transition::ToRetrieveModeSelect,
        KeyCode::Char('d') | KeyCode::Char('D') => Transition::ToSelectAlgo,
        KeyCode::Char('p') | KeyCode::Char('P') => Transition::ToPlayTestSelect,
        KeyCode::Char('f') | KeyCode::Char('F') => Transition::ToForgetModeSelect,
        KeyCode::Char('c') | KeyCode::Char('C') => Transition::ToMain,
        KeyCode::Char('i') | KeyCode::Char('I') => Transition::ToCommand("inspect ".into()),
        KeyCode::Char('b') | KeyCode::Char('B') => Transition::ToSelectBatchDir,
        KeyCode::Char(':') => Transition::ToCommand(String::new()),
        KeyCode::Char('q') | KeyCode::Char('Q') => Transition::Quit,
        _ => Transition::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{AlgoType, ForgetMode};

    #[test]
    fn test_f_key_enters_forget_mode_select() {
        let t = handle_key(KeyEvent::from(KeyCode::Char('f')));
        assert!(matches!(t, Transition::ToForgetModeSelect));
    }

    #[test]
    fn test_q_key_quits() {
        let t = handle_key(KeyEvent::from(KeyCode::Char('q')));
        assert!(matches!(t, Transition::Quit));
    }

    #[test]
    fn test_forget_modes_route_to_dataset() {
        // 模式选择页的三个模式都应路由到选图（AlgoType::Forget(mode)）
        let m = ForgetMode::Mask;
        assert!(matches!(
            Transition::ToSelectDataset(AlgoType::Forget(m)),
            Transition::ToSelectDataset(AlgoType::Forget(_))
        ));
    }
}
