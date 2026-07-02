use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::base::Transition;
use crate::tui::components::status_bar;

pub struct MainState;

impl MainState {
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
            "  可用算法测试:",
            "    ●  检索 (retrieve)",
            "    ●  巩固 (consolidate)",
            "    ●  遗忘 (forget)",
            "",
            "  输入 `:` 进入命令模式",
            "  或按 `T` 快速开始测试向导",
        ]
        .join("\n");
        frame.render_widget(Paragraph::new(content), layout[1]);

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[:]".into(), "命令".into()),
                ("[T]".into(), "测试".into()),
                ("[Q]".into(), "退出".into()),
            ],
        );
    }

    pub fn handle_key(key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char(':') => Transition::ToCommand(String::new()),
            KeyCode::Char('t') | KeyCode::Char('T') => Transition::ToCommand("test ".into()),
            KeyCode::Char('q') | KeyCode::Char('Q') => Transition::Quit,
            _ => Transition::None,
        }
    }
}
