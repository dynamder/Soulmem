use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::base::{AlgoType, RetrieveMode, Transition};
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
            "  快捷键:",
            "    [R]  Retrieve             完整流水线检索测试",
            "    [C]  Consolidate          巩固（未实现）",
            "    [F]  Forget               遗忘（未实现）",
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
                ("[B]".into(), "批量".into()),
                ("[:]".into(), "命令".into()),
                ("[Q]".into(), "退出".into()),
            ],
        );
    }

    pub fn handle_key(key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('r') | KeyCode::Char('R') => {
                Transition::ToSelectDataset(AlgoType::Retrieve(RetrieveMode::FullPipeline))
            }
            KeyCode::Char('f') | KeyCode::Char('F') => Transition::ToMain,
            KeyCode::Char('c') | KeyCode::Char('C') => Transition::ToMain,
            KeyCode::Char('b') | KeyCode::Char('B') => Transition::ToSelectBatchDir,
            KeyCode::Char(':') => Transition::ToCommand(String::new()),
            KeyCode::Char('q') | KeyCode::Char('Q') => Transition::Quit,
            _ => Transition::None,
        }
    }
}
