use std::path::PathBuf;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::base::{RetrieveMode, Transition};
use crate::component::{Component, ComponentEvent};
use crate::tui::components::status_bar;

pub struct BatchModeState {
    pub dir: PathBuf,
    pub dir_name: String,
}

impl BatchModeState {
    pub fn new(dir: PathBuf) -> Self {
        let dir_name = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| dir.to_string_lossy().to_string());
        Self { dir, dir_name }
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
            .title(" 选择批量模式 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let content = vec![
            "",
            &format!("  目录: {}", self.dir_name),
            "",
            "  选择检索模式:",
            "    [E]  Embedding       余弦相似度检索 (含权重扫描)",
            "    [A]  Association     PPR 图关联检索",
            "    [F]  Full Pipeline   完整流水线检索",
            "",
            "  [Esc] 返回",
        ]
        .join("\n");
        frame.render_widget(Paragraph::new(content), layout[1]);

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[E]".into(), "Embedding".into()),
                ("[A]".into(), "Association".into()),
                ("[F]".into(), "Full".into()),
                ("[Esc]".into(), "返回".into()),
            ],
        );
    }

    pub fn handle_key(&self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('e') | KeyCode::Char('E') => {
                Transition::ToBatchRun(self.dir.clone(), RetrieveMode::Embedding)
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                Transition::ToBatchRun(self.dir.clone(), RetrieveMode::Association)
            }
            KeyCode::Char('f') | KeyCode::Char('F') => {
                Transition::ToBatchRun(self.dir.clone(), RetrieveMode::FullPipeline)
            }
            KeyCode::Esc => Transition::ToMain,
            _ => Transition::None,
        }
    }
}

impl Component for BatchModeState {
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
