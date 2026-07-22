use std::path::PathBuf;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::engine::playtest::{ConversationEntry, DialogueFile};
use crate::widgets::status_bar;

pub struct PlayTestInputState {
    pub graph_dir: Option<PathBuf>,
    pub messages: TextArea<'static>,
    error: Option<String>,
}

impl PlayTestInputState {
    pub fn new() -> Self {
        let mut msgs = TextArea::default();
        msgs.set_placeholder_text("你好，你是谁？\n你有什么爱好？");
        Self {
            graph_dir: None,
            messages: msgs,
            error: None,
        }
    }

    pub fn with_graph_dir(mut self, dir: PathBuf) -> Self {
        self.graph_dir = Some(dir);
        self
    }

    fn submit(&mut self) -> Transition {
        let graph_dir = match &self.graph_dir {
            Some(d) => d.clone(),
            None => {
                self.error = Some("请选择图目录（按 B 浏览）".to_string());
                return Transition::None;
            }
        };

        let lines: Vec<String> = self
            .messages
            .lines()
            .iter()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            self.error = Some("请输入至少一条用户消息".to_string());
            return Transition::None;
        }

        let graph_path_abs = graph_dir.join("graph.json");
        let dialogue = DialogueFile {
            name: Some("手动输入对话".to_string()),
            graph_path: graph_path_abs.to_string_lossy().to_string(),
            config: None,
            conversations: lines
                .into_iter()
                .map(|msg| ConversationEntry { user_message: msg })
                .collect(),
        };

        Transition::ToPlayTestManualRun(dialogue)
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
            .title(" 手动输入对话表 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let inner = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Length(2),
                Constraint::Fill(1),
            ])
            .split(layout[1]);

        // Graph directory display
        let gp_display = match &self.graph_dir {
            Some(d) => d.to_string_lossy().to_string(),
            None => "[未选择]".to_string(),
        };
        let gp_block = Block::bordered().title(" 图目录 ").fg(Color::Green);
        let gp_inner = gp_block.inner(inner[0]);
        gp_block.render(inner[0], frame.buffer_mut());
        frame.render_widget(Paragraph::new(gp_display), gp_inner);

        // Error message
        if let Some(ref err) = self.error {
            frame.render_widget(Paragraph::new(err.as_str()).fg(Color::Red), inner[1]);
        }

        // Messages input
        let msg_block = Block::bordered()
            .title(" 对话内容（一行一条用户消息）")
            .fg(Color::Green);
        let msg_inner = msg_block.inner(inner[2]);
        msg_block.render(inner[2], frame.buffer_mut());
        frame.render_widget(&self.messages, msg_inner);

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[B]".into(), "浏览图目录".into()),
                ("[Enter]".into(), "开始测试".into()),
                ("[Esc]".into(), "返回".into()),
            ],
        );
    }

    fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Char('b') | KeyCode::Char('B') => Transition::ToGraphBrowse,
            KeyCode::Enter => self.submit(),
            _ => {
                self.messages.input(key);
                Transition::None
            }
        }
    }
}

impl Component for PlayTestInputState {
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
