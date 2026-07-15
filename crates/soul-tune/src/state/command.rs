use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::base::{AlgoType, RetrieveMode, Transition};
use crate::cmd::CmdRegistry;
use crate::tui::components::{command_bar, status_bar};

pub struct CommandState {
    pub input: TextArea<'static>,
    pub suggestions: Vec<String>,
    pub selected_suggestion: usize,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,
}

impl CommandState {
    pub fn new() -> Self {
        let mut input = TextArea::default();
        input.set_placeholder_text("输入命令...");
        Self {
            input,
            suggestions: Vec::new(),
            selected_suggestion: 0,
            history: Vec::new(),
            history_idx: None,
        }
    }

    pub fn update_suggestions(&mut self, registry: &CmdRegistry) {
        let text = self.input.lines().first().map(|s| s.as_str()).unwrap_or("");
        if text.is_empty() {
            self.suggestions.clear();
            return;
        }

        let trimmed = text.trim();
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let ends_with_space = text.ends_with(' ');

        if parts.len() >= 1 && (parts[0] == "test" || parts[0] == "t") {
            let subcommands = [
                ("retrieve", "检索算法测试"),
                ("consolidate", "巩固算法测试"),
                ("forget", "遗忘算法测试"),
            ];

            if parts.len() == 1 && !ends_with_space {
                self.suggestions = vec!["test — 运行算法测试".into()];
                self.suggestions.extend(
                    subcommands
                        .iter()
                        .map(|(n, d)| format!("test {} — {}", n, d)),
                );
            } else {
                let partial = parts.get(1).copied().unwrap_or("");
                self.suggestions = subcommands
                    .iter()
                    .filter(|(name, _)| name.starts_with(partial))
                    .map(|(name, desc)| format!("test {} — {}", name, desc))
                    .collect();
            }
            self.selected_suggestion = 0;
            return;
        }

        self.suggestions = registry
            .fuzzy_cmd_find(text)
            .into_iter()
            .map(|cmd| format!("{} — {}", cmd.name(), cmd.description()))
            .take(5)
            .collect();
        if self.selected_suggestion >= self.suggestions.len() {
            self.selected_suggestion = self.suggestions.len().saturating_sub(1);
        }
    }

    fn current_text(&self) -> &str {
        self.input.lines().first().map(|s| s.as_str()).unwrap_or("")
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Fill(1),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(area);

        Block::bordered()
            .title(" 命令模式 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let mut suggestion_lines: Vec<String> = Vec::new();
        if !self.suggestions.is_empty() {
            suggestion_lines.push("  匹配命令:".into());
            for (i, sug) in self.suggestions.iter().enumerate() {
                let prefix = if i == self.selected_suggestion {
                    "  ▶ "
                } else {
                    "    "
                };
                suggestion_lines.push(format!("{}{}", prefix, sug));
            }
        } else {
            let text = self.current_text();
            if !text.is_empty() {
                suggestion_lines.push("  (无匹配命令)".into());
            }
        }
        frame.render_widget(Paragraph::new(suggestion_lines.join("\n")), layout[1]);

        command_bar::render_command_input(frame, layout[2], &self.input);

        status_bar::render_status_bar(
            frame,
            layout[3],
            &[
                ("[Enter]".into(), "执行".into()),
                ("[Esc]".into(), "取消".into()),
                ("[Tab]".into(), "补全".into()),
                ("[↑↓]".into(), "历史/建议".into()),
            ],
        );
    }

    pub fn handle_key(&mut self, key: KeyEvent, registry: &CmdRegistry) -> Transition {
        match key.code {
            KeyCode::Enter => self.execute_command(),
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Tab => {
                if !self.suggestions.is_empty() {
                    let suggestion = &self.suggestions[self.selected_suggestion];
                    let cmd_name = suggestion.split(" — ").next().unwrap_or(suggestion);
                    let trimmed = self.current_text().trim();
                    let words: Vec<&str> = trimmed.split_whitespace().collect();
                    let cmd_words: Vec<&str> = cmd_name.split_whitespace().collect();

                    let new_text = if cmd_name.starts_with(trimmed) && trimmed != cmd_name {
                        cmd_name.to_string()
                    } else if words.len() < cmd_words.len() {
                        let mut combined = words.clone();
                        combined.extend(cmd_words[words.len()..].iter());
                        format!("{} ", combined.join(" "))
                    } else {
                        let mut combined = words.clone();
                        let last_idx = combined.len().saturating_sub(1);
                        combined[last_idx] = cmd_words[last_idx];
                        combined.join(" ")
                    };

                    self.input = TextArea::default();
                    self.input.insert_str(&new_text);
                    self.update_suggestions(registry);
                }
                Transition::None
            }
            KeyCode::Up => {
                if !self.suggestions.is_empty() {
                    self.selected_suggestion = self.selected_suggestion.saturating_sub(1);
                } else if !self.history.is_empty() {
                    let idx = match self.history_idx {
                        Some(i) if i > 0 => i - 1,
                        _ => self.history.len().saturating_sub(1),
                    };
                    self.history_idx = Some(idx);
                    self.input = TextArea::default();
                    self.input.insert_str(&self.history[idx]);
                }
                Transition::None
            }
            KeyCode::Down => {
                if !self.suggestions.is_empty() {
                    let max = self.suggestions.len().saturating_sub(1);
                    if self.selected_suggestion < max {
                        self.selected_suggestion += 1;
                    }
                } else if let Some(idx) = self.history_idx {
                    if idx + 1 < self.history.len() {
                        let new_idx = idx + 1;
                        self.history_idx = Some(new_idx);
                        self.input = TextArea::default();
                        self.input.insert_str(&self.history[new_idx]);
                    } else {
                        self.history_idx = None;
                        self.input = TextArea::default();
                    }
                }
                Transition::None
            }
            _ => {
                self.input.input(key);
                self.update_suggestions(registry);
                Transition::None
            }
        }
    }

    fn execute_command(&mut self) -> Transition {
        let text = self.current_text().trim().to_string();
        if text.is_empty() {
            return Transition::ToMain;
        }
        self.history.push(text.clone());
        self.history_idx = None;

        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.is_empty() {
            return Transition::ToMain;
        }

        match parts[0] {
            "test" | "t" => {
                let algo = if parts.len() > 1 {
                    match parts[1] {
                        "retrieve" | "r" | "retrieve/embedding" | "re" => {
                            AlgoType::Retrieve(RetrieveMode::Embedding)
                        }
                        "retrieve/association" | "ra" => {
                            AlgoType::Retrieve(RetrieveMode::Association)
                        }
                        "retrieve/full" | "rf" => AlgoType::Retrieve(RetrieveMode::FullPipeline),
                        "consolidate" | "c" => AlgoType::Consolidate,
                        "forget" | "f" => AlgoType::Forget,
                        _ => return Transition::ToMain,
                    }
                } else {
                    return Transition::ToMain;
                };
                Transition::ToSelectDataset(algo)
            }
            "quit" | "q" => Transition::Quit,
            "help" | "h" => Transition::None,
            _ => Transition::ToMain,
        }
    }
}
