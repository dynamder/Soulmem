use std::collections::BTreeMap;
use std::sync::OnceLock;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::base::{AlgoType, ForgetMode, RetrieveMode, Transition};
use crate::cmd::CmdRegistry;
use crate::widgets::scroll::ScrollState;
use crate::widgets::{command_bar, status_bar};

pub struct CommandState {
    pub input: TextArea<'static>,
    pub suggestions: Vec<String>,
    pub suggestion_is_header: Vec<bool>,
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
            suggestion_is_header: Vec::new(),
            selected_suggestion: 0,
            history: Vec::new(),
            history_idx: None,
        }
    }

    pub fn update_suggestions(&mut self, registry: &CmdRegistry) {
        let text = self.input.lines().first().map(|s| s.as_str()).unwrap_or("");
        if text.is_empty() {
            self.suggestions.clear();
            self.suggestion_is_header.clear();
            return;
        }

        let trimmed = text.trim();
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let ends_with_space = text.ends_with(' ');

        if parts.len() >= 1 && (parts[0] == "inspect" || parts[0] == "i") {
            if parts.len() == 1 && !ends_with_space {
                self.suggestions = vec!["inspect <path> — 直接检视测试数据集".into()];
                self.suggestion_is_header = vec![false];
                self.selected_suggestion = 0;
                return;
            } else if ends_with_space || parts.len() > 1 {
                let partial = if parts.len() > 1 {
                    parts[1..].join(" ")
                } else {
                    String::new()
                };
                let all = collect_fixture_entries();
                let partial_lower = partial.to_lowercase();
                let matched: Vec<&FixturePath> = all
                    .iter()
                    .filter(|f| {
                        partial_lower.is_empty() || f.path.to_lowercase().contains(&partial_lower)
                    })
                    .collect();

                // Group by PathKind, limit 20 per group
                let mut groups: BTreeMap<&str, Vec<&FixturePath>> = BTreeMap::new();
                for f in &matched {
                    let cat = match f.kind {
                        PathKind::Graph => "graph",
                        PathKind::Question => "question",
                    };
                    groups.entry(cat).or_default().push(f);
                }

                let mut suggestions = Vec::new();
                let mut is_header = Vec::new();
                for (cat, paths) in &groups {
                    suggestions.push(format!("[{}]", cat));
                    is_header.push(true);
                    for f in paths {
                        suggestions.push(format!("inspect {}", f.path));
                        is_header.push(false);
                    }
                }

                if suggestions.is_empty() {
                    self.suggestions = vec!["inspect <path> — 直接检视测试数据集".into()];
                    self.suggestion_is_header = vec![false];
                } else {
                    self.suggestions = suggestions;
                    self.suggestion_is_header = is_header;
                }
            }
            self.selected_suggestion = 0;
            // If first suggestion is a header, move to the next selectable
            if !self.suggestion_is_header.is_empty() && self.suggestion_is_header[0] {
                self.selected_suggestion = 1;
            }
            return;
        }

        if parts.len() >= 1 && (parts[0] == "test" || parts[0] == "t") {
            let subcommands = [
                ("retrieve", "检索算法测试"),
                ("consolidate", "巩固算法测试"),
                ("forget", "遗忘算法测试"),
            ];

            if parts.len() == 1 && !ends_with_space {
                let mut subs = vec!["test — 运行算法测试".into()];
                subs.extend(
                    subcommands
                        .iter()
                        .map(|(n, d)| format!("test {} — {}", n, d)),
                );
                self.suggestions = subs;
                self.suggestion_is_header = vec![false; self.suggestions.len()];
            } else {
                let partial = parts.get(1).copied().unwrap_or("");
                let subs: Vec<String> = subcommands
                    .iter()
                    .filter(|(name, _)| name.starts_with(partial))
                    .map(|(name, desc)| format!("test {} — {}", name, desc))
                    .collect();
                self.suggestions = subs;
                self.suggestion_is_header = vec![false; self.suggestions.len()];
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
        self.suggestion_is_header = vec![false; self.suggestions.len()];
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
                if i < self.suggestion_is_header.len() && self.suggestion_is_header[i] {
                    suggestion_lines.push(format!("  {}", sug));
                } else {
                    let prefix = if i == self.selected_suggestion {
                        "  ▶ "
                    } else {
                        "    "
                    };
                    suggestion_lines.push(format!("{}{}", prefix, sug));
                }
            }
        } else {
            let text = self.current_text();
            if !text.is_empty() {
                suggestion_lines.push("  (无匹配命令)".into());
            }
        }
        let visible = (layout[1].height as usize).saturating_sub(1);
        let _selected_in_lines = self.selected_suggestion + 1;
        let scroll_off = ScrollState::offset(
            visible as u16,
            suggestion_lines.len(),
            self.selected_suggestion,
        );
        frame.render_widget(
            Paragraph::new(suggestion_lines.join("\n")).scroll((scroll_off as u16, 0)),
            layout[1],
        );

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
                if !self.suggestions.is_empty()
                    && !self.suggestion_is_header.is_empty()
                    && !self.suggestion_is_header[self.selected_suggestion]
                {
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
                        //用户已输入词数不少于命令词数时，只替换能对应上的词，
                        //防止last_idx越界cmd_words导致panic
                        let mut combined = words.clone();
                        let last_idx = combined.len().saturating_sub(1);
                        if last_idx < cmd_words.len() {
                            combined[last_idx] = cmd_words[last_idx];
                        }
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
                    let mut new_sel = self.selected_suggestion.saturating_sub(1);
                    // Skip header lines
                    while new_sel > 0
                        && new_sel < self.suggestion_is_header.len()
                        && self.suggestion_is_header[new_sel]
                    {
                        new_sel = new_sel.saturating_sub(1);
                    }
                    if new_sel < self.suggestion_is_header.len()
                        && !self.suggestion_is_header[new_sel]
                    {
                        self.selected_suggestion = new_sel;
                    }
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
                        let mut new_sel = self.selected_suggestion + 1;
                        // Skip header lines
                        while new_sel < max
                            && new_sel < self.suggestion_is_header.len()
                            && self.suggestion_is_header[new_sel]
                        {
                            new_sel += 1;
                        }
                        if new_sel < self.suggestion_is_header.len()
                            && !self.suggestion_is_header[new_sel]
                        {
                            self.selected_suggestion = new_sel;
                        }
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
                        "retrieve" | "r" | "retrieve/full" | "rf" => {
                            AlgoType::Retrieve(RetrieveMode::FullPipeline)
                        }
                        "retrieve/embedding" | "re" => AlgoType::Retrieve(RetrieveMode::Embedding),
                        "retrieve/association" | "ra" => {
                            AlgoType::Retrieve(RetrieveMode::Association)
                        }
                        "consolidate" | "c" => AlgoType::Consolidate,
                        "forget" | "f" => AlgoType::Forget(ForgetMode::Pipeline),
                        _ => return Transition::ToMain,
                    }
                } else {
                    return Transition::ToMain;
                };
                Transition::ToSelectDataset(algo)
            }
            // 独立 forget 命令：`forget` 或 `f` 直接进入遗忘测试选图
            "forget" | "f" => Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Pipeline)),
            // 独立 forget 命令：`forget` 或 `f` 直接进入遗忘测试选图
            "forget" | "f" => Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Pipeline)),
            "inspect" | "i" => {
                if parts.len() > 1 {
                    let path = std::path::PathBuf::from(parts[1..].join(" "));
                    if path.exists() {
                        Transition::ToInspect(path)
                    } else {
                        Transition::ToMain
                    }
                } else {
                    Transition::ToMain
                }
            }
            "quit" | "q" => Transition::Quit,
            "help" | "h" => Transition::None,
            _ => Transition::ToMain,
        }
    }
}

#[derive(Clone, PartialEq)]
enum PathKind {
    Graph,
    Question,
}

struct FixturePath {
    path: String,
    kind: PathKind,
}

fn collect_fixture_entries() -> Vec<FixturePath> {
    static CACHE: OnceLock<Vec<FixturePath>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut entries = Vec::new();
            let fixtures_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("fixtures"));
            let Some(fixtures_dir) = fixtures_dir else {
                return entries;
            };
            if !fixtures_dir.is_dir() {
                return entries;
            }
            let mut stack = vec![fixtures_dir.clone()];
            while let Some(dir) = stack.pop() {
                if let Ok(rd) = std::fs::read_dir(&dir) {
                    for entry in rd.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            stack.push(path);
                            continue;
                        }
                        if path.extension().map(|e| e != "json").unwrap_or(true) {
                            continue;
                        }
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        // Exclude non-test-data files
                        if fname == "graph_stats.json"
                            || fname == "graph_nodes.json"
                            || fname.starts_with("raw_failed_")
                        {
                            continue;
                        }
                        let kind = if fname == "graph.json"
                            || path.to_string_lossy().contains("/graphs/")
                        {
                            PathKind::Graph
                        } else if fname == "question.json"
                            || path.to_string_lossy().contains("/queries/")
                        {
                            PathKind::Question
                        } else {
                            continue; // skip unrecognized JSON files
                        };
                        if let Ok(rel) =
                            path.strip_prefix(fixtures_dir.parent().unwrap_or(&fixtures_dir))
                        {
                            entries.push(FixturePath {
                                path: rel.to_string_lossy().replace('\\', "/"),
                                kind,
                            });
                        }
                    }
                }
            }
            entries.sort_by(|a, b| a.path.cmp(&b.path));
            entries
        })
        .iter()
        .map(|e| FixturePath {
            path: e.path.clone(),
            kind: e.kind.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立 `forget` 命令应直接进入遗忘测试选图
    #[test]
    fn test_execute_forget_command() {
        let mut cmd = CommandState::new();
        cmd.input.insert_str("forget");
        let registry = CmdRegistry::new();
        let t = cmd.handle_key(KeyEvent::from(KeyCode::Enter), &registry);
        assert!(matches!(
            t,
            Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Pipeline))
        ));
    }

    /// `test forget` 子命令也应进入遗忘测试选图
    #[test]
    fn test_execute_test_forget_command() {
        let mut cmd = CommandState::new();
        cmd.input.insert_str("test forget");
        let registry = CmdRegistry::new();
        let t = cmd.handle_key(KeyEvent::from(KeyCode::Enter), &registry);
        assert!(matches!(
            t,
            Transition::ToSelectDataset(AlgoType::Forget(ForgetMode::Pipeline))
        ));
    }

    /// 未知命令应回主菜单（不 panic）
    #[test]
    fn test_execute_unknown_command_returns_main() {
        let mut cmd = CommandState::new();
        cmd.input.insert_str("nonexistent_algo");
        let registry = CmdRegistry::new();
        let t = cmd.handle_key(KeyEvent::from(KeyCode::Enter), &registry);
        assert!(matches!(t, Transition::ToMain));
    }
}
