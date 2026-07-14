use std::path::PathBuf;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::base::{AlgoType, Transition};
use crate::tui::components::{list, status_bar};

pub(crate) enum Panel {
    FileList,
    PathInput,
}

pub(crate) struct FileEntry {
    name: String,
    path: PathBuf,
    is_dir: bool,
}

pub struct DatasetState {
    pub algo_type: AlgoType,
    pub current_dir: PathBuf,
    pub entries: Vec<FileEntry>,
    pub selected: usize,
    pub scroll: usize,
    #[allow(dead_code)]
    pub path_input: TextArea<'static>,
    pub active_panel: Panel,
    pub preview_content: Option<String>,
}

impl DatasetState {
    pub fn new(algo_type: AlgoType) -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut state = Self {
            algo_type,
            current_dir: cwd,
            entries: Vec::new(),
            selected: 0,
            scroll: 0,
            path_input: TextArea::default(),
            active_panel: Panel::FileList,
            preview_content: None,
        };
        state.refresh_dir();
        state
    }

    pub fn refresh_dir(&mut self) {
        self.entries.clear();
        let cwd = self.current_dir.clone();
        if let Some(parent) = cwd.parent() {
            self.entries.push(FileEntry {
                name: "..".into(),
                path: parent.to_path_buf(),
                is_dir: true,
            });
        }
        if let Ok(dir) = std::fs::read_dir(&self.current_dir) {
            let mut files: Vec<_> = dir
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "json")
                        .unwrap_or(false)
                        || e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                })
                .map(|e| FileEntry {
                    name: e.file_name().to_string_lossy().into_owned(),
                    path: e.path(),
                    is_dir: e.file_type().map(|t| t.is_dir()).unwrap_or(false),
                })
                .collect();
            files.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
            self.entries.extend(files);
        }
        if self.selected >= self.entries.len() {
            self.selected = self.entries.len().saturating_sub(1);
        }
        self.update_preview();
    }

    fn update_preview(&mut self) {
        if let Some(entry) = self.entries.get(self.selected) {
            if !entry.is_dir {
                if let Ok(content) = std::fs::read_to_string(&entry.path) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                        let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                        let desc = val
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let entries_count = val
                            .get("test_cases")
                            .or(val.get("entries"))
                            .and_then(|v| v.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        let format_label = if val.get("test_cases").is_some() {
                            "query"
                        } else if val.get("nodes").is_some() {
                            "graph"
                        } else {
                            "dataset"
                        };
                        self.preview_content = Some(format!(
                            "名称: {}\n描述: {}\n{}条目数: {}",
                            name, desc, format_label, entries_count
                        ));
                        return;
                    }
                }
            }
        }
        self.preview_content = None;
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
            .title(format!(" 选择数据集 · {} ", self.algo_type))
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let mid_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Fill(1), Constraint::Fill(1)])
            .split(layout[1]);

        self.render_file_list(frame, mid_layout[0]);
        self.render_preview(frame, mid_layout[1]);

        let path_text = self.current_dir.to_string_lossy().to_string();
        let path_display = format!("路径: {}", path_text);
        let path_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Fill(1)])
            .split(layout[2]);

        let block = Block::bordered().title(" 文件路径 ").fg(
            if matches!(self.active_panel, Panel::PathInput) {
                Color::Green
            } else {
                Color::Reset
            },
        );
        let inner = block.inner(path_area[0]);
        block.render(path_area[0], frame.buffer_mut());
        let path_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Fill(1)])
            .split(inner);
        frame.render_widget(Paragraph::new(path_display), path_layout[0]);

        status_bar::render_status_bar(
            frame,
            layout[3],
            &[
                ("[↑↓]".into(), "选择".into()),
                ("[Enter]".into(), "确认/进入".into()),
                ("[Esc]".into(), "返回".into()),
            ],
        );
    }

    fn render_file_list(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .title(format!(
                " {} ",
                self.current_dir
                    .file_name()
                    .unwrap_or(std::ffi::OsStr::new("/"))
                    .to_string_lossy()
            ))
            .fg(if matches!(self.active_panel, Panel::FileList) {
                Color::Green
            } else {
                Color::Reset
            });
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let items: Vec<String> = self
            .entries
            .iter()
            .map(|e| {
                if e.is_dir {
                    format!("[DIR] {}", e.name)
                } else {
                    format!("[FILE] {}", e.name)
                }
            })
            .collect();

        let visible_height = inner.height as usize;
        let scroll = if visible_height > 0 && self.selected >= visible_height {
            self.selected - (visible_height - 1)
        } else {
            0
        };
        list::render_list(frame, inner, &items, self.selected, scroll);
    }

    fn render_preview(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered().title(" 预览 ");
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        if let Some(ref preview) = self.preview_content {
            frame.render_widget(
                Paragraph::new(preview.as_str()).wrap(Wrap { trim: false }),
                inner,
            );
        } else {
            frame.render_widget(Paragraph::new("(无预览)").fg(Color::DarkGray), inner);
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Up => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.update_preview();
                }
                Transition::None
            }
            KeyCode::Down => {
                if self.selected + 1 < self.entries.len() {
                    self.selected += 1;
                    self.update_preview();
                }
                Transition::None
            }
            KeyCode::Enter => {
                if let Some(entry) = self.entries.get(self.selected) {
                    if entry.is_dir {
                        self.current_dir = entry.path.clone();
                        self.selected = 0;
                        self.scroll = 0;
                        self.refresh_dir();
                        Transition::None
                    } else {
                        Transition::ToConfigParams(self.algo_type, entry.path.clone())
                    }
                } else {
                    Transition::None
                }
            }
            KeyCode::Tab => {
                self.active_panel = match self.active_panel {
                    Panel::FileList => Panel::PathInput,
                    Panel::PathInput => Panel::FileList,
                };
                Transition::None
            }
            _ => Transition::None,
        }
    }
}
