use std::path::{Path, PathBuf};

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use ratatui::crossterm::event::KeyModifiers;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::eval::loader::clear_embedding_cache;
use crate::tui::components::scroll::ScrollState;
use crate::tui::components::scroll_container::ScrollContainer;
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
    pub algo_type: crate::base::AlgoType,
    pub current_dir: PathBuf,
    pub entries: Vec<FileEntry>,
    pub list_scroll: ScrollContainer,
    #[allow(dead_code)]
    pub path_input: TextArea<'static>,
    pub active_panel: Panel,
    pub preview_content: Option<String>,
    pub batch_mode: bool,
    pub inspect_mode: bool,
}

impl DatasetState {
    pub fn new(algo_type: crate::base::AlgoType) -> Self {
        Self::with_dir(algo_type, false)
    }

    pub fn new_compare() -> Self {
        Self::with_dir(crate::base::AlgoType::Compare, false)
    }

    pub fn new_batch() -> Self {
        Self::with_dir(
            crate::base::AlgoType::Retrieve(crate::base::RetrieveMode::Embedding),
            true,
        )
    }

    pub fn new_inspect() -> Self {
        let mut state = Self::with_dir(
            crate::base::AlgoType::Retrieve(crate::base::RetrieveMode::Embedding),
            false,
        );
        state.inspect_mode = true;
        state
    }

    fn with_dir(algo_type: crate::base::AlgoType, batch_mode: bool) -> Self {
        let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("fixtures"))
            .filter(|p| p.is_dir());
        let cwd = if batch_mode {
            fixtures_dir
                .clone()
                .map(|d| d.join("example_data/test_batch_output-serde-fix"))
                .filter(|p| p.is_dir())
                .unwrap_or_else(|| {
                    fixtures_dir.unwrap_or_else(|| {
                        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                    })
                })
        } else {
            fixtures_dir
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        };
        let mut state = Self {
            algo_type,
            current_dir: cwd,
            entries: Vec::new(),
            list_scroll: ScrollContainer::new(),
            path_input: TextArea::default(),
            active_panel: Panel::FileList,
            preview_content: None,
            batch_mode,
            inspect_mode: false,
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
                    let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    if is_dir {
                        return true;
                    }
                    if self.batch_mode {
                        return false;
                    }
                    let path = e.path();
                    let is_json = path.extension().map(|ext| ext == "json").unwrap_or(false);
                    if !is_json {
                        return false;
                    }
                    let file_name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if file_name == "question.json" {
                        return true;
                    }
                    if path
                        .parent()
                        .and_then(|p| p.file_name())
                        .map(|n| n == "queries")
                        .unwrap_or(false)
                    {
                        return true;
                    }
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                            return val.get("test_cases").is_some()
                                || val.get("nodes").is_some()
                                || (val.get("graph_path").is_some()
                                    && val.get("conversations").is_some());
                        }
                    }
                    false
                })
                .map(|e| {
                    let base = e.file_name().to_string_lossy().into_owned();
                    let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    let name = if is_dir {
                        base
                    } else {
                        let dir_name = self
                            .current_dir
                            .file_name()
                            .map(|n| n.to_string_lossy())
                            .unwrap_or_else(|| "?".into());
                        format!("{}/{}", dir_name, base)
                    };
                    FileEntry {
                        name,
                        path: e.path(),
                        is_dir,
                    }
                })
                .collect();
            files.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
            self.entries.extend(files);
        }
        self.list_scroll.clamp_cursor(self.entries.len());
        self.update_preview();
    }

    fn update_preview(&mut self) {
        if let Some(entry) = self.entries.get(self.list_scroll.cursor) {
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

        let title = if self.inspect_mode {
            " 选择检视文件 — Enter检视选中文件 ".to_string()
        } else if self.batch_mode {
            " 选择批量目录 — Enter选择目录批量运行 ".to_string()
        } else {
            format!(" 选择数据集 · {} ", self.algo_type)
        };
        Block::bordered()
            .title(title)
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

        let hints: Vec<(String, String)> = if self.inspect_mode {
            vec![
                ("[↑↓]".into(), "选择".into()),
                ("[Enter]".into(), "检视".into()),
                ("[Esc]".into(), "返回".into()),
            ]
        } else {
            vec![
                ("[↑↓]".into(), "选择".into()),
                ("[Enter]".into(), "确认/进入".into()),
                ("[I]".into(), "检视".into()),
                ("[Shift+C]".into(), "清缓存".into()),
                ("[Esc]".into(), "返回".into()),
            ]
        };
        status_bar::render_status_bar(frame, layout[3], &hints);
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

        let (content_rect, bar_rect) = ScrollContainer::split_area(inner);
        let offset =
            ScrollContainer::offset(content_rect.height, items.len(), self.list_scroll.cursor);
        let s = ScrollState {
            cursor: self.list_scroll.cursor,
            offset,
        };
        list::render_simple_list(frame, content_rect, &items, &s);
        ScrollContainer::render_scrollbar(
            frame,
            bar_rect,
            items.len(),
            content_rect.height,
            offset,
        );
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
            KeyCode::Char('i') | KeyCode::Char('I') => {
                if let Some(entry) = self.entries.get(self.list_scroll.cursor) {
                    if !entry.is_dir {
                        return Transition::ToInspect(entry.path.clone());
                    }
                }
                Transition::None
            }
            KeyCode::Char('C') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                clear_embedding_cache(&self.current_dir);
                Transition::None
            }
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Up => {
                self.list_scroll.move_up();
                self.update_preview();
                Transition::None
            }
            KeyCode::Down => {
                self.list_scroll.move_down(self.entries.len());
                self.update_preview();
                Transition::None
            }
            KeyCode::Enter => {
                let selected = self.list_scroll.cursor;
                if let Some(entry) = self.entries.get(selected) {
                    if entry.is_dir && entry.name != ".." && self.batch_mode {
                        Transition::ToBatchModeSelect(entry.path.clone())
                    } else if entry.is_dir {
                        self.current_dir = entry.path.clone();
                        self.list_scroll.reset();
                        self.refresh_dir();
                        Transition::None
                    } else if self.inspect_mode {
                        Transition::ToInspect(entry.path.clone())
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

impl Component for DatasetState {
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
