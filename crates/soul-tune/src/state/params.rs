use std::collections::HashMap;
use std::path::PathBuf;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::Block;
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::base::{AlgoType, TestConfig, Transition};
use crate::component::{Component, ComponentEvent};
use crate::tui::components::scroll::ScrollState;
use crate::tui::components::scroll_container::ScrollContainer;
use crate::tui::components::{editable_table, status_bar};

pub struct ParamRow {
    pub name: String,
    pub value: String,
    pub description: String,
}

pub struct ParamState {
    pub algo_type: AlgoType,
    pub dataset_path: PathBuf,
    pub rows: Vec<ParamRow>,
    pub selected: usize,
    pub editing: Option<usize>,
    pub table_scroll: ScrollContainer,
    pub textareas: Vec<TextArea<'static>>,
}

impl ParamState {
    pub fn new(algo_type: AlgoType, dataset_path: PathBuf) -> Self {
        let default_rows = vec![
            ParamRow {
                name: "top_k".into(),
                value: "10".into(),
                description: "最大返回数量".into(),
            },
            ParamRow {
                name: "threshold".into(),
                value: "0.7".into(),
                description: "相似度阈值".into(),
            },
            ParamRow {
                name: "damping".into(),
                value: "0.85".into(),
                description: "PPR 阻尼因子".into(),
            },
            ParamRow {
                name: "iterations".into(),
                value: "20".into(),
                description: "迭代次数".into(),
            },
        ];
        let textareas: Vec<TextArea> = default_rows.iter().map(|_| TextArea::default()).collect();
        Self {
            algo_type,
            dataset_path,
            rows: default_rows,
            selected: 0,
            editing: None,
            table_scroll: ScrollContainer::new(),
            textareas,
        }
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
            .title(format!(
                " 配置参数 · {} · {} ",
                self.algo_type,
                self.dataset_path
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ))
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let inner = layout[1];
        let (content_rect, bar_rect) = ScrollContainer::split_area(inner);
        let offset = ScrollContainer::offset(
            content_rect.height.saturating_sub(1),
            self.rows.len(),
            self.table_scroll.cursor,
        );
        let scroll_state = ScrollState {
            cursor: self.table_scroll.cursor,
            offset,
        };
        editable_table::render_editable_table(
            frame,
            content_rect,
            &self.rows,
            &self.textareas,
            self.selected,
            self.editing,
            &scroll_state,
        );
        ScrollContainer::render_scrollbar(
            frame,
            bar_rect,
            self.rows.len(),
            content_rect.height.saturating_sub(1),
            offset,
        );

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[↑↓]".into(), "选择".into()),
                ("[Enter]".into(), "编辑".into()),
                ("[Ctrl+Enter]".into(), "运行".into()),
                ("[Esc]".into(), "返回".into()),
            ],
        );
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        if let Some(editing_idx) = self.editing {
            match key.code {
                KeyCode::Enter => {
                    let value = self.textareas[editing_idx]
                        .lines()
                        .first()
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    self.rows[editing_idx].value = value;
                    self.textareas[editing_idx] = TextArea::default();
                    self.editing = None;
                }
                KeyCode::Esc => {
                    self.textareas[editing_idx] = TextArea::default();
                    self.editing = None;
                }
                _ => {
                    self.textareas[editing_idx].input(key);
                }
            }
            return Transition::None;
        }

        match key.code {
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Up => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                Transition::None
            }
            KeyCode::Down => {
                if self.selected + 1 < self.rows.len() {
                    self.selected += 1;
                }
                Transition::None
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let params: HashMap<String, String> = self
                    .rows
                    .iter()
                    .map(|r| (r.name.clone(), r.value.clone()))
                    .collect();
                Transition::ToTestRunning(TestConfig {
                    algo: self.algo_type,
                    dataset_path: self.dataset_path.clone(),
                    params,
                })
            }
            KeyCode::Enter => {
                self.editing = Some(self.selected);
                Transition::None
            }
            KeyCode::Tab => {
                let next = (self.selected + 1).min(self.rows.len().saturating_sub(1));
                self.selected = next;
                Transition::None
            }
            _ => Transition::None,
        }
    }
}

impl Component for ParamState {
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
