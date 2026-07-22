use ratatui::layout::Rect;
use ratatui::widgets::{Scrollbar, ScrollbarOrientation, ScrollbarState};
use ratatui::Frame;

/// Cursor-based scroll state with auto-offset computation and scrollbar rendering.
/// `cursor` is used for keyboard navigation (which item is selected).
/// `offset` is used for free-hand scrolling (mouse wheel on paragraphs).
#[derive(Debug, Clone)]
pub struct ScrollContainer {
    pub cursor: usize,
    pub offset: usize,
}

impl ScrollContainer {
    pub fn new() -> Self {
        Self {
            cursor: 0,
            offset: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
        self.offset = 0;
    }

    pub fn move_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn move_down(&mut self, max: usize) {
        if max > 0 && self.cursor + 1 < max {
            self.cursor += 1;
        }
    }

    pub fn move_to(&mut self, idx: usize) {
        self.cursor = idx;
    }

    pub fn clamp_cursor(&mut self, max: usize) {
        if max == 0 {
            self.cursor = 0;
        } else if self.cursor >= max {
            self.cursor = max.saturating_sub(1);
        }
    }

    pub fn scroll_up(&mut self) {
        self.offset = self.offset.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        self.offset = self.offset.saturating_add(1);
    }

    pub fn clamp_offset(&mut self, max_offset: usize) {
        self.offset = self.offset.min(max_offset);
    }

    /// Compute scroll offset to keep cursor visible.
    /// Pure function — called in render(&self).
    pub fn offset(visible: u16, total: usize, cursor: usize) -> usize {
        let v = visible as usize;
        if v == 0 || total <= v {
            return 0;
        }
        if cursor >= v {
            cursor - (v - 1)
        } else {
            0
        }
    }

    /// Split area into content (left, width-1) and scrollbar column (right, width=1).
    pub fn split_area(area: Rect) -> (Rect, Rect) {
        let content_w = area.width.saturating_sub(1);
        let content = Rect::new(area.x, area.y, content_w, area.height);
        let bar = Rect::new(area.x + content_w, area.y, 1, area.height);
        (content, bar)
    }

    /// Render vertical scrollbar in the given column.
    pub fn render_scrollbar(
        frame: &mut Frame,
        area: Rect,
        content_length: usize,
        viewport: u16,
        scroll_offset: usize,
    ) {
        let mut state = ScrollbarState::new(content_length)
            .position(scroll_offset)
            .viewport_content_length(viewport as usize);
        let widget = Scrollbar::new(ScrollbarOrientation::VerticalRight);
        frame.render_stateful_widget(widget, area, &mut state);
    }
}

impl Default for ScrollContainer {
    fn default() -> Self {
        Self::new()
    }
}
