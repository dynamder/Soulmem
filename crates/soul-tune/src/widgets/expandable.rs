#[derive(Debug, Clone)]
pub struct ExpandableRow {
    expanded: bool,
}

impl ExpandableRow {
    pub fn new() -> Self {
        Self { expanded: false }
    }

    pub fn toggle(&mut self) {
        self.expanded = !self.expanded;
    }

    pub fn expand(&mut self) {
        self.expanded = true;
    }

    pub fn collapse(&mut self) {
        self.expanded = false;
    }

    pub fn is_expanded(&self) -> bool {
        self.expanded
    }
}

/// List of expandable rows with shared operations.
#[derive(Debug, Clone)]
pub struct ExpandableList {
    rows: Vec<ExpandableRow>,
}

impl ExpandableList {
    pub fn new(size: usize) -> Self {
        let mut rows = Vec::with_capacity(size);
        for _ in 0..size {
            rows.push(ExpandableRow::new());
        }
        Self { rows }
    }

    /// Toggle a row's expand state (Enter).
    pub fn toggle(&mut self, idx: usize) {
        if let Some(row) = self.rows.get_mut(idx) {
            row.toggle();
        }
    }

    /// Force-expand a row (Shift+move).
    pub fn expand(&mut self, idx: usize) {
        if let Some(row) = self.rows.get_mut(idx) {
            row.expand();
        }
    }

    /// Check if a row is expanded (for rendering).
    pub fn is_expanded(&self, idx: usize) -> bool {
        self.rows.get(idx).map_or(false, |r| r.is_expanded())
    }

    /// Collapse all rows (x key).
    pub fn clear_all(&mut self) {
        for row in &mut self.rows {
            row.collapse();
        }
    }

    /// Resize to a new capacity, preserving existing states.
    pub fn resize(&mut self, new_size: usize) {
        self.rows.resize_with(new_size, ExpandableRow::new);
    }
}
