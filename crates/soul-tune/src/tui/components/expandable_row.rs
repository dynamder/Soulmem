/// State for a single expandable row.
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
