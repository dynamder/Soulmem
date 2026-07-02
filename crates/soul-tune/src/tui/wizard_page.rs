use ratatui::layout::Constraint;
use ratatui::layout::Direction;
use ratatui::layout::Layout;
use ratatui::layout::Rect;
use std::rc::Rc;

pub struct WizardPage {}
pub struct WizardPageLayout {
    pub outer: Rc<[Rect]>,
    pub panel: Rc<[Rect]>,
}

impl WizardPage {
    pub fn page_layout(frame: Rect) -> WizardPageLayout {
        let outer_layout = Layout::default()
            .direction(Direction::Vertical)
            .margin(10)
            .constraints(vec![Constraint::Fill(1), Constraint::Min(5)])
            .split(frame);

        let panel_layout = Layout::default()
            .direction(Direction::Horizontal)
            .spacing(2)
            .constraints(vec![
                Constraint::Fill(1),
                Constraint::Fill(1),
                Constraint::Fill(1),
            ])
            .split(outer_layout[0]);

        WizardPageLayout {
            outer: outer_layout,
            panel: panel_layout,
        }
    }
}
