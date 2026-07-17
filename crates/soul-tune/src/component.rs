use ratatui::crossterm::event::{KeyEvent, MouseEvent};
use ratatui::Frame;

use crate::base::Transition;

pub enum ComponentEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Tick,
}

pub trait Component {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition;
    fn view(&self, frame: &mut Frame);
}
