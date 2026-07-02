use ratatui::crossterm::{self, event};

pub enum SoulTuneEvent {
    CrossTerm(crossterm::event::Event),
}
impl From<crossterm::event::Event> for SoulTuneEvent {
    fn from(event: crossterm::event::Event) -> Self {
        Self::CrossTerm(event)
    }
}

pub trait EventHandler {
    fn handle_event(&mut self, event: SoulTuneEvent) -> Option<SoulTuneEvent>;
}
