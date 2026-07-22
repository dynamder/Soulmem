use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyEvent, KeyEventKind, MouseEvent};
use ratatui::crossterm::execute;
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};

use super::{App, AppState};

impl App {
    pub fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            let state = &self.app_state;
            self.terminal
                .draw(|frame| Self::render_state(frame, state))?;

            self.tick_running();

            if event::poll(Duration::from_millis(40))? {
                match event::read()? {
                    Event::Key(key) => {
                        if self.handle_key(key) {
                            break;
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse(mouse);
                    }
                    _ => {}
                }
            }
        }
        let _ = execute!(
            std::io::stdout(),
            ratatui::crossterm::event::DisableMouseCapture
        );
        ratatui::restore();
        Ok(())
    }

    fn tick_running(&mut self) {
        let t = match &mut self.app_state {
            AppState::TestRunning(s) => s.handle_event(ComponentEvent::Tick),
            AppState::BatchRunning(s) => s.handle_event(ComponentEvent::Tick),
            AppState::PlayTestRun(s) => s.handle_event(ComponentEvent::Tick),
            _ => Transition::None,
        };
        self.apply(t);
    }

    fn render_state(frame: &mut Frame, state: &AppState) {
        match state {
            AppState::Main(s) => Component::view(s, frame),
            AppState::CommandMode(s) => s.render(frame),
            AppState::SelectDataset(s) => s.view(frame),
            AppState::SelectBatchDir(s) => s.view(frame),
            AppState::RetrieveModeSelect(s) => s.view(frame),
            AppState::SelectAlgo(s) => s.view(frame),
            AppState::ConfigParams(s) => s.view(frame),
            AppState::TestRunning(s) => s.view(frame),
            AppState::TestResults(s) => s.view(frame),
            AppState::CompareResults(s) => s.view(frame),
            AppState::BatchModeSelect(s) => s.view(frame),
            AppState::BatchRunning(s) => s.view(frame),
            AppState::InspectData(s) => s.view(frame),
            AppState::PlayTestSelect(s) => s.view(frame),
            AppState::PlayTestInput(s) => s.view(frame),
            AppState::PlayTestRun(s) => s.view(frame),
            AppState::PlayTestJudge(s) => s.view(frame),
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        let transition = match &mut self.app_state {
            AppState::Main(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::RetrieveModeSelect(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::SelectAlgo(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::TestResults(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::CompareResults(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::BatchModeSelect(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::CommandMode(s) => s.handle_key(key, &self.cmd_registry),
            AppState::SelectDataset(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::SelectBatchDir(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::ConfigParams(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::TestRunning(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::BatchRunning(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::InspectData(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::PlayTestSelect(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::PlayTestInput(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::PlayTestRun(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::PlayTestJudge(s) => s.handle_event(ComponentEvent::Key(key)),
        };
        self.apply(transition)
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        match &mut self.app_state {
            AppState::TestResults(s) => {
                s.handle_event(ComponentEvent::Mouse(mouse));
            }
            AppState::CompareResults(s) => {
                s.handle_event(ComponentEvent::Mouse(mouse));
            }
            AppState::BatchRunning(s) => {
                s.handle_event(ComponentEvent::Mouse(mouse));
            }
            AppState::PlayTestJudge(s) => {
                s.handle_event(ComponentEvent::Mouse(mouse));
            }
            _ => {}
        }
    }
}
