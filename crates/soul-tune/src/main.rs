use std::time::Duration;

use color_eyre::owo_colors::colors::xterm::Black;
use ratatui::crossterm;
use ratatui::crossterm::event::{Event, KeyCode};
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Widget};
use ratatui::{DefaultTerminal, layout};

use crate::tui::wizard_page::WizardPage;

pub(crate) mod base;
pub(crate) mod cmd;
pub(crate) mod tui;
pub(crate) mod utils;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut terminal = ratatui::init();
    let result = run(&mut terminal);
    ratatui::restore();
    result
}

fn run(terminal: &mut DefaultTerminal) -> color_eyre::Result<()> {
    loop {
        terminal.draw(|frame| {
            render(frame);
        })?;
        if crossterm::event::poll(Duration::from_millis(40))? {
            match crossterm::event::read()? {
                Event::Key(key) => {
                    if key.code == KeyCode::Char('q') {
                        break Ok(());
                    }
                }
                _ => {}
            }
        }
    }
}

fn render(frame: &mut ratatui::Frame) {
    let layout = WizardPage::page_layout(frame.area());
    Block::bordered()
        .fg(Color::Cyan)
        .title("tips")
        .render(layout.outer[1], frame.buffer_mut());

    Block::bordered()
        .fg(Color::LightYellow)
        .title("检索")
        .render(layout.panel[0], frame.buffer_mut());

    Block::bordered()
        .fg(Color::LightYellow)
        .title("巩固")
        .render(layout.panel[1], frame.buffer_mut());

    Block::bordered()
        .fg(Color::LightYellow)
        .title("遗忘")
        .render(layout.panel[2], frame.buffer_mut());
}
