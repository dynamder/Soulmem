use ratatui::style::{Color, Style, Stylize};

pub fn header() -> Style {
    Style::new().yellow().bold()
}

pub fn pass() -> Style {
    Style::new().green()
}

pub fn fail() -> Style {
    Style::new().red()
}

pub fn muted() -> Style {
    Style::new().dark_gray()
}

pub fn cursor_style() -> Style {
    Style::new().yellow().bold()
}

pub fn bg_cursor() -> Style {
    Style::default().bg(Color::DarkGray)
}
