pub(crate) mod app;
pub(crate) mod base;
pub(crate) mod cmd;
pub(crate) mod metric;
pub(crate) mod reporter;
pub(crate) mod state;
pub(crate) mod tui;
pub(crate) mod utils;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut app = app::App::new()?;
    app.run()
}
