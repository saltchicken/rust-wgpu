use clap::Parser;
use winit::event_loop::EventLoop;

mod app;
mod cli;
mod state;
mod vertex;

use app::App;
use cli::Args;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let event_loop = EventLoop::new()?;
    let mut app = App::new(args);
    event_loop.run_app(&mut app)?;
    Ok(())
}
