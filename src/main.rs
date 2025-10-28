use clap::{Parser, ValueEnum};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::Window,
};

mod state;
use state::State;

mod vertex;
use vertex::create_vertex_grid;

const COMPUTE_WORKGROUP_SIZE: u32 = 256;

#[derive(ValueEnum, Clone, Debug, Default)]
enum ShaderChoice {
    #[default]
    #[value(name = "affine-rotate")]
    AffineRotate,

    #[value(name = "julia")]
    Julia,

    #[value(name = "shader")]
    Shader,

    #[value(name = "shader2")]
    Shader2,
}

impl ShaderChoice {
    fn as_path(&self) -> &'static str {
        match self {
            ShaderChoice::AffineRotate => "shaders/affine_rotate.wgsl",
            ShaderChoice::Julia => "shaders/julia.wgsl",
            ShaderChoice::Shader => "shaders/shader.wgsl",
            ShaderChoice::Shader2 => "shaders/shader2.wgsl",
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the WGSL shader file
    #[arg(short, long, default_value = "affine-rotate")]
    shader_name: ShaderChoice,

    /// Number of points on the X-axis
    #[arg(short = 'x', long, default_value_t = 50)]
    points_x: u32,

    /// Number of points on the Y-axis
    #[arg(short = 'y', long, default_value_t = 50)]
    points_y: u32,
}
struct App {
    state: Option<State>,
    args: Args,
}

impl App {
    pub fn new(args: Args) -> Self {
        Self { state: None, args }
    }
}

impl ApplicationHandler for App {
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_size = LogicalSize::new(1920, 1080);
        let window_attributes = Window::default_attributes()
            .with_title("Native WGPU App")
            .with_inner_size(window_size)
            .with_transparent(true);
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let base_grid = create_vertex_grid(self.args.points_x, self.args.points_y);
        println!("{}", self.args.shader_name.as_path());

        self.state = Some(
            pollster::block_on(State::new(
                window,
                base_grid,
                COMPUTE_WORKGROUP_SIZE,
                self.args.shader_name.as_path(),
            ))
            .unwrap(),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if it's lost or outdated
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    let size = state.window().inner_size();
                    state.resize(size.width, size.height);
                }
                Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                Err(e) => {
                    log::error!("Unable to render: {}", e);
                }
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let event_loop = EventLoop::new()?;
    let mut app = App::new(args);
    event_loop.run_app(&mut app)?;
    Ok(())
}
