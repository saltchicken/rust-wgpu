use clap::{Parser, Subcommand, ValueEnum};
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
use vertex::{create_single_point, create_vertex_grid, Vertex};

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

    #[value(name = "particle-shader")]
    ParticleShader,
}

impl ShaderChoice {
    fn as_path(&self) -> &'static str {
        match self {
            ShaderChoice::AffineRotate => "shaders/affine_rotate.wgsl",
            ShaderChoice::Julia => "shaders/julia.wgsl",
            ShaderChoice::Shader => "shaders/shader.wgsl",
            ShaderChoice::Shader2 => "shaders/shader2.wgsl",
            ShaderChoice::ParticleShader => "shaders/particle_shader.wgsl",
        }
    }
}

#[derive(Subcommand, Debug)]
enum InputCommand {
    /// Generate a grid of points (Default)
    Grid {
        /// Number of points on the X-axis
        #[arg(short = 'x', long, default_value_t = 50)]
        points_x: u32,
        /// Number of points on the Y-axis
        #[arg(short = 'y', long, default_value_t = 50)]
        points_y: u32,
    },
    /// Generate a single point
    Point {
        /// X coordinate
        #[arg(short = 'x', long, default_value_t = 0.0)]
        x: f32,
        /// Y coordinate
        #[arg(short = 'y', long, default_value_t = 0.0)]
        y: f32,
        #[arg(short = 'n', long, default_value_t = 100000)]
        num_particles: u32,
    },
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "affine-rotate")]
    shader_name: ShaderChoice,

    #[command(subcommand)]
    command: Option<InputCommand>,
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

        // let base_grid = create_vertex_grid(self.args.points_x, self.args.points_y);
        let (base_grid, num_output_vertices) = match self.args.command.as_ref() {
            Some(InputCommand::Grid { points_x, points_y }) => {
                println!("Generating {}x{} grid", points_x, points_y);
                let grid = create_vertex_grid(*points_x, *points_y);
                // For a grid, input count == output count
                let count = grid.as_flat_vec().len() as u32;
                (grid, count)
            }
            Some(InputCommand::Point {
                x,
                y,
                num_particles,
            }) => {
                println!(
                    "Generating single point emitter with {} particles",
                    num_particles
                );
                let grid = create_single_point([*x, *y]);
                // For a point, input count (1) != output count
                (grid, *num_particles)
            }
            None => {
                println!("No input command, defaulting to 50x50 grid");
                let grid = create_vertex_grid(50, 50);
                // Default is a grid
                let count = grid.as_flat_vec().len() as u32;
                (grid, count)
            }
        };
        println!("{}", self.args.shader_name.as_path());

        self.state = Some(
            pollster::block_on(State::new(
                window,
                base_grid,
                num_output_vertices,
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
