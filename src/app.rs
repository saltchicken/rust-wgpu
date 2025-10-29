use std::sync::Arc;
use winit::{
    application::ApplicationHandler, dpi::LogicalSize, event::*, event_loop::ActiveEventLoop,
    keyboard::PhysicalKey, window::Window,
};

use crate::cli::{Args, InputCommand};
use crate::state::State;
use crate::vertex::{create_single_point, create_vertex_grid};

pub struct App {
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

        let (base_grid, num_output_vertices) = match self.args.command.as_ref() {
            Some(InputCommand::Grid {
                points_x,
                points_y,
                step_x,
                step_y,
            }) => {
                println!("Generating {}x{} grid", points_x, points_y);
                let grid = create_vertex_grid(*points_x, *points_y, *step_x, *step_y);
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
                (grid, *num_particles)
            }
            None => {
                println!("No input command, defaulting to 50x50 grid");
                let grid = create_vertex_grid(50, 50, 0.1, 0.1);
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
