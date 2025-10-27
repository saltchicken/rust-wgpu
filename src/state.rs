use glam::{Mat4, Vec4};
use std::{iter, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

// ‼️ New Grid struct
#[derive(Debug)]
pub struct Grid<T> {
    data: Vec<T>,
    width: u32,
    height: u32,
}

impl<T> Grid<T> {
    /// Get an immutable reference to an item in the grid
    pub fn get(&self, x: u32, y: u32) -> Option<&T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get((y * self.width + x) as usize)
    }

    /// Get a mutable reference to an item in the grid
    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get_mut((y * self.width + x) as usize)
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get a reference to the internal flat vector
    pub fn as_flat_vec(&self) -> &Vec<T> {
        &self.data
    }
}

// ‼️ Clone implementation for our Grid (needed in State::new)
impl<T: Clone> Clone for Grid<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
        }
    }
}

fn create_vertex_grid(points_x: u32, points_y: u32) -> Grid<Vertex> {
    if points_x == 0 || points_y == 0 {
        // Return an empty grid
        return Grid {
            data: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    // ‼️ Define the buffer size.
    // ‼️ sqrt(2) is the distance from the center (0,0) to the corner (1,1).

    let additional_buffer = 0.1;
    let grid_buffer = 2.0_f32.sqrt(); // approx 1.414
                                      //
    let buffer = grid_buffer + additional_buffer;

    // ‼️ The total width/height of the grid will be from -buffer to +buffer.
    let grid_range = buffer * 2.0;

    // Calculate total vertices for the grid
    let total_vertices = (points_x * points_y) as usize;
    let mut vertices = Vec::with_capacity(total_vertices);

    // Calculate step size for each axis independently
    // If only 1 point, step is 0 (it will be centered)
    let step_x = if points_x > 1 {
        grid_range / (points_x - 1) as f32 // ‼️ Use new grid_range
    } else {
        0.0
    };
    let step_y = if points_y > 1 {
        grid_range / (points_y - 1) as f32 // ‼️ Use new grid_range
    } else {
        0.0
    };

    for j in 0..points_y {
        // Current y-coordinate
        // If 1 point, center it at 0.0, otherwise map from -buffer to +buffer
        let y = if points_y == 1 {
            0.0
        } else {
            -buffer + (j as f32 * step_y) // ‼️ Start at -buffer
        };

        for i in 0..points_x {
            // Current x-coordinate
            // If 1 point, center it at 0.0, otherwise map from -buffer to +buffer
            let x = if points_x == 1 {
                0.0
            } else {
                -buffer + (i as f32 * step_x) // ‼️ Start at -buffer
            };
            vertices.push(Vertex { position: [x, y] });
        }
    }

    // Return the populated Grid struct
    Grid {
        data: vertices,
        width: points_x,
        height: points_y,
    }
}
// Helper function to create the MSAA texture view
fn create_msaa_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> wgpu::TextureView {
    let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Framebuffer"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    // ‼️ Store vertices in Grid structs
    animated_grid: Grid<Vertex>, // ‼️ This is mutated and its flat data sent to GPU
    base_grid: Grid<Vertex>,     // ‼️ This is the static, original grid
    num_vertices: u32,
    window: Arc<Window>,
    start_time: Instant,
    time_uniform: TimeUniform,
    time_buffer: wgpu::Buffer,
    time_bind_group: wgpu::BindGroup,
    last_update_time: Instant,
    frame_count: u32,
    msaa_sample_count: u32,
    msaa_view: wgpu::TextureView,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);
        let desired_modes = [
            wgpu::CompositeAlphaMode::PostMultiplied,
            wgpu::CompositeAlphaMode::Auto,
            wgpu::CompositeAlphaMode::Inherit,
        ];
        let alpha_mode = desired_modes
            .into_iter()
            .find(|mode| surface_caps.alpha_modes.contains(mode))
            .unwrap_or_else(|| {
                surface_caps
                    .alpha_modes
                    .iter()
                    .copied()
                    .find(|mode| *mode != wgpu::CompositeAlphaMode::Opaque)
                    .unwrap_or(surface_caps.alpha_modes[0])
            });

        if alpha_mode == wgpu::CompositeAlphaMode::Opaque {
            log::warn!("Surface does not support transparency, falling back to Opaque.");
        }

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            // present_mode,
            present_mode: surface_caps.present_modes[0],
            // alpha_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        if size.width > 0 && size.height > 0 {
            surface.configure(&device, &config);
        }

        // Set sample count and create the initial MSAA view
        let msaa_sample_count = 4;
        let msaa_view = create_msaa_view(&device, &config, msaa_sample_count);

        let start_time = Instant::now();
        let time_uniform = TimeUniform { time: 0.0 };
        let time_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Time Buffer"),
            contents: bytemuck::cast_slice(&[time_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let time_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("time_bind_group_layout"),
            });

        let time_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &time_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: time_buffer.as_entire_binding(),
            }],
            label: Some("time_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&time_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: true, // Improves line anti-aliasing
            },
            multiview: None,
            cache: None,
        });

        // ‼️ Create the base grid
        let base_grid = create_vertex_grid(75, 200);
        // ‼️ Create the mutable grid by cloning the base
        let animated_grid = base_grid.clone();
        // ‼️ Get the count from the grid's flat data
        let num_vertices = animated_grid.as_flat_vec().len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            // ‼️ Use the flat Vec from the animated_grid
            contents: bytemuck::cast_slice(animated_grid.as_flat_vec()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: size.width > 0 && size.height > 0,
            render_pipeline,
            vertex_buffer,
            animated_grid, // ‼️
            base_grid,     // ‼️
            num_vertices,
            window,
            start_time,
            time_uniform,
            time_buffer,
            time_bind_group,
            msaa_sample_count,
            msaa_view,
            last_update_time: Instant::now(),
            frame_count: 0,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.is_surface_configured = true;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            // Recreate the MSAA view with the new size
            self.msaa_view = create_msaa_view(&self.device, &self.config, self.msaa_sample_count);
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if key == KeyCode::Escape && pressed {
            event_loop.exit();
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        // Update the time uniform before rendering
        self.time_uniform.time = self.start_time.elapsed().as_secs_f32();
        let time = self.time_uniform.time;

        self.process_vertices(&time); // Enable vertex processing

        self.queue.write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[self.time_uniform]),
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Render to the multisampled texture view
                    view: &self.msaa_view,
                    // Resolve to the swapchain texture view
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        // Discard the multisampled texture's contents after resolving
                        store: wgpu::StoreOp::Discard,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.time_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        self.frame_count += 1;
        let now = Instant::now();
        let delta = now.duration_since(self.last_update_time);

        // Update and log FPS every 1 second
        if delta.as_secs_f64() >= 1.0 {
            let fps = self.frame_count as f64 / delta.as_secs_f64();
            println!("FPS: {:.2}", fps);
            // Reset for the next interval
            self.frame_count = 0;
            self.last_update_time = now;
        }

        Ok(())
    }

    // ‼️ Updated function to use the Grid
    fn process_vertices(&mut self, &time: &f32) {
        // ‼️ 1. Create a rotation matrix (rotating around Z-axis)
        // We use Mat4 (4x4 matrix) as it's standard for 3D/GPU math
        let rotation = Mat4::from_rotation_z(time * 0.5);

        for y in 0..self.base_grid.height() {
            for x in 0..self.base_grid.width() {
                let base_vertex = self.base_grid.get(x, y).unwrap();
                let base_x = base_vertex.position[0];
                let base_y = base_vertex.position[1];

                // ‼️ 2. Convert 2D position to Vec4
                // (x, y, z, w) - w=1.0 is crucial for translation/transformation
                let base_pos_vec4 = Vec4::new(base_x, base_y, 0.0, 1.0);

                // ‼️ 3. Apply the matrix transform
                let rotated_pos = rotation * base_pos_vec4;

                // ‼️ 4. Calculate the sine wave offset
                // We use the *rotated* x/y to make the wave pattern move with the grid
                let offset =
                    0.1 * f32::sin(rotated_pos.x * 10.0 + rotated_pos.y * 10.0 + time * 2.0);

                // ‼️ 5. Get the mutable vertex and set its new position
                let anim_vertex = self.animated_grid.get_mut(x, y).unwrap();

                // We take the transformed x/y and add the offset to the y
                anim_vertex.position[0] = rotated_pos.x;
                anim_vertex.position[1] = rotated_pos.y + offset;
            }
        }

        // ‼️ Write the animated_grid's *flat data* to the GPU buffer
        self.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(self.animated_grid.as_flat_vec()),
        );
    }
}
