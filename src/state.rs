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

// fn create_vertices(num_vertices: u32) -> Vec<Vertex> {
//     // TODO: Proper error handling when not enough vertices
//     if num_vertices < 2 {
//         return Vec::new(); // Need at least 2 vertices to form a line
//     }
//
//     let mut vertices = Vec::with_capacity(num_vertices as usize);
//     let step = 2.0 / (num_vertices as f32 - 1.0);
//
//     for i in 0..num_vertices {
//         // Calculate the x-coordinate for the current vertex
//         let x = -1.0 + (i as f32 * step);
//         // Initialize y-coordinate to 0.0
//         vertices.push(Vertex { position: [x, 0.0] });
//     }
//
//     vertices
// }

fn create_vertices(points_per_axis: u32) -> Vec<Vertex> {
    if points_per_axis == 0 {
        return Vec::new();
    }
    // ‼️ Handle a single point
    if points_per_axis == 1 {
        return vec![Vertex {
            position: [0.0, 0.0],
        }];
    }

    // ‼️ Calculate total vertices for the grid
    let total_vertices = (points_per_axis * points_per_axis) as usize;
    let mut vertices = Vec::with_capacity(total_vertices);
    // ‼️ Calculate step size to span clip space from -1.0 to 1.0
    let step = 2.0 / (points_per_axis as f32 - 1.0);

    // ‼️ Use nested loops to generate grid points
    for j in 0..points_per_axis {
        // Current y-coordinate
        let y = -1.0 + (j as f32 * step);
        for i in 0..points_per_axis {
            // Current x-coordinate
            let x = -1.0 + (i as f32 * step);
            vertices.push(Vertex { position: [x, y] });
        }
    }
    vertices
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
    base_vertices: Vec<Vertex>,
    animated_vertices: Vec<Vertex>,
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

        let base_vertices = create_vertices(50);
        let animated_vertices = base_vertices.clone();
        let num_vertices = base_vertices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&base_vertices),
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
            base_vertices,
            animated_vertices,
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

        self.process_vertices(&time);

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
    fn process_vertices(&mut self, &time: &f32) {
        // ‼️ Use the new, clearer names
        for (base, anim) in self
            .base_vertices
            .iter()
            .zip(self.animated_vertices.iter_mut())
        {
            let x = base.position[0];
            let y = base.position[1];

            // Calculate an offset based on the original (x, y) grid position and time
            // This creates a wave that respects the initial position
            let offset = 0.1 * f32::sin(x * 10.0 + y * 10.0 + time * 2.0);

            // Apply the offset to the original y-position
            anim.position[0] = x; // Keep original x
            anim.position[1] = y + offset; // Add the wave offset to the original y
        }

        // Write the modified vertex data (self.animated_vertices) to the GPU buffer
        self.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&self.animated_vertices),
        );
    }
}
