use pipelink_audio_lib::{AudioMetadata, METADATA_SIZE};
use proclink::ShmemReader;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{iter, mem, sync::Arc, time::Instant};
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

fn create_vertices(num_vertices: u32) -> Vec<Vertex> {
    if num_vertices < 2 {
        return Vec::new();
    }
    let mut vertices = Vec::with_capacity(num_vertices as usize);
    let step = 2.0 / (num_vertices as f32 - 1.0);
    for i in 0..num_vertices {
        let x = -1.0 + (i as f32 * step);
        vertices.push(Vertex { position: [x, 0.0] });
    }
    vertices
}

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

// ‼️ --- START OF CHANGES ---
// ‼️ Helper function for linear interpolation
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}
// ‼️ --- END OF CHANGES ---

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<Vertex>, // ‼️ This is the *current* state being drawn
    num_vertices: u32,
    window: Arc<Window>,
    start_time: Instant,
    time_uniform: TimeUniform,
    time_buffer: wgpu::Buffer,
    time_bind_group: wgpu::BindGroup,
    last_update_time: Instant, // ‼️ Used for BOTH dt calculation and FPS logging
    frame_count: u32,
    msaa_sample_count: u32,
    msaa_view: wgpu::TextureView,
    reader: ShmemReader,
    fft_planner: FftPlanner<f32>,
    fft_plan: Option<(usize, Arc<dyn Fft<f32>>)>,
    complex_buffer: Vec<Complex<f32>>,

    // ‼️ --- START OF CHANGES ---
    /// Holds the "target" or "goal" vertex positions from the latest FFT.
    target_vertices: Vec<Vertex>,
    // ‼️ --- END OF CHANGES ---
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
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        if size.width > 0 && size.height > 0 {
            surface.configure(&device, &config);
        }

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
                // ‼️ Changed back to LineStrip, but PointList would also work
                topology: wgpu::PrimitiveTopology::LineStrip,
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
                alpha_to_coverage_enabled: true,
            },
            multiview: None,
            cache: None,
        });

        let vertices = create_vertices(128);
        let num_vertices = vertices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let reader = ShmemReader::new("pipelink-audio").expect("Failed to open shared memory");
        let fft_planner = FftPlanner::new();
        let fft_plan: Option<(usize, Arc<dyn Fft<f32>>)> = None;
        let complex_buffer: Vec<Complex<f32>> = Vec::new();

        // ‼️ --- START OF CHANGES ---
        let target_vertices = vertices.clone(); // Initialize target
                                                // ‼️ --- END OF CHANGES ---

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: size.width > 0 && size.height > 0,
            render_pipeline,
            vertex_buffer,
            vertices, // Current state
            num_vertices,
            window,
            start_time,
            time_uniform,
            time_buffer,
            time_bind_group,
            msaa_sample_count,
            msaa_view,
            last_update_time: Instant::now(), // For dt and FPS
            frame_count: 0,
            reader,
            fft_planner,
            fft_plan,
            complex_buffer,
            target_vertices, // Target state
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
            self.msaa_view = create_msaa_view(&self.device, &self.config, self.msaa_sample_count);
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if key == KeyCode::Escape && pressed {
            event_loop.exit();
        }
    }

    // ‼️ --- START OF CHANGES ---
    /// Processes audio data and updates the *target* vertices.
    fn process_audio_data(&mut self, data: &[u8]) {
        // ... (metadata reading, FFT planning, buffer filling - NO CHANGES HERE) ...
        if data.len() < METADATA_SIZE {
            println!(
                "[AudioReaderFFT] ⚠️ Received data is too small for metadata! Need {}, got {}",
                METADATA_SIZE,
                data.len()
            );
            return; // ‼️ Early return if data too small
        }
        let (metadata_bytes, audio_data) = data.split_at(METADATA_SIZE);
        let metadata: &AudioMetadata = bytemuck::from_bytes(metadata_bytes);
        let audio_data_len = audio_data.len();
        let expected_bytes = (metadata.n_samples_per_channel
            * metadata.n_channels
            * mem::size_of::<f32>() as u32) as usize;

        // ‼️ Removed print statements for brevity, you can add them back if needed

        if audio_data_len != expected_bytes {
            println!(
                "[AudioReaderFFT] ⚠️ WARNING: Received audio data size does not match metadata!"
            );
            // ‼️ Consider returning here if data is invalid
            // return;
        }

        if metadata.n_samples_per_channel > 0 && metadata.n_channels > 0 {
            let n_samples = metadata.n_samples_per_channel as usize;
            let n_chans_usize = metadata.n_channels as usize;
            let fft = match &mut self.fft_plan {
                Some((size, plan)) if *size == n_samples => plan,
                _ => {
                    println!(
                        "[AudioReaderFFT] ‼️ Creating new FFT plan for size {}",
                        n_samples
                    );
                    let plan = self.fft_planner.plan_fft_forward(n_samples);
                    self.fft_plan = Some((n_samples, plan));
                    &self.fft_plan.as_mut().unwrap().1
                }
            };
            self.complex_buffer.clear();
            self.complex_buffer.resize(n_samples, Complex::default());
            let audio_floats: &[f32] = bytemuck::cast_slice(audio_data);
            for (i, sample_f32) in audio_floats
                .iter()
                .step_by(n_chans_usize)
                .enumerate()
                .take(n_samples)
            {
                self.complex_buffer[i] = Complex {
                    re: *sample_f32,
                    im: 0.0,
                };
            }
            fft.process(&mut self.complex_buffer);

            let num_useful_bins = n_samples / 2;
            let num_vertices = self.vertices.len(); // Should be 128
            if num_vertices == 0 || num_useful_bins == 0 {
                return;
            }

            let db_min = -50.0;
            let db_max = 0.0;
            let normalization_factor = (n_samples / 2) as f32;
            let total_freq_range = metadata.sample_rate as f32 / 2.0;
            let desired_max_freq = 6000.0;
            let freq_ratio = (desired_max_freq / total_freq_range).min(1.0).max(0.0);
            let max_bin_to_display = ((num_useful_bins - 1) as f32 * freq_ratio).round() as usize;

            // ‼️ Update the TARGET vertices, not the main vertices
            for (i, vertex) in self.target_vertices.iter_mut().enumerate() {
                let percent = i as f32 / (num_vertices - 1) as f32;
                let bin_index = (percent * max_bin_to_display as f32).round() as usize;

                // ‼️ Ensure bin_index is within bounds (can happen due to rounding)
                let safe_bin_index = bin_index.min(num_useful_bins - 1);

                let magnitude = self.complex_buffer[safe_bin_index].norm();
                let normalized_mag = magnitude / normalization_factor;
                let db = 20.0 * (normalized_mag + 1e-9).log10();
                let scaled_db = ((db - db_min) / (db_max - db_min)).max(0.0).min(1.0);
                let y = (scaled_db * 2.0) - 1.0;
                vertex.position[1] = y;
            }
            // ‼️ REMOVED the queue.write_buffer call from here
        }
    }
    // ‼️ --- END OF CHANGES ---

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        // ‼️ --- START OF CHANGES ---
        let now = Instant::now();
        // ‼️ Calculate delta time since the last frame
        let dt = now.duration_since(self.last_update_time).as_secs_f32();

        // ‼️ 1. Read audio data and update TARGET vertices if available
        match self.reader.read() {
            Ok(Some(data)) => {
                // ‼️ This updates `self.target_vertices`
                self.process_audio_data(&data);
            }
            Ok(None) => {} // No new data, interpolation continues towards the last target
            Err(e) => {
                eprintln!("[AudioReaderFFT] ❌ Error reading: {}", e);
            }
        }

        // ‼️ 2. Interpolate CURRENT vertices towards TARGET vertices
        let lerp_speed = 10.0; // Adjust for desired smoothness (higher = faster)
        let lerp_factor = (dt * lerp_speed).min(1.0);

        for i in 0..self.num_vertices as usize {
            // ‼️ Check if target_vertices has the same length, just in case
            if i < self.target_vertices.len() {
                self.vertices[i].position[1] = lerp(
                    self.vertices[i].position[1],
                    self.target_vertices[i].position[1],
                    lerp_factor,
                );
            }
        }

        // ‼️ 3. Write the *interpolated* CURRENT vertices to the GPU buffer
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));

        // ‼️ --- END OF CHANGES ---

        // Update time uniform (for fragment shader, if needed)
        self.time_uniform.time = self.start_time.elapsed().as_secs_f32();
        self.queue.write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[self.time_uniform]),
        );

        // --- Standard Render Pass ---
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
                    view: &self.msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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

        // --- FPS Counter ---
        self.frame_count += 1;
        // let now = Instant::now(); // Already have `now` from above
        let delta_fps = now.duration_since(self.last_update_time);

        // ‼️ Update last_update_time *every* frame for correct dt calculation next frame
        self.last_update_time = now;

        // Log FPS about once per second (using delta_fps)
        if delta_fps.as_secs_f64() >= 1.0 {
            // This calculation is now slightly off because delta_fps isn't exactly 1s
            // but it's good enough for logging.
            let fps = self.frame_count as f64 / delta_fps.as_secs_f64();
            println!("FPS: {:.2}", fps);
            self.frame_count = 0;
            // Don't reset last_update_time here anymore
        }

        Ok(())
    }
}
