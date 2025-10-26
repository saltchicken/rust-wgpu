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

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<Vertex>,
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
    reader: ShmemReader,
    fft_planner: FftPlanner<f32>,
    fft_plan: Option<(usize, Arc<dyn Fft<f32>>)>,
    complex_buffer: Vec<Complex<f32>>,
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

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: size.width > 0 && size.height > 0,
            render_pipeline,
            vertex_buffer,
            vertices,
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
            reader,
            fft_planner,
            fft_plan,
            complex_buffer,
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

    fn process_audio_data(&mut self, data: &[u8]) {
        if data.len() < METADATA_SIZE {
            println!(
                "[AudioReaderFFT] ⚠️ Received data is too small for metadata! Need {}, got {}",
                METADATA_SIZE,
                data.len()
            );
        }

        let (metadata_bytes, audio_data) = data.split_at(METADATA_SIZE);
        let metadata: &AudioMetadata = bytemuck::from_bytes(metadata_bytes);
        let audio_data_len = audio_data.len();
        let expected_bytes = (metadata.n_samples_per_channel
            * metadata.n_channels
            * mem::size_of::<f32>() as u32) as usize;
        let num_floats_received = audio_data_len / mem::size_of::<f32>();

        println!("[AudioReaderFFT] ✅ Read {} bytes total.", data.len());
        println!("  Sample Rate: {} Hz", metadata.sample_rate);
        println!("  Channels: {}", metadata.n_channels);
        println!("  Samples per Channel: {}", metadata.n_samples_per_channel);
        println!(
            "  Audio Data Bytes: {} (Expected: {})",
            audio_data_len, expected_bytes
        );
        println!("  Total Floats Received: {}\n", num_floats_received);

        if audio_data_len != expected_bytes {
            println!(
                "[AudioReaderFFT] ⚠️ WARNING: Received audio data size does not match metadata!"
            );
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

            let num_useful_bins = n_samples / 2; // 512 bins

            let (peak_bin_index, peak_magnitude) = self.complex_buffer[..num_useful_bins]
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.norm()))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));

            let bin_width = metadata.sample_rate / (n_samples as f32);
            let peak_frequency = peak_bin_index as f32 * bin_width;

            println!(
                "  [FFT] ‼️ Peak Frequency (Ch 0): {:.2} Hz (Magnitude: {:.2})\n",
                peak_frequency, peak_magnitude
            );

            let num_vertices = self.vertices.len(); // 512 vertices
            if num_vertices == 0 || num_useful_bins == 0 {
                return;
            }

            let db_min = -50.0;
            let db_max = 0.0;
            let normalization_factor = (n_samples / 2) as f32;

            let total_freq_range = metadata.sample_rate as f32 / 2.0;
            // ‼️ --- CHANGE HERE ---
            let desired_max_freq = 6000.0; // ‼️ Your desired 6kHz limit
                                           // ‼️ --- END OF CHANGE ---

            let freq_ratio = (desired_max_freq / total_freq_range).min(1.0).max(0.0);
            let max_bin_to_display = ((num_useful_bins - 1) as f32 * freq_ratio).round() as usize;

            println!(
                "  [VIS] ‼️ Mapping {} vertices to FFT bins [0, {}] (0 Hz to {:.2} Hz)",
                num_vertices, max_bin_to_display, desired_max_freq
            );

            for (i, vertex) in self.vertices.iter_mut().enumerate() {
                let percent = i as f32 / (num_vertices - 1) as f32; // 0.0 to 1.0
                let bin_index = (percent * max_bin_to_display as f32).round() as usize;
                let magnitude = self.complex_buffer[bin_index].norm();

                let normalized_mag = magnitude / normalization_factor;
                let db = 20.0 * (normalized_mag + 1e-9).log10();
                let scaled_db = ((db - db_min) / (db_max - db_min)).max(0.0).min(1.0);
                let y = (scaled_db * 2.0) - 1.0;

                vertex.position[1] = y;
            }

            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        match self.reader.read() {
            Ok(Some(data)) => {
                self.process_audio_data(&data);
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("[AudioReaderFFT] ❌ Error reading: {}", e);
            }
        }

        self.time_uniform.time = self.start_time.elapsed().as_secs_f32();
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

        self.frame_count += 1;
        let now = Instant::now();
        let delta = now.duration_since(self.last_update_time);
        if delta.as_secs_f64() >= 1.0 {
            let fps = self.frame_count as f64 / delta.as_secs_f64();
            println!("FPS: {:.2}", fps);
            self.frame_count = 0;
            self.last_update_time = now;
        }

        Ok(())
    }
}
