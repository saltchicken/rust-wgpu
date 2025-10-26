use std::{iter, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

// ‼️ Renamed from PersistenceUniform
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DenoiseUniform {
    factor: f32, // ‼️ This will now be the denoise_factor itself
}

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

// Helper function to create the MSAA texture view
fn create_msaa_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> wgpu::TextureView {
    create_msaa_view_with_format(
        device,
        config.width,
        config.height,
        config.format,
        sample_count,
    )
}

fn create_feedback_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat, // ‼️ Accept format as an argument
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Feedback Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_msaa_view_with_format(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::TextureView {
    let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Framebuffer"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
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
    render_pipeline: wgpu::RenderPipeline, // ‼️ This is now the "points" pipeline
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

    // ‼️ --- Fields for temporal accumulation ---
    denoise_factor: f32,
    denoise_uniform: DenoiseUniform,     // ‼️ Renamed
    denoise_buffer: wgpu::Buffer,        // ‼️ Renamed
    denoise_bind_group: wgpu::BindGroup, // ‼️ Renamed

    feedback_textures: [wgpu::Texture; 2],
    feedback_texture_views: [wgpu::TextureView; 2],

    // ‼️ This BGL is for *reading* a single texture (used by composite pass)
    texture_read_bind_group_layout: wgpu::BindGroupLayout,
    // ‼️ These BGs read from the feedback textures (for the composite pass)
    feedback_read_bind_groups: [wgpu::BindGroup; 2],

    // ‼️ NEW texture to hold the newly drawn points
    points_texture: wgpu::Texture,
    points_texture_view: wgpu::TextureView,

    sampler: wgpu::Sampler,

    feedback_pipeline: wgpu::RenderPipeline, // ‼️ This is now the "mix" pipeline

    composite_pipeline: wgpu::RenderPipeline,

    points_msaa_view: wgpu::TextureView,

    frame_index: usize,

    feedback_texture_format: wgpu::TextureFormat,

    // ‼️ NEW BGL and BGs for the mix pass
    feedback_mix_bgl: wgpu::BindGroupLayout,
    feedback_mix_bind_groups: [wgpu::BindGroup; 2],
    // ‼️ --- End of new fields ---
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
            .request_device(&wgpu::DeviceDescriptor::default())
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

        let feedback_texture_format = wgpu::TextureFormat::Rgba16Float;

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

        // ‼️ --- Setup for Feedback Effect ---

        // ‼️ 1. Create denoise uniform
        let denoise_factor = 0.1;
        let denoise_uniform = DenoiseUniform {
            factor: denoise_factor,
        };
        let denoise_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Denoise Buffer"), // ‼️ Renamed
            contents: bytemuck::cast_slice(&[denoise_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let denoise_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Denoise BGL"), // ‼️ Renamed
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
            });
        let denoise_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Denoise BG"), // ‼️ Renamed
            layout: &denoise_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: denoise_buffer.as_entire_binding(),
            }],
        });

        // ‼️ 2. Create ping-pong textures, points texture, and sampler
        let (ping_texture, ping_texture_view) = create_feedback_texture(
            &device,
            config.width,
            config.height,
            feedback_texture_format,
        );
        let (pong_texture, pong_texture_view) = create_feedback_texture(
            &device,
            config.width,
            config.height,
            feedback_texture_format,
        );
        let feedback_textures = [ping_texture, pong_texture];
        let feedback_texture_views = [ping_texture_view, pong_texture_view];

        // ‼️ NEW points texture
        let (points_texture, points_texture_view) = create_feedback_texture(
            &device,
            config.width,
            config.height,
            feedback_texture_format,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Feedback Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ‼️ 3. Create BGL for reading a single texture (for composite pass)
        let texture_read_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Read BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let feedback_read_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Read BG 0"),
            layout: &texture_read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&feedback_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let feedback_read_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Read BG 1"),
            layout: &texture_read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&feedback_texture_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let feedback_read_bind_groups = [feedback_read_bind_group_0, feedback_read_bind_group_1];

        // ‼️ 4. Create BGL and BGs for the *mix* pass
        let feedback_mix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Feedback Mix BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // t_old
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // s_old
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // t_points
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // ‼️ Create two mix BGs, one for reading from feedback[0] and one from feedback[1]
        let feedback_mix_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Mix BG 0"),
            layout: &feedback_mix_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    // t_old = feedback_textures[0]
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&feedback_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    // t_points
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&points_texture_view),
                },
            ],
        });
        let feedback_mix_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Mix BG 1"),
            layout: &feedback_mix_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    // t_old = feedback_textures[1]
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&feedback_texture_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    // t_points
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&points_texture_view),
                },
            ],
        });
        let feedback_mix_bind_groups = [feedback_mix_bind_group_0, feedback_mix_bind_group_1];

        // ‼️ 5. Create shaders for feedback and composite passes
        let fs_quad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Full-screen Quad VS"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                struct VsOut {
                    @builtin(position) clip_position: vec4<f32>,
                    @location(0) uv: vec2<f32>,
                }

                @vertex
                fn vs_main(@builtin(vertex_index) v_idx: u32) -> VsOut {
                    var out: VsOut;
                    let x = f32(i32(v_idx) % 2) * 2.0;
                    let y = f32(i32(v_idx) / 2) * 2.0;
                    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
                    out.uv = vec2<f32>(x, y);
                    return out;
                }
                "#
                .into(),
            ),
        });

        // ‼️ This is the NEW mix shader
        let feedback_fs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Feedback Mix FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("feedback_shader.wgsl").into()),
        });

        let composite_fs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite FS"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var t_composite: texture_2d<f32>;
                @group(0) @binding(1) var s_composite: sampler;
                
                struct VsOut {
                    @builtin(position) clip_position: vec4<f32>,
                    @location(0) uv: vec2<f32>,
                }

                @fragment
                fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
                    return textureSample(t_composite, s_composite, in.uv);
                }
                "#
                .into(),
            ),
        });

        // ‼️ 6. Create feedback "mix" pipeline
        let feedback_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Feedback Mix Pipeline Layout"),
                bind_group_layouts: &[
                    &feedback_mix_bgl,          // ‼️ Group 0
                    &denoise_bind_group_layout, // ‼️ Group 1
                ],
                push_constant_ranges: &[],
            });

        let feedback_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Feedback Mix Pipeline"),
            layout: Some(&feedback_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fs_quad_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &feedback_fs_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: feedback_texture_format,
                    blend: Some(wgpu::BlendState::REPLACE), // ‼️ We are calculating the blend
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ‼️ 7. Create composite pipeline (draws final texture to screen)
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&texture_read_bind_group_layout], // ‼️ Use simple layout
                push_constant_ranges: &[],
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fs_quad_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_fs_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ‼️ 8. Create off-screen MSAA buffer for points
        let points_msaa_view = create_msaa_view_with_format(
            &device,
            config.width,
            config.height,
            feedback_texture_format,
            msaa_sample_count,
        );

        // ‼️ --- End of Feedback Setup ---

        // ‼️ --- This is now the "Points" Pipeline ---
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
            label: Some("Points Pipeline"), // ‼️ Renamed
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
                    format: feedback_texture_format,
                    blend: Some(wgpu::BlendState::REPLACE), // ‼️ Draw points opaquely
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
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

        let vertices = create_vertices(200);
        let num_vertices = vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
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
            // ‼️ Initialize new fields
            denoise_factor,
            denoise_uniform,
            denoise_buffer,
            denoise_bind_group,
            feedback_textures,
            feedback_texture_views,
            texture_read_bind_group_layout,
            feedback_read_bind_groups,
            points_texture,
            points_texture_view,
            sampler,
            feedback_pipeline,
            composite_pipeline,
            points_msaa_view,
            frame_index: 0,
            feedback_texture_format,
            feedback_mix_bgl,
            feedback_mix_bind_groups,
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

            // ‼️ --- Recreate Feedback Resources ---
            let (ping_texture, ping_texture_view) =
                create_feedback_texture(&self.device, width, height, self.feedback_texture_format);
            let (pong_texture, pong_texture_view) =
                create_feedback_texture(&self.device, width, height, self.feedback_texture_format);
            self.feedback_textures = [ping_texture, pong_texture];
            self.feedback_texture_views = [ping_texture_view, pong_texture_view];

            // ‼️ Recreate points texture
            let (points_texture, points_texture_view) =
                create_feedback_texture(&self.device, width, height, self.feedback_texture_format);
            self.points_texture = points_texture;
            self.points_texture_view = points_texture_view;

            // ‼️ Recreate simple read BGs
            let feedback_read_bind_group_0 =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Feedback Read BG 0"),
                    layout: &self.texture_read_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.feedback_texture_views[0],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
            let feedback_read_bind_group_1 =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Feedback Read BG 1"),
                    layout: &self.texture_read_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.feedback_texture_views[1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
            self.feedback_read_bind_groups =
                [feedback_read_bind_group_0, feedback_read_bind_group_1];

            // ‼️ Recreate mix BGs
            let feedback_mix_bind_group_0 =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Feedback Mix BG 0"),
                    layout: &self.feedback_mix_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.feedback_texture_views[0],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.points_texture_view),
                        },
                    ],
                });
            let feedback_mix_bind_group_1 =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Feedback Mix BG 1"),
                    layout: &self.feedback_mix_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.feedback_texture_views[1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.points_texture_view),
                        },
                    ],
                });
            self.feedback_mix_bind_groups = [feedback_mix_bind_group_0, feedback_mix_bind_group_1];

            self.points_msaa_view = create_msaa_view_with_format(
                &self.device,
                self.config.width,
                self.config.height,
                self.feedback_texture_format,
                self.msaa_sample_count,
            );
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if !pressed {
            return;
        }

        match key {
            KeyCode::Escape => {
                event_loop.exit();
            }
            KeyCode::ArrowUp => {
                self.denoise_factor = (self.denoise_factor + 0.01).min(1.0);
                println!("Denoise Factor (Less Trail): {:.2}", self.denoise_factor);
            }
            KeyCode::ArrowDown => {
                self.denoise_factor = (self.denoise_factor - 0.01).max(0.0);
                println!("Denoise Factor (More Trail): {:.2}", self.denoise_factor);
            }
            _ => {}
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        // ‼️ --- Update Buffers ---
        self.time_uniform.time = self.start_time.elapsed().as_secs_f32();
        let time = self.time_uniform.time;
        for vertex in self.vertices.iter_mut() {
            let x = vertex.position[0];
            vertex.position[1] = 0.5 * f32::sin(x * 5.0 + time * 2.0);
        }
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        self.queue.write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[self.time_uniform]),
        );

        // ‼️ Update denoise buffer
        self.denoise_uniform.factor = self.denoise_factor;
        self.queue.write_buffer(
            &self.denoise_buffer,
            0,
            bytemuck::cast_slice(&[self.denoise_uniform]),
        );
        // ‼️ --- End of Update Buffers ---

        let output = self.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let read_index = self.frame_index;
        let write_index = 1 - self.frame_index;

        // ‼️ --- Pass 1: Points Pass ---
        // Draw the new points into points_texture
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Points Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.points_msaa_view,                    // ‼️ Render to MSAA
                    resolve_target: Some(&self.points_texture_view), // ‼️ Resolve to points_texture
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), // ‼️ Clear
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.render_pipeline); // ‼️ Use points pipeline
            render_pass.set_bind_group(0, &self.time_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        // ‼️ --- Pass 2: Feedback Mix Pass ---
        // Mix old frame and new points into new frame
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Feedback Mix Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.feedback_texture_views[write_index], // ‼️ Write to new feedback tex
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // ‼️ Don't care, will be overwritten
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.feedback_pipeline); // ‼️ Use MIX pipeline
            render_pass.set_bind_group(0, &self.feedback_mix_bind_groups[read_index], &[]); // ‼️ Read old
            render_pass.set_bind_group(1, &self.denoise_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // ‼️ --- Pass 3: Composite Pass ---
        // Draw the final mixed frame to the screen
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    resolve_target: Some(&surface_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), // ‼️ Clear screen
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.composite_pipeline);
            render_pass.set_bind_group(0, &self.feedback_read_bind_groups[write_index], &[]); // ‼️ Read new
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        self.frame_index = write_index; // ‼️ Swap frames

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
