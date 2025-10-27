use std::{iter, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

const POINTS_X: u32 = 50;
const POINTS_Y: u32 = 50;
// const TOTAL_VERTICES: u32 = POINTS_X * POINTS_Y;
const COMPUTE_WORKGROUP_SIZE: u32 = 256;
// This MUST match the amplitude in the shader
const WAVE_AMPLITUDE: f32 = 0.1;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    time: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DenoiseUniform {
    factor: f32,
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

#[derive(Debug)]
pub struct Grid<T> {
    data: Vec<T>,
    width: u32,
    height: u32,
}

impl<T> Grid<T> {
    pub fn get(&self, x: u32, y: u32) -> Option<&T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get((y * self.width + x) as usize)
    }
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

    pub fn as_flat_vec(&self) -> &Vec<T> {
        &self.data
    }
}

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
        return Grid {
            data: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    let rotation_buffer = 2.0_f32.sqrt();
    let buffer = rotation_buffer + WAVE_AMPLITUDE;
    let grid_range = buffer * 2.0;

    let total_vertices = (points_x * points_y) as usize;
    let mut vertices = Vec::with_capacity(total_vertices);

    let step_x = if points_x > 1 {
        grid_range / (points_x - 1) as f32
    } else {
        0.0
    };
    let step_y = if points_y > 1 {
        grid_range / (points_y - 1) as f32
    } else {
        0.0
    };

    for j in 0..points_y {
        let y = if points_y == 1 {
            0.0
        } else {
            -buffer + (j as f32 * step_y)
        };
        for i in 0..points_x {
            let x = if points_x == 1 {
                0.0
            } else {
                -buffer + (i as f32 * step_x)
            };
            vertices.push(Vertex { position: [x, y] });
        }
    }

    Grid {
        data: vertices,
        width: points_x,
        height: points_y,
    }
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

fn create_feedback_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
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

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    base_grid: Grid<Vertex>,
    base_vertex_buffer: wgpu::Buffer,
    animated_vertex_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
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

    // Feedback effect fields
    denoise_factor: f32,
    denoise_uniform: DenoiseUniform,
    denoise_buffer: wgpu::Buffer,
    denoise_bind_group: wgpu::BindGroup,
    feedback_textures: [wgpu::Texture; 2],
    feedback_texture_views: [wgpu::TextureView; 2],
    texture_read_bind_group_layout: wgpu::BindGroupLayout,
    feedback_read_bind_groups: [wgpu::BindGroup; 2],
    points_texture: wgpu::Texture,
    points_texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    feedback_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    points_msaa_view: wgpu::TextureView,
    frame_index: usize,
    feedback_texture_format: wgpu::TextureFormat,
    feedback_mix_bgl: wgpu::BindGroupLayout,
    feedback_mix_bind_groups: [wgpu::BindGroup; 2],
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
            // present_mode: surface_caps.present_modes[0],
            alpha_mode,
            // alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        if size.width > 0 && size.height > 0 {
            surface.configure(&device, &config);
        }

        let msaa_sample_count = 4;
        let msaa_view = create_msaa_view(&device, &config, msaa_sample_count);

        // ‼️ Use a high-precision format for accumulation
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
                    // Time is used by COMPUTE and FRAGMENT
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
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
            label: Some("Denoise Buffer"),
            contents: bytemuck::cast_slice(&[denoise_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let denoise_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Denoise BGL"),
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
            label: Some("Denoise BG"),
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
                    &feedback_mix_bgl,          // Group 0
                    &denoise_bind_group_layout, // Group 1
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
            multisample: wgpu::MultisampleState::default(), // ‼️ No MSAA here
            multiview: None,
            cache: None,
        });

        // ‼️ 7. Create composite pipeline (draws final texture to screen)
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&texture_read_bind_group_layout],
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
                    format: config.format, // ‼️ This one draws to the screen
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
                count: msaa_sample_count, // ‼️ This one uses screen MSAA
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

        // --- Create GPU Buffers ---
        let base_grid = create_vertex_grid(POINTS_X, POINTS_Y);
        let base_vertices_data = base_grid.as_flat_vec();
        let num_vertices = base_vertices_data.len() as u32;

        let base_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Base Vertex Buffer"),
            contents: bytemuck::cast_slice(base_vertices_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let animated_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Animated Vertex Buffer"),
            size: (base_vertices_data.len() * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Create Compute Pipeline ---
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    // @binding(0) base_vertices
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1) animated_vertices
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: animated_vertex_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                // Group 0 is time, Group 1 is storage buffers
                bind_group_layouts: &[&time_bind_group_layout, &compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Create Render Pipeline ---
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                // Render pipeline only needs the time bind group
                bind_group_layouts: &[&time_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()], // Describes the animated_vertex_buffer
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // ‼️ FIX #1: This pipeline renders to the off-screen points texture,
                    // ‼️ so it MUST match that texture's format.
                    format: feedback_texture_format,
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
                count: msaa_sample_count, // ‼️ It renders to an MSAA view
                mask: !0,
                alpha_to_coverage_enabled: true,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: size.width > 0 && size.height > 0,
            render_pipeline,
            base_grid,
            base_vertex_buffer,
            animated_vertex_buffer,
            compute_pipeline,
            compute_bind_group,
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

    pub fn modify_base_vertex(&mut self, x: u32, y: u32, new_pos: [f32; 2]) {
        // 1. Get width *before* the mutable borrow to avoid the error
        let width = self.base_grid.width();

        // 2. Update the CPU-side grid
        if let Some(vertex) = self.base_grid.get_mut(x, y) {
            vertex.position = new_pos;

            // 3. Calculate the byte offset in the GPU buffer
            let vertex_size = std::mem::size_of::<Vertex>() as wgpu::BufferAddress;
            // Use the `width` variable here
            let offset = (y * width + x) as wgpu::BufferAddress * vertex_size;

            // 4. Write just this one vertex's data to the GPU buffer
            self.queue.write_buffer(
                &self.base_vertex_buffer,
                offset,
                bytemuck::cast_slice(&[*vertex]), // Create a slice containing one copied vertex
            );
        }
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
        self.feedback_read_bind_groups = [feedback_read_bind_group_0, feedback_read_bind_group_1];

        // ‼️ Recreate mix BGs
        let feedback_mix_bind_group_0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Mix BG 0"),
            layout: &self.feedback_mix_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.feedback_texture_views[0]),
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
        let feedback_mix_bind_group_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Mix BG 1"),
            layout: &self.feedback_mix_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.feedback_texture_views[1]),
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

        // Update time
        self.time_uniform.time = self.start_time.elapsed().as_secs_f32();
        self.queue.write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[self.time_uniform]),
        );

        // Update denoise
        self.denoise_uniform.factor = self.denoise_factor;
        self.queue.write_buffer(
            &self.denoise_buffer,
            0,
            bytemuck::cast_slice(&[self.denoise_uniform]),
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
            // --- Compute Pass ---
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.time_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.compute_bind_group, &[]);
            let workgroup_count_x =
                (self.num_vertices as f32 / COMPUTE_WORKGROUP_SIZE as f32).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroup_count_x, 1, 1);
        }

        let read_index = self.frame_index;
        let write_index = 1 - self.frame_index;

        {
            // ‼️ --- Pass 1: Points Pass ---
            // ‼️ Render the animated points to the off-screen "points_texture"
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Points Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.points_msaa_view,                    // Render to MSAA
                    resolve_target: Some(&self.points_texture_view), // Resolve to points_texture
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), // Clear
                        store: wgpu::StoreOp::Store, // ‼️ Store the resolved texture
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.render_pipeline); // Use points pipeline
            render_pass.set_bind_group(0, &self.time_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.animated_vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        // ‼️ --- Pass 2: Feedback Mix Pass ---
        // ‼️ Mix old frame (from feedback_textures[read]) and new points (from points_texture)
        // ‼️ and write the result to the new frame (feedback_textures[write])
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Feedback Mix Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.feedback_texture_views[write_index], // Write to new feedback tex
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't care, will be overwritten
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.feedback_pipeline); // Use MIX pipeline
            render_pass.set_bind_group(0, &self.feedback_mix_bind_groups[read_index], &[]); // Read old
            render_pass.set_bind_group(1, &self.denoise_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // ‼️ --- Pass 3: Composite Pass ---
        // ‼️ Draw the final mixed frame (feedback_textures[write]) to the screen
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,       // Render to screen's MSAA
                    resolve_target: Some(&view), // Resolve to screen
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), // Clear screen
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.composite_pipeline);
            render_pass.set_bind_group(0, &self.feedback_read_bind_groups[write_index], &[]); // Read new
            render_pass.draw(0..3, 0..1);
        }

        // ‼️ FIX #2: The redundant, fourth render pass has been DELETED from here.

        // Submit all passes at once
        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        // ‼️ Swap frames for next tick
        self.frame_index = write_index;

        // ... (FPS counter remains the same) ...
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
