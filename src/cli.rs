use clap::{Parser, Subcommand, ValueEnum};

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ShaderChoice {
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
    #[value(name = "julia-spawner")]
    JuliaSpawner,
    #[value(name = "spiral-fractal")]
    SpiralFractal,
    #[value(name = "julia-spiral-blend")]
    JuliaSpiralBlend,
    #[value(name = "spiral-spawner")]
    SpiralSpawner,
}

impl ShaderChoice {
    pub fn as_path(&self) -> &'static str {
        match self {
            ShaderChoice::AffineRotate => "shaders/affine_rotate.wgsl",
            ShaderChoice::Julia => "shaders/julia.wgsl",
            ShaderChoice::Shader => "shaders/shader.wgsl",
            ShaderChoice::Shader2 => "shaders/shader2.wgsl",
            ShaderChoice::ParticleShader => "shaders/particle_shader.wgsl",
            ShaderChoice::JuliaSpawner => "shaders/julia_spawner.wgsl",
            ShaderChoice::SpiralFractal => "shaders/spiral_fractal.wgsl",
            ShaderChoice::JuliaSpiralBlend => "shaders/julia_spiral_blend.wgsl",
            ShaderChoice::SpiralSpawner => "shaders/spiral_spawner.wgsl",
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum InputCommand {
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
        #[arg(short = 'x', long, default_value_t = 0.0, allow_hyphen_values = true)]
        x: f32,
        /// Y coordinate
        #[arg(short = 'y', long, default_value_t = 0.0, allow_hyphen_values = true)]
        y: f32,
        #[arg(
            short = 'n',
            long,
            default_value_t = 100000,
            allow_hyphen_values = true
        )]
        num_particles: u32,
    },
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "affine-rotate")]
    pub shader_name: ShaderChoice,
    #[command(subcommand)]
    pub command: Option<InputCommand>,
}
