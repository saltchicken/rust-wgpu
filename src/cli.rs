use clap::{Parser, Subcommand, ValueEnum};

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ShaderChoice {
    #[default]
    #[value(name = "shader")]
    Shader,
    #[value(name = "shader2")]
    Shader2,
}

impl ShaderChoice {
    pub fn as_path(&self) -> &'static str {
        match self {
            ShaderChoice::Shader => "src/shaders/shader.wgsl",
            ShaderChoice::Shader2 => "src/shaders/shader2.wgsl",
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
    #[arg(short, long, default_value = "shader")]
    pub shader_name: ShaderChoice,
    #[command(subcommand)]
    pub command: Option<InputCommand>,
}
