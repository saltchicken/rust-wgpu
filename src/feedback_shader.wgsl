// ‼️ This is the BGL for the mix pass
@group(0) @binding(0) var t_old: texture_2d<f32>;
@group(0) @binding(1) var s_old: sampler;
@group(0) @binding(2) var t_points: texture_2d<f32>;
// ‼️ This is the denoise uniform
@group(1) @binding(0) var<uniform> denoise: DenoiseUniform;

struct DenoiseUniform {
    factor: f32, // This is the denoise_factor (e.g., 0.1)
}

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let old_color = textureSample(t_old, s_old, in.uv);
    let new_points = textureSample(t_points, s_old, in.uv); // Use the same sampler
    
    // ‼️ This is the correct interpolation logic
    let persistence_factor = 1.0 - denoise.factor;
    
    // new_frame = (old_frame * persistence) + (new_points * denoise)
    return (old_color * persistence_factor) + (new_points * denoise.factor);
}

