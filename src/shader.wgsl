// ‼️ Changed: Renamed struct and added resolution/point_size
struct Globals {
    resolution: vec2<f32>,
    time: f32,
    point_size: f32,
}

// ‼️ Changed: Renamed uniform
@group(0) @binding(0)
// ‼️ Changed: Added visibility to vertex shader
var<uniform> u_globals: Globals;

// ‼️ Changed: This struct now represents per-instance data
struct InstanceInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

// ‼️ Changed: Swapped argument order
@vertex
fn vs_main(
    instance: InstanceInput, // ‼️ Per-instance data first (from buffer 0)
    @builtin(vertex_index) v_idx: u32, // ‼️ Per-vertex data second (from buffer 1)
) -> VertexOutput {
    // ‼️ Added: Define 6 vertices for a quad (2 triangles)
    // from -0.5 to 0.5
    let quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
    );

    // ‼️ Added: Logic to scale the point size
    // 1. Get the clip-space size of a single pixel
    let pixel_clip_size = vec2<f32>(2.0 / u_globals.resolution.x, 2.0 / u_globals.resolution.y);
    // 2. Calculate the offset by scaling the quad vertex by the desired pixel size
    let offset = quad_positions[v_idx] * u_globals.point_size * pixel_clip_size;
    
    var out: VertexOutput;
    // 3. Final position is the instance's center + the scaled quad vertex offset
    out.clip_position = vec4<f32>(instance.position + offset, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // ‼️ Changed: Use u_globals.time
    let green = (sin(u_globals.time * 2.0) + 1.0) * 0.5;
    return vec4<f32>(0.0, green, 0.2, 1.0);
}
