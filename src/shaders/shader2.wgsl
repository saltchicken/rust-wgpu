struct Time {
    time: f32,
}
@group(0) @binding(0)
var<uniform> u_time: Time;

struct Vertex {
    position: vec2<f32>,
}

// Group 1: Compute shader storage buffers
// @binding(0) is the "read-only" base grid
@group(1) @binding(0)
var<storage, read> base_vertices: array<Vertex>;
// @binding(1) is the "write-to" animated grid
@group(1) @binding(1)
var<storage, read_write> animated_vertices: array<Vertex>;

// ‼️ Helper function rotation_z is no longer needed and has been removed.

// Compute Shader
@compute @workgroup_size(256, 1, 1) // Use a 1D workgroup
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_vertices = arrayLength(&base_vertices); 
    
    // Bounds check to prevent out-of-bounds access
    if idx >= total_vertices {
        return;
    }

    // Get the original vertex
    let base_vertex = base_vertices[idx];
    let base_pos = base_vertex.position;

    // ‼️ 1. Calculate the distance from the origin (0.0, 0.0)
    let distance = length(base_pos);
    
    // ‼️ 2. Define wave parameters
    let amplitude = 0.1;  // Max height of wave (matches vertex.rs)
    let frequency = 20.0; // How many wave crests (spatial density)
    let speed = 5.0;    // How fast the wave travels outwards
    
    // ‼️ 3. Calculate the wave argument
    // The wave equation is based on distance and time.
    // (distance * frequency) creates static concentric rings.
    // Subtracting (time * speed) makes the rings move outwards.
    let wave_arg = distance * frequency - u_time.time * speed;
    
    // ‼️ 4. Calculate the vertical offset using sin()
    let offset = amplitude * sin(wave_arg);
    
    // ‼️ 5. Set the new position.
    // We keep the original x-coordinate,
    // and add the calculated 'offset' to the y-coordinate.
    // This makes the points move up and down based on their distance
    // from the center, creating a ripple effect.
    animated_vertices[idx].position = vec2<f32>(base_pos.x, base_pos.y + offset);
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// (No changes needed below)
// ------------------------------------------------------------------
struct VertexInput {
    // This @location(0) will come from the animated_vertices buffer
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    // Position calculated in the compute shader
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    //let green = (sin(u_time.time * 2.0) + 1.0) * 0.5;
    return vec4<f32>(0.0, 1.0, 1.0, 1.0);
}

