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

// Helper for 4x4 Z-axis rotation
fn rotation_z(angle: f32) -> mat4x4<f32> {
    let s = sin(angle);
    let c = cos(angle);
    // WGSL matrices are column-major
    return mat4x4<f32>(
        vec4<f32>(c, s, 0.0, 0.0),   // Column 0
        vec4<f32>(-s, c, 0.0, 0.0),  // Column 1
        vec4<f32>(0.0, 0.0, 1.0, 0.0), // Column 2
        vec4<f32>(0.0, 0.0, 0.0, 1.0)  // Column 3
    );
}

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
    let base_x = base_vertex.position.x;
    let base_y = base_vertex.position.y;

    // 1. Create rotation matrix (same as Rust)
    let rotation = rotation_z(u_time.time * 0.5);
    
    // 2. Convert to Vec4
    let base_pos_vec4 = vec4<f32>(base_x, base_y, 0.0, 1.0);

    // 3. Apply transform
    let rotated_pos = rotation * base_pos_vec4;
    
    // 4. Calculate animation offset
    let offset = 0.1 * sin(rotated_pos.x * 10.0 + rotated_pos.y * 10.0 + u_time.time * 2.0);

    // 5. Set new position in the writable buffer
    animated_vertices[idx].position = vec2<f32>(rotated_pos.x, rotated_pos.y + offset);
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
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
    return vec4<f32>(0.0, 0.01, 0.01, 1.0);
}
