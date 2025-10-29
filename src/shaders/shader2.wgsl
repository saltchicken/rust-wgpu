struct Time {
    time: f32,
}
@group(0) @binding(0)
var<uniform> u_time: Time;

struct Vertex {
    position: vec3<f32>, // ‼️ Changed to vec3
}

// Group 1: Compute shader storage buffers
@group(1) @binding(0)
var<storage, read> base_vertices: array<Vertex>;
@group(1) @binding(1)
var<storage, read_write> animated_vertices: array<Vertex>;

// Compute Shader
@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_vertices = arrayLength(&base_vertices);

    if idx >= total_vertices {
        return;
    }

    let base_vertex = base_vertices[idx];
    let base_pos = base_vertex.position; // ‼️ This is now a vec3

    // ‼️ 1. Calculate the distance from the origin on the X-Z plane
    let distance = length(base_pos.xz);
   
    // ‼️ 2. Define wave parameters
    let amplitude = 0.2; // ‼️ Increased amplitude for better 3D visibility
    let frequency = 20.0;
    let speed = 5.0; 
   
    // ‼️ 3. Calculate the wave argument (no change)
    let wave_arg = distance * frequency - u_time.time * speed;
   
    // ‼️ 4. Calculate the vertical offset using sin()
    let offset = amplitude * sin(wave_arg);
   
    // ‼️ 5. Set the new position.
    // ‼️ We keep the original x and z, and apply the 'offset' to the y-coordinate.
    animated_vertices[idx].position = vec3<f32>(base_pos.x, offset, base_pos.z);
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// ------------------------------------------------------------------

// ‼️ Add Camera uniform (Group 1, Binding 0)
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> u_camera: CameraUniform;


struct VertexInput {
    @location(0) position: vec3<f32>, // ‼️ Changed to vec3
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    // ‼️ Apply View-Projection matrix
    out.clip_position = u_camera.proj * u_camera.view * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 1.0, 1.0);
}
