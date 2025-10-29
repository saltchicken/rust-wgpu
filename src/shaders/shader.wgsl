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

// Helper for 4x4 Z-axis rotation (remains the same)
fn rotation_z(angle: f32) -> mat4x4<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return mat4x4<f32>(
        vec4<f32>(c, s, 0.0, 0.0),   // Column 0
        vec4<f32>(-s, c, 0.0, 0.0),  // Column 1
        vec4<f32>(0.0, 0.0, 1.0, 0.0), // Column 2
        vec4<f32>(0.0, 0.0, 0.0, 1.0)  // Column 3
    );
}

// Compute Shader
@compute @workgroup_size(128, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_vertices = arrayLength(&base_vertices);

    if idx >= total_vertices {
        return;
    }

    let base_vertex = base_vertices[idx];
    let base_x = base_vertex.position.x;
    // ‼️ Read the Z coordinate
    let base_z = base_vertex.position.z;

    let rotation = rotation_z(u_time.time * 0.5);
   
    // ‼️ Convert X, 0.0, Z to Vec4
    let base_pos_vec4 = vec4<f32>(base_x, 0.0, base_z, 1.0);

    let rotated_pos = rotation * base_pos_vec4;
   
    // ‼️ Calculate offset based on rotated X and Z
    let offset = 0.1 * sin(rotated_pos.x * 10.0 + rotated_pos.z * 10.0 + u_time.time * 2.0);
    
    // ‼️ Set new position, applying offset to the Y coordinate
    // animated_vertices[idx].position = vec3<f32>(rotated_pos.x, rotated_pos.y, rotated_pos.z);
    animated_vertices[idx].position = vec3<f32>(base_vertex.position.x, base_vertex.position.y, base_vertex.position.z);
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// ------------------------------------------------------------------

// ‼️ Add Camera uniform
// ‼️ This is @group(1) because the render pipeline layout in state.rs
// ‼️ specifies [&time_bind_group_layout, &camera_bind_group_layout]
// ‼️ (Group 0 = time, Group 1 = camera)
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
    return vec4<f32>(0.0, 0.5, 0.5, 1.0);
}
