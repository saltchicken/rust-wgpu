// ‼️ This is a brand new shader file combining both concepts.
struct Time {
    time: f32,
}
@group(0) @binding(0)
var<uniform> u_time: Time;

struct Vertex {
    position: vec2<f32>,
}

// Group 1: Compute shader storage buffers
@group(1) @binding(0)
var<storage, read> base_vertices: array<Vertex>; // ‼️ Will have 1 element (the 'c' value)
@group(1) @binding(1)
var<storage, read_write> animated_vertices: array<Vertex>; // ‼️ Will have N elements (the particles)

// ‼️ Helper function for a 2D rotation matrix
// (Copied from spiral_fractal.wgsl)
fn rotation_matrix(angle: f32) -> mat2x2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return mat2x2<f32>(
        vec2<f32>(c, s),  // Column 0
        vec2<f32>(-s, c) // Column 1
    );
}

// ‼️ Helper function for the *spiral* Julia iteration:
// f(z) = (z^2 + c) * k
// (Copied from spiral_fractal.wgsl)
fn spiral_iteration(z: vec2<f32>, c: vec2<f32>, rot_matrix: mat2x2<f32>) -> vec2<f32> {
    // 1. Standard Julia step: j = z^2 + c
    let z_squared = vec2<f32>(
        z.x * z.x - z.y * z.y, // real part
        2.0 * z.x * z.y      // imaginary part
    );
    let j = z_squared + c;
    
    // 2. ‼️ Apply twist rotation: z_new = j * k
    return rot_matrix * j;
}

// Compute Shader
@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // ‼️ Get the total number of particles we need to calculate
    let total_output_particles = arrayLength(&animated_vertices);

    // Bounds check
    if idx >= total_output_particles {
        return;
    }
    
    // ‼️ Safety check
    if arrayLength(&base_vertices) == 0u {
        return;
    }

    // ‼️ 1. Get 'c' from the *single input point*.
    // (Logic from julia_spawner.wgsl)
    let t = u_time.time * 0.3;
    let c_offset = vec2<f32>(sin(t) * 0.05, cos(t) * 0.05);
    let c = base_vertices[0].position + c_offset;

    // ‼️ 2. Generate a "virtual" starting grid point (z0) based on the particle's index.
    // (Logic from julia_spawner.wgsl)
    let grid_dim_f = floor(sqrt(f32(total_output_particles)));
    let grid_dim_u = u32(grid_dim_f);
    let virtual_x = f32(idx % grid_dim_u);
    let virtual_y = f32(idx / grid_dim_u);

    // ‼️ 3. Normalize coordinates and map to the complex plane
    // (Logic from julia_spawner.wgsl)
    let scale = 1.7; // Zoom
    var z = vec2<f32>(
        (virtual_x / (grid_dim_f - 1.0) * 2.0 - 1.0) * scale,
        (virtual_y / (grid_dim_f - 1.0) * 2.0 - 1.0) * scale
    );

    // ‼️ 4. Define the spiral parameters
    // (Logic from spiral_fractal.wgsl)
    let iterations = 50;
    let twist_angle = 0.1 + sin(u_time.time * 0.1) * 0.05;
    let rot_matrix = rotation_matrix(twist_angle);

    // ‼️ 5. Run the *spiral* iteration
    for (var i = 0; i < iterations; i = i + 1) {
        z = spiral_iteration(z, c, rot_matrix);
    }
    
    // ‼️ 6. Set the new, transformed position
    animated_vertices[idx].position = z;
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// (Using the same colorful fragment shader as the other spawners)
// ------------------------------------------------------------------
struct VertexInput {
    @location(0) position: vec2<f32>,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.world_pos = model.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Color the particle based on its *final* position
    let r = sin(in.world_pos.x * 2.0) * 0.5 + 0.5;
    let g = sin(in.world_pos.y * 2.0) * 0.5 + 0.5;
    let b = 1.0 - g;
    let intensity = 0.1;
    return vec4<f32>(r * intensity, g * intensity, b * intensity, 1.0);
}
