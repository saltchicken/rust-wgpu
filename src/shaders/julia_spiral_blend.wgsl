// ‼️ This is a new shader file that blends Julia and Spiral fractals.
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
var<storage, read> base_vertices: array<Vertex>;
@group(1) @binding(1)
var<storage, read_write> animated_vertices: array<Vertex>;

// Helper function for a 2D rotation matrix
fn rotation_matrix(angle: f32) -> mat2x2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return mat2x2<f32>(
        vec2<f32>(c, s),  // Column 0
        vec2<f32>(-s, c) // Column 1
    );
}

// Helper function for the *spiral* Julia iteration:
// f(z) = (z^2 + c) * k
fn spiral_iteration(z: vec2<f32>, c: vec2<f32>, rot_matrix: mat2x2<f32>) -> vec2<f32> {
    // 1. Standard Julia step: j = z^2 + c
    let z_squared = vec2<f32>(
        z.x * z.x - z.y * z.y, // real part
        2.0 * z.x * z.y      // imaginary part
    );
    let j = z_squared + c;
    
    // 2. Apply twist rotation: z_new = j * k
    return rot_matrix * j;
}

// Compute Shader
@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_vertices = arrayLength(&base_vertices);  
    
    // Bounds check
    if idx >= total_vertices {
        return;
    }

    // 1. Get the original vertex position as our starting complex number, z0.
    var z = base_vertices[idx].position * 0.8;

    // 2. 'c' constant *defines* the shape (same as julia.wgsl)
    let t = u_time.time * 0.22;
    let c = vec2<f32>(-0.32 + cos(t) * 0.681, 0.4 + sin(t) * 0.403);

    // 3. Number of iterations.
    let iterations = 30;

    // ‼️ 4. Define a twist angle that animates from 0.0 (Julia) to 0.15 (Spiral)
    // The (sin(t) * 0.5 + 0.5) part creates an oscillator from 0.0 to 1.0.
    let max_twist = 0.15;
    let twist_angle = (sin(u_time.time * 0.4) * 0.5 + 0.5) * max_twist;
    
    // ‼️ When twist_angle is 0.0, this becomes an identity matrix,
    // which results in the standard julia.wgsl behavior.
    let rot_matrix = rotation_matrix(twist_angle);

    // 5. Apply the blended iteration repeatedly.
    for (var i = 0; i < iterations; i = i + 1) {
        z = spiral_iteration(z, c, rot_matrix);
    }
    
    // 6. Set the new, transformed position
    animated_vertices[idx].position = z;
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// (Using the same colorful fragment shader from spiral_fractal.wgsl)
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
    let r = sin(in.world_pos.x * 2.0) * 0.5 + 0.5;
    let g = sin(in.world_pos.y * 2.0) * 0.5 + 0.5;
    let b = 1.0 - g;
    let intensity = 0.1; // Keep intensity low for feedback
    return vec4<f32>(r * intensity, g * intensity, b * intensity, 1.0);
}
