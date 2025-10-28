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

// ‼️ Helper for 4x4 Z-axis rotation is no longer needed.

// ‼️ Helper function for our 2D affine transform matrix (A)
// This matrix will scale and rotate the point.
fn get_transform_matrix() -> mat2x2<f32> {
    // ‼️ A scale slightly less than 1.0 makes it a "contraction"
    let scale = 0.99; 
    
    // ‼️ We can use time to slowly change the rotation
    let angle = u_time.time * 0.05;
    let s = sin(angle);
    let c = cos(angle);
    
    // ‼️ WGSL matrices are column-major
    return mat2x2<f32>(
        vec2<f32>(scale * c, scale * s),  // Column 0
        vec2<f32>(scale * -s, scale * c) // Column 1
    );
}

// ‼️ Helper function for our 2D affine translation vector (b)
// This is the 'b' in the transform T(v) = Av + b
fn get_translation_vector() -> vec2<f32> {
    // ‼️ We make the translation vector move in a small circle.
    // This makes the fractal attractor move around.
    let t = u_time.time * 0.7;
    return vec2<f32>(sin(t) * 0.02, cos(t) * 0.02);
}

// Compute Shader
@compute @workgroup_size(256, 1, 1) // Use a 1D workgroup
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_vertices = arrayLength(&base_vertices); 
    
    // Bounds check
    if idx >= total_vertices {
        return;
    }

    // ‼️ 1. Get the original vertex position as our starting point
    var p = base_vertices[idx].position;

    // ‼️ 2. Define the number of iterations to apply the transform.
    // More iterations make the points converge more sharply.
    let iterations = 25;

    // ‼️ 3. Get the affine transform components
    let matrix_a = get_transform_matrix();
    let vector_b = get_translation_vector();

    // ‼️ 4. Apply the affine transform T(p) = A*p + b repeatedly.
    // This loop is the core of the fractal attractor.
    // Each point in the grid is pulled towards the same
    // converging point/shape.
    for (var i = 0; i < iterations; i = i + 1) {
        p = (matrix_a * p) + vector_b;
    }
    
    // ‼️ 5. Set the new, transformed position
    animated_vertices[idx].position = p;
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// (No changes needed below)
// ------------------------------------------------------------------
struct VertexInput {
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
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 1.0, 1.0);
}
