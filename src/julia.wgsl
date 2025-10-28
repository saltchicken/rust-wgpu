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

// ‼️ Helper functions for affine transform are removed.

// ‼️ Helper function for the Julia iteration: f(z) = z^2 + c
// We use vec2<f32> to represent complex numbers: (real, imaginary)
fn julia_iteration(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    // (x + iy)^2 = (x^2 - y^2) + i(2xy)
    let z_squared = vec2<f32>(
        z.x * z.x - z.y * z.y, // real part
        2.0 * z.x * z.y        // imaginary part
    );
    
    // z^2 + c
    return z_squared + c;
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

    // ‼️ 1. Get the original vertex position as our starting complex number, z0.
    // We scale it down a bit to fit more of the interesting part on screen.
    var z = base_vertices[idx].position * 0.8;

    // ‼️ 2. Define the constant 'c'. This constant *defines* the shape
    // of the Julia set. We animate it with time to make it morph.
    let t = u_time.time * 0.3;
    let c = vec2<f32>(-0.4, 0.4 + sin(t) * 0.2);
    // Try static values too! e.g., vec2<f32>(-0.8, 0.156)

    // ‼️ 3. Define the number of iterations to apply the transform.
    // This is not an "escape time" check, but a fixed number
    // of transformations.
    let iterations = 20;

    // ‼️ 4. Apply the Julia iteration repeatedly.
    // z = f(f(f(...f(z0)...)))
    for (var i = 0; i < iterations; i = i + 1) {
        z = julia_iteration(z, c);
    }
    
    // ‼️ 5. Set the new, transformed position
    // Points that are "outside" the set will have flown
    // towards infinity (Inf), and WGPU will cull them.
    // Points "inside" will remain in a bounded, fractal region.
    animated_vertices[idx].position = z;
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
    // ‼️ We set z=0.0 to keep the points in the 2D plane
    out.clip_position = vec4<f32>(model.position.x, model.position.y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 1.0, 1.0);
}
