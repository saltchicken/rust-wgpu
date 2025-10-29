// ‼️ This is a brand new shader file.

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
var<storage, read> base_vertices: array<Vertex>; // ‼️ Will have 1 element
@group(1) @binding(1)
var<storage, read_write> animated_vertices: array<Vertex>; // ‼️ Will have N elements

// Helper function for the Julia iteration: f(z) = z^2 + c
// (Copied from julia.wgsl)
fn julia_iteration(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    // (x + iy)^2 = (x^2 - y^2) + i(2xy)
    let z_squared = vec2<f32>(
        z.x * z.x - z.y * z.y, // real part
        2.0 * z.x * z.y      // imaginary part
    );
    // z^2 + c
    return z_squared + c;
}

// Compute Shader
@compute @workgroup_size(256, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // ‼️ Get the total number of particles we need to calculate
    let total_output_particles = arrayLength(&animated_vertices);

    // ‼️ Bounds check
    if idx >= total_output_particles {
        return;
    }
    
    // ‼️ Safety check in case no base vertex is provided
    if arrayLength(&base_vertices) == 0u {
        return;
    }
    // ‼️ 1. Get 'c' from the *single input point*.
    // We animate it slightly around this point for a cool effect.
    let t = u_time.time * 0.3;
    let c_offset = vec2<f32>(sin(t) * 0.005, cos(t) * 0.005);
    let c = base_vertices[0].position + c_offset;

    // ‼️ 2. Generate a "virtual" starting grid point (z0) based on the particle's index.
    // We'll calculate a virtual grid dimension.
    let grid_dim_f = floor(sqrt(f32(total_output_particles)));
    let grid_dim_u = u32(grid_dim_f);

    let virtual_x = f32(idx % grid_dim_u);
    let virtual_y = f32(idx / grid_dim_u);

    // ‼️ 3. Normalize coordinates (0.0 to 1.0) and map to the complex plane (-2.0 to 2.0)
    let scale = 1.7; // Zoom
    var z = vec2<f32>(
        (virtual_x / (grid_dim_f - 1.0) * 2.0 - 1.0) * scale,
        (virtual_y / (grid_dim_f - 1.0) * 2.0 - 1.0) * scale
    );

    // ‼️ 4. Run the Julia iteration
    let iterations = 50;
    for (var i = 0; i < iterations; i = i + 1) {
        z = julia_iteration(z, c);
    }
    
    // ‼️ 5. Set the new, transformed position
    animated_vertices[idx].position = z;
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// ‼️ Using the colorful fragment shader from the particle fountain
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

    return vec4<f32>(r, g, b, 1.0);
}

