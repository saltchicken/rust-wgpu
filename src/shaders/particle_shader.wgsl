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

    // ‼️ This shader can run in two modes:
    // 1. "Point" mode: base_vertices has 1 element. All particles use it as an emitter.
    // 2. "Grid" mode: base_vertices has N elements. Each particle idx uses base_vertices[idx % N].
    // This makes it work for both of your new command-line options!
    let total_base_vertices = arrayLength(&base_vertices);
    let emitter_idx = idx % total_base_vertices;
    let emitter_pos = base_vertices[emitter_idx].position;

    let particle_id = f32(idx);
    let time = u_time.time;

    // ‼️ Give each particle a unique, stable angle
    let angle = particle_id * 0.37; // Using a golden angle-ish multiplier
    let s = sin(angle);
    let c = cos(angle);

    // ‼️ Give each particle a unique, stable initial speed
    let initial_speed = 0.5 + (particle_id % 1000.0) / 2000.0; // Varies from 0.5 to 1.0

    // ‼️ Each particle has a "life" that cycles, offset by its ID
    let particle_life = (time * 0.5 + particle_id * 0.01) % 4.0; // 4-second lifespan

    // ‼️ Basic projectile motion (a "fountain")
    let gravity = -0.8;
    let vx = s * initial_speed;
    let vy = c * initial_speed * 3.0; // Shoot "up" (in all directions, but Y velocity is stronger)

    let pos_x = emitter_pos.x + vx * particle_life;
    let pos_y = emitter_pos.y + vy * particle_life + 0.5 * gravity * particle_life * particle_life;
    
    // ‼️ 5. Set the new, transformed position
    animated_vertices[idx].position = vec2<f32>(pos_x, pos_y);
}

// ------------------------------------------------------------------
// Render Pipeline Shaders
// ------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // ‼️ We'll pass the world position to the fragment shader for coloring
    @location(0) world_pos: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    // ‼️ Pass the position through
    out.world_pos = model.position;
    return out;
}

@fragment
// ‼️ Get the world_pos from the vertex shader
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ‼️ Color the particle based on its position and time
    let dist_from_origin = length(in.world_pos);

    let r = sin(dist_from_origin * 1.0 - u_time.time) * 0.5 + 0.5;
    let g = sin(dist_from_origin * 0.5 + u_time.time * 0.5) * 0.5 + 0.5;
    let b = 1.0 - r;

    return vec4<f32>(r, g, b, 1.0); // Colorful!
}

