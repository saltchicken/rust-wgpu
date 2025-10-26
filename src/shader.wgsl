struct Time {
    time: f32,
}

@group(0) @binding(0)
var<uniform> u_time: Time;

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
    let green = (sin(u_time.time * 2.0) + 1.0) * 0.5;
    return vec4<f32>(0.0, green, 1.0, 1.0);
}
