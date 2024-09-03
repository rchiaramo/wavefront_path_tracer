@group(0) @binding(0) var<storage, read> image_buffer: array<array<f32, 3>>;
@group(0) @binding(1) var<uniform> frame_buffer: FrameBuffer;

struct FrameBuffer {
    width: u32,
    height: u32,
    frame: u32,
    accumulated_samples: u32
}

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) TexCoord: vec2<f32>,
};

@vertex
fn vs(
    @builtin(vertex_index) VertexIndex: u32,
) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0)
    );

    var texCoords = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0)
    );

    var output: VertexOutput;
    output.Position = vec4<f32>(positions[VertexIndex], 0.0, 1.0);
    output.TexCoord = texCoords[VertexIndex];
    return output;
}

@fragment
fn fs(@location(0) TexCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let x = u32(TexCoord.x * f32(frame_buffer.width));
    let y = u32(TexCoord.y * f32(frame_buffer.height));
    let idx = x + y * frame_buffer.width;

    let invN = 1.0 / f32(frame_buffer.accumulated_samples);
    var color = vec3(image_buffer[idx][0], image_buffer[idx][1], image_buffer[idx][2]);
    color = sqrt(invN * color);

    return vec4(color.xyz, 1.0);
}