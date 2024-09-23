struct Ray {
    origin: vec4f,
    direction: vec4f,
    invDirection: vec3f,
    pixel_idx: u32
}

@group(0) @binding(0) var<storage, read_write> image_buffer: array<array<f32, 3>>;
@group(0) @binding(1) var<storage, read> ray_buffer: array<Ray>;
@group(0) @binding(2) var<storage, read_write> miss_buffer: array<u32>;
@group(0) @binding(3) var<storage, read> counter_buffer: array<u32>;

@compute @workgroup_size(8,4,1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3u) {

    let workgroup_index = workgroup_id.x +
        workgroup_id.y * num_workgroups.x +
        workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let idx = workgroup_index * 32u + local_index;

    if idx >= counter_buffer[0] {
        return;
    }

    let ray_idx = miss_buffer[idx];
    let ray = ray_buffer[ray_idx];
    let pixel_idx = ray.pixel_idx;

    let a: f32 = 0.5 * (ray.direction.y + 1.0);
    let pixel_color = (1.0 - a) * vec3f(1.0, 1.0, 1.0) + a * vec3f(0.5, 0.7, 1.0);

    image_buffer[pixel_idx][0] *= pixel_color.x;
    image_buffer[pixel_idx][1] *= pixel_color.y;
    image_buffer[pixel_idx][2] *= pixel_color.z;
}