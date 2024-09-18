
struct Ray {
    origin: vec3f,
    direction: vec3f,
    invDirection: vec3f,
}

@group(0) @binding(0) var<storage, read_write> image_buffer: array<array<f32, 3>>;
@group(0) @binding(1) var<storage, read> ray_buffer: array<Ray>;
@group(0) @binding(2) var<storage, read_write> miss_buffer: array<u32>;

@compute @workgroup_size(8,8,1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3u) {

    let workgroup_index = workgroup_id.x +
        workgroup_id.y * num_workgroups.x +
        workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let ind = workgroup_index * 64u + local_index;

    let ray_idx = miss_buffer[ind];
    let ray = ray_buffer[ray_idx];
    let a: f32 = 0.5 * (ray.direction.y + 1.0);
    let pixel_color = (1.0 - a) * vec3f(1.0, 1.0, 1.0) + a * vec3f(0.5, 0.7, 1.0);

    image_buffer[ray_idx][0] = pixel_color.x;
    image_buffer[ray_idx][1] = pixel_color.y;
    image_buffer[ray_idx][2] = pixel_color.z;
}