@group(0) @binding(0) var<storage, read_write> image_buffer: array<array<f32, 3>>;
@group(0) @binding(1) var<storage, read_write> accumulated_image_buffer: array<array<f32, 3>>;

@compute @workgroup_size(8,8,1)
fn main(@builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3u) {

    let workgroup_index = workgroup_id.x +
        workgroup_id.y * num_workgroups.x; // +
//        workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let idx = workgroup_index * 64u + local_index;

    accumulated_image_buffer[idx][0] += image_buffer[idx][0];
    accumulated_image_buffer[idx][1] += image_buffer[idx][1];
    accumulated_image_buffer[idx][2] += image_buffer[idx][2];
}