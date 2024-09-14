const EPSILON = 0.001f;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_PI_2 = 1.5707964f;

struct Ray {
    origin: vec3f,
    direction: vec3f,
    invDirection: vec3f,
}

struct CameraData {
    pos: vec4f,
    pitch: f32,
    yaw: f32,
    defocusRadius: f32,
    focusDistance: f32
}

struct FrameBuffer {
    width: u32,
    height: u32,
    frame: u32,
    accumulated_samples: u32
}

struct ProjectionBuffer {
    invProj: mat4x4f
}

struct ViewBuffer {
    view: mat4x4f
}

@group(0) @binding(0) var<storage, read_write> ray_buffer: array<Ray>;
@group(1) @binding(0) var<uniform> frame_buffer: FrameBuffer;
@group(1) @binding(1) var<uniform> camera: CameraData;
@group(1) @binding(2) var<uniform> projection_matrix: ProjectionBuffer;
@group(1) @binding(3) var<uniform> view_matrix: ViewBuffer;

@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let width = frame_buffer.width;
    let height = frame_buffer.height;
    let screen_pos = id.xy;
    let idx = id.x + id.y * width;

    var rng_state:u32 = init_rng(screen_pos, vec2(width, height), frame_buffer.frame);

    var offset: vec3f = rng_next_vec3in_unit_disk(&rng_state);
    var ray: Ray;

    var ndc_point = vec2((f32(id.x) + offset.x) / f32(width), 1.0 - (f32(id.y) + offset.y) / f32(height) );
    ndc_point = 2.0 * ndc_point - 1.0;
    var proj_point = projection_matrix.invProj * vec4<f32>(ndc_point.xy, 1.0, 1.0);
    proj_point = proj_point / proj_point.w;

    ray.origin = camera.pos.xyz;

    if camera.defocusRadius > 0.0 {
        offset = rng_next_vec3in_unit_disk(&rng_state);
        var p_lens = vec4f((camera.defocusRadius * offset).xyz, 1.0);
        var lens_origin = view_matrix.view * p_lens;
        lens_origin = lens_origin / lens_origin.w;
        ray.origin = lens_origin.xyz;

        let tf = camera.focusDistance / proj_point.z;
        proj_point = tf * proj_point - p_lens;
    }

    let ray_dir = view_matrix.view * vec4<f32>(proj_point.xyz, 0.0);

    ray.direction = normalize(ray_dir.xyz);
    ray.invDirection = 1.0 / ray.direction;

    ray_buffer[idx] = ray;
}

fn rng_next_in_unit_hemisphere(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rng_next_float(state);
    let r2 = rng_next_float(state);

    let phi = 2.0 * PI * r1;
    let sin_theta = sqrt(1.0 - r2 * r2);

    let x = cos(phi) * sin_theta;
    let y = sin(phi) * sin_theta;
    let z = r2;

    return vec3(x, y, z);
}

fn rng_next_vec3in_unit_disk(state: ptr<function, u32>) -> vec3<f32> {
    // r^2 is distributed as U(0, 1).
    let r = sqrt(rng_next_float(state));
    let alpha = 2.0 * PI * rng_next_float(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0.0);
}

fn rng_next_vec3in_unit_sphere(state: ptr<function, u32>) -> vec3<f32> {
    // probability density is uniformly distributed over r^3
    let r = pow(rng_next_float(state), 0.33333f);
    // and need to distribute theta according to arccos(U[-1,1])
    // let theta = acos(2f * rng_next_float(state) - 1.0);
    let cos_theta = 2f * rng_next_float(state) - 1f;
    let sin_theta = sqrt(1 - cos_theta * cos_theta);
    let phi = 2.0 * PI * rng_next_float(state);

    let x = r * sin_theta * cos(phi);
    let y = r * sin_theta * sin(phi);
    let z = r * cos_theta;

    return vec3(x, y, z);
}

fn rng_next_float(state: ptr<function, u32>) -> f32 {
    rng_next_int(state);
    return f32(*state) * 2.3283064365387e-10f;  // / f32(0xffffffffu - 1f);
}

fn init_rng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32 {
    let seed = dot(pixel, vec2<u32>(1u, resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(seed);
}

fn rng_next_int(state: ptr<function, u32>) {
    // PCG hash RXS-M-XS
    let old_state = *state + 747796405u + 2891336453u;
    let word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    *state = (word >> 22u) ^ word;
}

fn jenkins_hash(input: u32) -> u32 {
    var x = input;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}