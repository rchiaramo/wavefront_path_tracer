const EPSILON = 0.001f;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_PI_2 = 1.5707964f;

struct Pixel {
    r: f32,
    g: f32,
    b: f32
}

struct Sphere {
    center: vec4f,
    radius: f32,
    mat_idx: u32,
    mat_type: u32,
}

struct Material {
    albedo: vec4f,
    fuzz: f32,
    refract_idx: f32,
    mat_type: u32
}

struct Ray {
    origin: vec4f,
    direction: vec4f,
    invDirection: vec3f,
    pixel_idx: u32
}

struct HitPayload {
    t: f32,
    ray_idx: u32,
    sphere_idx: u32,
    mat_type: u32
}

struct FrameBuffer {
    width: u32,
    height: u32,
    frame: u32,
    sample_number: u32
}

@group(0) @binding(0) var<storage, read_write> image_buffer: array<Pixel>;
@group(0) @binding(1) var<uniform> frame_buffer: FrameBuffer;
@group(0) @binding(2) var<storage, read_write> ray_buffer: array<Ray>;
@group(0) @binding(3) var<storage, read_write> extension_ray_buffer: array<Ray>;
@group(0) @binding(4) var<storage, read_write> hit_buffer: array<HitPayload>;
@group(0) @binding(5) var<storage, read_write> counter_buffer: array<atomic<u32>>;
@group(1) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> materials: array<Material>;

@compute @workgroup_size(8,4,1)
fn main(@builtin(global_invocation_id) id: vec3u,
        @builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(num_workgroups) num_workgroups: vec3u) {

    let workgroup_index = workgroup_id.x +
            workgroup_id.y * num_workgroups.x +
            workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let idx = workgroup_index * 32u + local_index;
    if idx >= counter_buffer[1] {
            return;
    }

    // set up the rng
    let image_size = vec2u(frame_buffer.width, frame_buffer.height);
    var rng_state:u32 = init_rng(id.xy, image_size, frame_buffer.frame);
    advance(&rng_state, frame_buffer.sample_number * 10u);

    // set the appropriate indices
    let payload = hit_buffer[idx];
    let ray_idx = payload.ray_idx;
    let ray = ray_buffer[ray_idx];
    let pixel_idx = ray.pixel_idx;
    let sphere = spheres[payload.sphere_idx];
    let mat_idx = sphere.mat_idx;

    // load the stored pixel color
    var pixel_color = vec3f(image_buffer[pixel_idx].r, image_buffer[pixel_idx].g, image_buffer[pixel_idx].b);

    // multiply the new contribution in
    pixel_color *= materials[mat_idx].albedo.xyz;

    image_buffer[pixel_idx].r = pixel_color.x;
    image_buffer[pixel_idx].g = pixel_color.y;
    image_buffer[pixel_idx].b = pixel_color.z;

    // determine the extension ray
    let mat_type = payload.mat_type;
    let p = ray.origin + payload.t * ray.direction;
    // annoying but Rays are all vec4 while I want to work below with vec3
    let n = normalize(p - sphere.center).xyz;

    var extension_ray = Ray();
    extension_ray.pixel_idx = ray.pixel_idx;
    extension_ray.origin = p;

    // the code below determines extension rays based on material type
    var extension_direction = vec3f(0.0, 0.0, 0.0);
    switch (mat_type) {
        case 0u, default {
            var random_bounce: vec3f = normalize(rng_next_vec3in_unit_sphere(&rng_state));

            extension_direction = n + random_bounce;
            if length(extension_direction) < 0.001 {
                extension_direction = n;
            }
        }
        case 1u {
            var random_bounce: vec3f = normalize(rng_next_vec3in_unit_sphere(&rng_state));
            let fuzz: f32 = materials[mat_idx].fuzz;
            extension_direction = reflect(ray.direction.xyz, n) + fuzz * random_bounce;
        }
        case 2u {
            let refract_idx: f32 = materials[mat_idx].refract_idx;
            var norm: vec3f = n;
            let uv = normalize(ray.direction.xyz);
            var cos_theta: f32 = min(dot(norm, -uv), 1.0); // as uv represents incoming, -uv is outgoing direction
            var eta_over_eta_prime: f32 = 0.0;

            // in old code, the normal vector was always determined at the time of hit and properly directioned
            // i.e. I determined if the hit was on the outside/front face by taking dot product of imcoming ray with
            // the normal; if it was negative front_face was false and norm *= -1, so normal pointed inward
            // in the case of a ray from inside hitting, dot(-inDir, norm) would be positive
            //
            // now i'm not doing that; so first I need to see if dot(norm, -uv) > 0, ie the incoming ray is on the
            // outside, as norm is ALWAYS facing outward; if so, use 1/refract_index
            if cos_theta >= 0.0 {
                eta_over_eta_prime = 1.0 / refract_idx;
            } else {
            // however, if dot(norm, -uv) < 0, the incoming ray is on the inside; now I need to flip the norm to face
            // inside; my initial calc of cos_theta is also off by a sign as the norm wasn't pointing the right way
                eta_over_eta_prime = refract_idx;
                norm *= -1.0;
                cos_theta *= -1.0;
            }

            let reflectance: f32 = schlick(cos_theta, eta_over_eta_prime);
            var refract_direction: vec3f = vec3f(0.0);

            if refract(uv, norm, eta_over_eta_prime, &refract_direction) {
                if reflectance > rng_next_float(&rng_state) {
                    extension_direction = reflect(uv, norm);
                } else {
                    extension_direction = refract_direction;
                }
            } else {
                extension_direction = reflect(uv, norm);
            }
        }
    }
    extension_ray.invDirection = 1.0 / extension_direction;
    extension_ray.direction = vec4f(extension_direction, 0.0);
    extension_ray_buffer[atomicAdd(&counter_buffer[2], 1u)] = extension_ray;
}

fn schlick(cosine: f32, refraction_index: f32) -> f32 {
    var r0 = (1f - refraction_index) / (1f + refraction_index);
    r0 = r0 * r0;
    return r0 + (1f - r0) * pow((1f - cosine), 5f);
}

fn reflect(r: vec3f, n: vec3f) -> vec3f {
    return r - 2.0 * dot(r,n) * n;
}

fn refract(uv: vec3f, n: vec3f, ri: f32, dir: ptr<function, vec3f>) -> bool {
    let cos_theta: f32 = dot(uv, n);
    let k: f32 = 1 - ri * ri * (1 - cos_theta * cos_theta);
    if k >= 0.0 {
        *dir = ri * uv - (ri * cos_theta + sqrt(k)) * n;
        return true;
    }
    return false;
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
    let cos_theta = 1f - 2f * rng_next_float(state);
    let sin_theta = sqrt(1 - cos_theta * cos_theta);
    let phi = 2.0 * PI * rng_next_float(state);

    let x = r * sin_theta * cos(phi);
    let y = r * sin_theta * sin(phi);
    let z = r * cos_theta;

    return vec3(x, y, z);
}

fn rng_next_float(state: ptr<function, u32>) -> f32 {
    let x = rng_next_int(state);
    return f32(x) * 2.3283064365387e-10f;  // / f32(0xffffffffu - 1f);
}

fn init_rng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32 {
    let seed = dot(pixel, vec2<u32>(1u, resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(seed);
}

// This is (I think) a correct implementation of the PCG-RXS-M-XS rng;  the
// LCG is the state, and we can advance ahead in it if we like
// the next int function returns an output function based on the LCG
fn rng_next_int(state: ptr<function, u32>) -> u32 {
    // PCG hash RXS-M-XS
    let new_state = *state * 747796405u + 2891336453u;  // LCG
    *state = new_state;  // store this as the new state
    // below is the output function for RXS-M-XS
    let word = ((new_state >> ((new_state >> 28u) + 4u)) ^ new_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn advance(state: ptr<function, u32>, advance_by: u32) {
    var acc_mult = 1u;
    var acc_plus = 0u;
    var cur_mult = 747796405u;
    var cur_plus = 2891336453u;
    var delta = advance_by;
    while delta > 0 {
        if delta == 1 {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1u) * cur_plus;
        cur_mult *= cur_mult;
        delta = delta >> 1;
    }
    *state = *state * acc_mult + acc_plus;
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