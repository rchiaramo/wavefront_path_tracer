const EPSILON = 0.001f;

const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_PI_2 = 1.5707964f;
const USE_BVH = true;

struct BVHNode {
    aabbMin: vec3f,
    leftFirst: u32,
    aabbMax: vec3f,
    primCount: u32,
}

struct Sphere {
    center: vec4f,
    radius: f32,
    mat_idx: u32,
}

struct Material {
    albedo: vec4f,
    fuzz: f32,
    refract_idx: f32,
    mat_type: u32
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
    invDirection: vec3f,
}

struct HitPayload {
    t: f32,
    p: vec3f,
    n: vec3f,
    idx: u32,
}

struct CameraData {
    pos: vec4f,
    pitch: f32,
    yaw: f32,
    defocusRadius: f32,
    focusDistance: f32
}

struct SamplingParameters {
    samples_per_frame: u32,
    num_bounces: u32,
    clear_image_buffer: u32
}

struct FrameBuffer {
    width: u32,
    height: u32,
    frame: u32,
    accumulated_samples: u32
}

const STACKSIZE:u32 = 10;

@group(0) @binding(0) var<storage, read_write> image_buffer: array<array<f32, 3>>;
@group(0) @binding(1) var<uniform> frame_buffer: FrameBuffer;
@group(0) @binding(2) var<storage, read> ray_buffer: array<Ray>;
@group(1) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> materials: array<Material>;
@group(1) @binding(2) var<storage, read> bvhTree: array<BVHNode>;
@group(2) @binding(0) var<uniform> camera: CameraData;
@group(2) @binding(1) var<uniform> sampling_parameters: SamplingParameters;

@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) id: vec3u) {

    let image_size = vec2(frame_buffer.width, frame_buffer.height);
    let screen_pos = id.xy;
    let idx = id.x + id.y * image_size.x;

    // load the stored pixel color
    var pixel_color: vec3f = vec3f(image_buffer[idx][0], image_buffer[idx][1], image_buffer[idx][2]);
    var rng_state:u32 = init_rng(screen_pos, image_size, frame_buffer.frame);
    advance(&rng_state, 120u);

    if (sampling_parameters.clear_image_buffer == 1) {
        pixel_color = vec3f(0.0, 0.0, 0.0);
    }

    for (var i: u32 = 0; i < sampling_parameters.samples_per_frame; i++) {
        var ray: Ray = ray_buffer[idx];
        pixel_color += ray_color(ray, &rng_state);
    }

    image_buffer[idx][0] = pixel_color.x;
    image_buffer[idx][1] = pixel_color.y;
    image_buffer[idx][2] = pixel_color.z;
}

fn ray_color(primary_ray: Ray, state: ptr<function, u32>) -> vec3<f32> {
    // for every ray, we want to trace the ray through num_bounces
    // ray_color calls traceRay to get a hit, then calls it again
    // with new bounce ray
    var next_ray = primary_ray;
    var throughput: vec3f = vec3f(1.0);
    var pixel_color: vec3f = vec3f(0.0);
    for (var i: u32 = 0; i < sampling_parameters.num_bounces; i++) {
        var pay_load = HitPayload();

        if trace_ray(next_ray, &pay_load) {
            // depending on what kind of material, I need to find the scatter ray and the attenuation
            let mat_idx:u32 = spheres[pay_load.idx].mat_idx;
            get_scatter_ray(&next_ray, mat_idx, &pay_load, state);

            throughput *= materials[mat_idx].albedo.xyz;
        } else {
            let a: f32 = 0.5 * (primary_ray.direction.y + 1.0);
            pixel_color = throughput * ((1.0 - a) * vec3f(1.0, 1.0, 1.0) + a * vec3f(0.5, 0.7, 1.0));
            break;
        }
    }
    return pixel_color;
}

fn trace_ray(ray: Ray, hit: ptr<function, HitPayload>) -> bool {
    // runs through objects in the scene and returns true if the ray hits one, and updates
    // the hitPayload with the closest hit

    var nearest_hit: f32 = 1e30;
    let sphere_count = arrayLength(&spheres);
    var temp_hit_payload = HitPayload();

    if USE_BVH {
        // this is where I will implement the BVH tree search rather than using a full primitive search
        var stack = array<BVHNode, STACKSIZE>();
        var stackPointer:u32 = 0;
        var node: BVHNode = bvhTree[0];
        while true {
            if node.primCount > 0 {
                // this is a leaf and has primitives, so check to see if primitives are hit
                for (var idx:u32 = 0; idx < node.primCount; idx++) {
                    var new_hit_payload = HitPayload();
                    if hit(ray, node.leftFirst + idx, 0.001, nearest_hit, &new_hit_payload) {
                        nearest_hit = new_hit_payload.t;
                        temp_hit_payload = new_hit_payload;
                    }
                }
                // we are now done with this node; if stack is empty, break; otherwise
                // set node based on the stack
                if stackPointer == 0 {
                    break;
                }
                else {
                    stackPointer--;
                    node = stack[stackPointer];
                    continue;
                }
            } else {
                // if not a leaf, check to see if this node's children have been hit
                var leftChild = bvhTree[node.leftFirst];
                var rightChild = bvhTree[node.leftFirst + 1];
                var t_left:f32 = hit_bvh_node(leftChild, ray, nearest_hit);
                var t_right:f32 = hit_bvh_node(rightChild, ray, nearest_hit);

                // make sure the left node is always the closer node
                var swap = false;
                if t_left > t_right {
                    let temp_t:f32 = t_left;
                    t_left = t_right;
                    t_right = temp_t;

                    var temp = leftChild;
                    leftChild = rightChild;
                    rightChild = temp;
                }
                // if the left hit is bigger than nearest hit, no need to do anything else here
                if t_left > nearest_hit {
                    if stackPointer == 0 {
                        break;
                    } else {
                        stackPointer--;
                        node = stack[stackPointer];
                    }
                } else {
                    node = leftChild;
                    // if the rightChild hit distance is also smaller than nearest_hit, save to the stack
                    if t_right < nearest_hit {
                        stack[stackPointer] = rightChild;
                        stackPointer++;
                    }
                }
            }
        }
    } else {
        // this is the old code with full primitive search
        for (var i: u32 = 0; i < sphere_count; i++) {
            var new_hit_payload = HitPayload();

            // I could update this code so that hit only determines if a hit happened and, if it did,
            // modifies the nearest_hit_t and stores the nearest_index
            if hit(ray, i, 0.001, nearest_hit, &new_hit_payload) {
                nearest_hit = new_hit_payload.t;
                temp_hit_payload = new_hit_payload;
            }
        }
    }

    // then after looping through the objects, we will know the nearest_hit_t and the index; we could call
    // for the payload then (as opposed to filling it out every time we hit a closer sphere)
    if nearest_hit < 1e30 {
        *hit = temp_hit_payload;
        return true;
    }
    return false;
}

fn hit_bvh_node(node: BVHNode, ray: Ray, nearest_hit: f32) -> f32 {
    let t_x_min = (node.aabbMin.x - ray.origin.x) * ray.invDirection.x;
    let t_x_max = (node.aabbMax.x - ray.origin.x) * ray.invDirection.x;
    var tmin = min(t_x_min, t_x_max);
    var tmax = max(t_x_min, t_x_max);
    let t_y_min = (node.aabbMin.y - ray.origin.y) * ray.invDirection.y;
    let t_y_max = (node.aabbMax.y - ray.origin.y) * ray.invDirection.y;
    tmin = max(min(t_y_min, t_y_max), tmin);
    tmax = min(max(t_y_min, t_y_max), tmax);
    let t_z_min = (node.aabbMin.z - ray.origin.z) * ray.invDirection.z;
    let t_z_max = (node.aabbMax.z - ray.origin.z) * ray.invDirection.z;
    tmin = max(min(t_z_min, t_z_max), tmin);
    tmax = min(max(t_z_min, t_z_max), tmax);

    if tmin > tmax || tmax <= 0.0 || tmin > nearest_hit {
        return 1e30;
    } else {
        return tmin;
    }
}

fn hit(ray: Ray, sphere_idx: u32, t_min: f32, t_nearest: f32, payload: ptr<function, HitPayload>) -> bool {
    // checks if the ray intersects the sphere given by sphere_idx; if so, returns true and modifies
    // a hitPayload to give the details of the hit
    let sphere: Sphere = spheres[sphere_idx];
    let sphere_center = sphere.center.xyz;
    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = dot(ray.direction, ray.origin - sphere_center);
    let c: f32 = dot(ray.origin - sphere_center, ray.origin - sphere_center) -
        sphere.radius * sphere.radius;
    let discrim: f32 = b * b - a * c;


    if (discrim >= 0) {
        var t: f32 = (-b - sqrt(discrim)) / a;
        if (t > t_min && t < t_nearest) {
            *payload = hit_sphere(t, ray, sphere, sphere_idx);
            return true;
        }

        t = (-b + sqrt(discrim)) / a;
        if (t > t_min && t < t_nearest) {
            *payload = hit_sphere(t, ray, sphere, sphere_idx);
            return true;
        }
    }
    return false;
}

fn hit_sphere(t: f32, ray: Ray, sphere: Sphere, idx: u32) -> HitPayload {
    // make the hitPayload struct
    // note that decision here is that normals ALWAYS point out of the sphere
    // thus, to test whether a ray in intersecting the sphere from the inside vs the outside,
    // the dot product of the ray direction and the normal is evaluated;  if negative, ray comes
    // from outside; if positive, ray comes from within
    let p: vec3f = ray.origin + t * ray.direction;
    let n: vec3f = normalize(p - sphere.center.xyz);

    return HitPayload(t, p, n, idx);
}

fn get_scatter_ray(in_ray: ptr<function, Ray>, mat_idx: u32, hit: ptr<function, HitPayload>, state: ptr<function, u32>) {
    // when we show up here, hit.n is necessarily the outward normal of the sphere
    // we need to orient it correctly
    let pay_load = *hit;
    var ray = Ray();
    ray.origin = pay_load.p;

    let mat_type: u32 = materials[mat_idx].mat_type;

    switch (mat_type) {
        case 0u, default {
            var random_bounce: vec3f = normalize(rng_next_vec3in_unit_sphere(state));

            ray.direction = pay_load.n + random_bounce;
            if length(ray.direction) < 0.001 {
                ray.direction = pay_load.n;
            }
        }
        case 1u {
            var random_bounce: vec3f = normalize(rng_next_vec3in_unit_sphere(state));
            let fuzz: f32 = materials[mat_idx].fuzz;
            ray.direction = reflect((*in_ray).direction, pay_load.n) + fuzz * random_bounce;
        }
        case 2u {
            let refract_idx: f32 = materials[mat_idx].refract_idx;
            var norm: vec3f = pay_load.n;
            let uv = normalize((*in_ray).direction);
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
                if reflectance > rng_next_float(state) {
                    ray.direction = reflect(uv, norm);
                } else {
                    ray.direction = refract_direction;
                }
            } else {
                ray.direction = reflect(uv, norm);
            }
        }
    }
    ray.invDirection = 1.0 / ray.direction;
    *in_ray = ray;
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
    // let theta = acos(2f * rng_next_float(state) - 1.0);
    let cos_theta = 2f * rng_next_float(state) - 1f;
    let sin_theta = sqrt(1 - cos_theta * cos_theta);
    let phi = 2.0 * PI * rng_next_float(state);

    let x = r * sin_theta * cos(phi);
    let y = r * sin_theta * sin(phi);
    let z = r * cos_theta;

    return vec3(x, y, z);
}

fn rng_next_uint_in_range(state: ptr<function, u32>, min: u32, max: u32) -> u32 {
    rng_next_int(state);
    return min + (*state) % (max - min);
}

fn rng_next_float(state: ptr<function, u32>) -> f32 {
    let x = rng_next_int(state);
    return f32(x) * 2.3283064365387e-10f;  // / f32(0xffffffffu - 1f);
}

fn init_rng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32 {
    let seed = dot(pixel, vec2<u32>(1u, resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(seed);
}

fn rng_next_int(state: ptr<function, u32>) -> u32 {
    // PCG hash RXS-M-XS
    let old_state = *state * 747796405u + 2891336453u;
    *state = old_state;
    let word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
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