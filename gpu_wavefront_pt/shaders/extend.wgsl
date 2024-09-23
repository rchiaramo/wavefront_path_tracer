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
    mat_type: u32,
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

const STACKSIZE:u32 = 10;

@group(0) @binding(0) var<storage, read> ray_buffer: array<Ray>;
@group(0) @binding(1) var<storage, read_write> counter_buffer: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> miss_buffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> hit_buffer: array<HitPayload>;
@group(1) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> bvhTree: array<BVHNode>;

@compute @workgroup_size(8,4,1)
fn main(@builtin(workgroup_id) workgroup_id: vec3u,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(num_workgroups) num_workgroups: vec3u) {

    let workgroup_index = workgroup_id.x +
            workgroup_id.y * num_workgroups.x +
            workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let idx = workgroup_index * 32u + local_index;

    if idx >= counter_buffer[2] {
        return;
    }

    let ray = ray_buffer[idx];
    var payload = HitPayload();

    if trace_ray(ray, &payload) {
        payload.ray_idx = idx;
        hit_buffer[atomicAdd(&counter_buffer[1], 1u)] = payload;
    } else {
        miss_buffer[atomicAdd(&counter_buffer[0], 1u)] = idx;
    }
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
    let sphere_center = sphere.center;
    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = dot(ray.direction, ray.origin - sphere_center);
    let c: f32 = dot(ray.origin - sphere_center, ray.origin - sphere_center) -
        sphere.radius * sphere.radius;
    let discrim: f32 = b * b - a * c;

    if (discrim >= 0) {
        var t: f32 = (-b - sqrt(discrim)) / a;
        if (t > t_min && t < t_nearest) {
            *payload = HitPayload(t, 0, sphere_idx, sphere.mat_type);
            return true;
        }

        t = (-b + sqrt(discrim)) / a;
        if (t > t_min && t < t_nearest) {
            *payload = HitPayload(t, 0, sphere_idx, sphere.mat_type);
            return true;
        }
    }
    return false;
}