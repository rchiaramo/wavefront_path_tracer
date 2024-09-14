use crate::bvh::BVHNode;
use crate::gpu_buffer::GPUStorageBuffer;
use crate::gpu_structs::{GPUSamplingParameters};
use crate::material::Material;
use crate::sphere::Sphere;
use crate::gpu_rng::{GpuRng};
use wavefront_common::gpu_structs::GPUFrameBuffer;
use wavefront_common::camera_controller::{GPUCamera};
use glam::{Mat4, UVec2, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use rayon::iter::{ParallelIterator, IntoParallelIterator, IntoParallelRefIterator};
use wgpu::Queue;
use crate::ray::Ray;
use wavefront_common::constants::{PI, USE_BVH};

pub struct ComputeShader {
    spheres: Vec<Sphere>,
    materials: Vec<Material>,
    bvh_tree: Vec<BVHNode>,
    camera_data: GPUCamera,
    sampling_parameters: GPUSamplingParameters,
    frame_buffer: [u32;4],
    pixels: Vec<[f32;3]>,
    rng_state: GpuRng,
}

#[derive(Copy, Clone, Default)]
struct HitPayload {
    t: f32,
    p: Vec3,
    n: Vec3,
    idx: u32,
}

// Frame buffer
// [width, height, frame, accumulated_samples]

impl ComputeShader {

    pub fn new(spheres: Vec<Sphere>,
               materials: Vec<Material>,
               bvh_tree: Vec<BVHNode>,
               camera_data: GPUCamera,
               sampling_parameters: GPUSamplingParameters,
               frame_buffer: GPUFrameBuffer,
               max_size: u32) -> Self {

        let pixels = vec![[0.0f32; 3]; max_size as usize];
        Self {
            spheres,
            materials,
            bvh_tree,
            camera_data,
            sampling_parameters,
            frame_buffer: frame_buffer.into_array(),
            pixels,
            rng_state: GpuRng::default(),
        }
    }

    pub fn queue_camera(&mut self, gpucamera: GPUCamera) {
        self.camera_data = gpucamera;
    }

    pub fn queue_sampling(&mut self, sampling_parameters: GPUSamplingParameters) {
        self.sampling_parameters = sampling_parameters
    }

    pub fn queue_frame(&mut self, frame: GPUFrameBuffer) {
        self.frame_buffer = frame.into_array();
    }

    pub fn run(&self, ray_buffer: &[Ray]) -> &Vec<[f32; 3]> {
        let (width, height) = (self.frame_buffer[0] as usize, self.frame_buffer[1] as usize);
        let image_size = width * height;
        let mut image = vec![[0f32; 3]; image_size];

        let bands: Vec<(usize, &mut [[f32; 3]])> = image.chunks_mut(width).enumerate().collect();
        bands.into_par_iter().for_each(|(i, image_row)| {
            let screen_pos = UVec2::new(0u32, i as u32);
            let mut rng_state = GpuRng::init_rng(screen_pos, (width, height), self.frame_buffer[2]);
            self.process_row(ray_buffer, image_row, i, width, &mut rng_state);
        });

        // if the accumulator = 1, clear the buffer first, otherwise add to it
        if self.sampling_parameters.clear_image() == 1 {
            for idx in 0..image_size {
                self.pixels[idx] = image[idx];
            }
        } else {
            for idx in 0..image_size {
                self.pixels[idx][0] += image[idx][0];
                self.pixels[idx][1] += image[idx][1];
                self.pixels[idx][2] += image[idx][2];
            }
        }
        &self.pixels
    }

    pub fn process_row(&self, ray_buffer: &[Ray], pixels: &mut [[f32;3]],
                               row: usize, width: usize, rng_state: &mut GpuRng) {
        for x in 0..width {
            let idx = x + width * row;
            let ray = ray_buffer[idx];
            pixels[idx] = self.ray_color(ray, rng_state).to_array();
        }
    }

    fn ray_color(&self, primary_ray: Ray, rng_state: &mut GpuRng) -> Vec3 {
        // for every ray, we want to trace the ray through num_bounces
        // ray_color calls traceRay to get a hit, then calls it again
        // with new bounce ray
        let mut next_ray = primary_ray.clone();
        let mut throughput = Vec3::ONE;
        let mut pixel_color = Vec3::ZERO;
        for _i in 0 .. self.sampling_parameters.num_bounces() {
            let mut pay_load = HitPayload::default();

            if self.trace_ray(next_ray, &mut pay_load) {
                // depending on what kind of material, I need to find the scatter ray and the attenuation
                let mat_idx:u32 = self.spheres[pay_load.idx as usize].material_idx();
                next_ray = self.getScatterRay_parallel(next_ray, mat_idx, pay_load, rng_state);

                throughput *= self.materials[mat_idx as usize].albedo().xyz();
            } else {
                let a: f32 = 0.5 * (primary_ray.direction.y + 1.0);
                pixel_color = throughput * ((1.0 - a) * Vec3::ONE + a * Vec3::new(0.5, 0.7, 1.0));
                break;
            }
        }
        pixel_color
    }

    fn trace_ray(&self, ray: Ray, hit: &mut HitPayload) -> bool {
        // runs through objects in the scene and returns true if the ray hits one, and updates
        // the hitPayload with the closest hit

        let mut nearest_hit: f32 = 1e29;
        let sphere_count = self.spheres.len();
        let mut temp_hit_payload = HitPayload::default();

        if USE_BVH {
            // this is where I will implement the BVH tree search rather than using a full primitive search
            let mut stack = [0usize; 32];
            let mut stack_pointer = 0usize;
            let mut node_index = 0usize;

            loop {
                if self.bvh_tree[node_index].prim_count > 0 {
                    // this is a leaf and has primitives, so check to see if primitives are hit
                    for idx in 0..self.bvh_tree[node_index].prim_count {
                        let mut new_hit_payload = HitPayload::default();
                        let i = self.bvh_tree[node_index].left_first;
                        if self.hit(ray, i + idx, 0.001, nearest_hit, &mut new_hit_payload) {
                            nearest_hit = new_hit_payload.t;
                            temp_hit_payload = new_hit_payload;
                        }
                    }
                    if stack_pointer == 0 {
                        break;
                    } else {
                        stack_pointer -= 1;
                        node_index = stack[stack_pointer];
                        continue;
                    }
                } else {
                    // if not a leaf, check to see if this node's children have been hit
                    let mut left_idx = self.bvh_tree[node_index].left_first as usize;
                    let mut right_idx = left_idx + 1;
                    let mut t_left = self.hit_bvh_node(&ray, &self.bvh_tree[left_idx]);
                    let mut t_right = self.hit_bvh_node(&ray, &self.bvh_tree[right_idx]);

                    // make sure what we call "left" is the closer distance; swap if not
                    if t_left > t_right {
                        let temp = t_left;
                        t_left = t_right;
                        t_right = temp;

                        left_idx += 1;
                        right_idx -= 1;
                    }
                    // if t_left > nearest_hit, nothing left to do with this node
                    if t_left > nearest_hit {
                        if stack_pointer == 0 {
                            break;
                        } else {
                            stack_pointer -= 1;
                            node_index = stack[stack_pointer];
                            continue;
                        }
                    } else {
                        node_index = left_idx;
                        if t_right < nearest_hit {
                            stack[stack_pointer] = right_idx;
                            stack_pointer += 1;
                        }
                    }
                }
            }
        } else {
            // this is the old code with full primitive search
            for i in 0..sphere_count {
                let mut new_hit_payload = HitPayload::default();

                // I could update this code so that hit only determines if a hit happened and, if it did,
                // modifies the nearest_hit_t and stores the nearest_index
                if self.hit(ray, i as u32, 0.001, nearest_hit, &mut new_hit_payload) {
                    nearest_hit = new_hit_payload.t;
                    temp_hit_payload = new_hit_payload;
                }
            }
        }
        // then after looping through the objects, we will know the nearest_hit_t and the index; we could call
        // for the payload then (as opposed to filling it out every time we hit a closer sphere)
        if nearest_hit < 1e29 {
            *hit = temp_hit_payload;
            return true;
        }
        false
    }

    fn hit_bvh_node(&self, ray: &Ray, node: &BVHNode) -> f32 {
        let t_x_min = (node.aabb_min.x - ray.origin.x) / ray.direction.x;
        let t_x_max = (node.aabb_max.x - ray.origin.x) / ray.direction.x;
        let mut tmin = t_x_min.min(t_x_max);
        let mut tmax = t_x_max.max(t_x_min);
        let t_y_min = (node.aabb_min.y - ray.origin.y) / ray.direction.y;
        let t_y_max = (node.aabb_max.y - ray.origin.y) / ray.direction.y;
        tmin = t_y_min.min(t_y_max).max(tmin);
        tmax = t_y_max.max(t_y_min).min(tmax);
        let t_z_min = (node.aabb_min.z - ray.origin.z) / ray.direction.z;
        let t_z_max = (node.aabb_max.z - ray.origin.z) / ray.direction.z;
        tmin = t_z_min.min(t_z_max).max(tmin);
        tmax = t_z_max.max(t_z_min).min(tmax);

        if tmin > tmax || tmax <= 0.0 {
            1e30
        } else {
            tmin
        }
    }

    fn hit(&self, ray: Ray, sphere_idx: u32, t_min: f32, t_nearest: f32, payload: & mut HitPayload) -> bool {
        // checks if the ray intersects the sphere given by sphere_idx; if so, returns true and modifies
        // a hitPayload to give the details of the hit
        let sphere: Sphere = self.spheres[sphere_idx as usize];
        let sphere_center = sphere.center.xyz();
        let a: f32 = ray.direction.dot(ray.direction);
        let b: f32 = ray.direction.dot(ray.origin - sphere_center);
        let c: f32 = (ray.origin - sphere_center).dot(ray.origin - sphere_center) -
            sphere.radius() * sphere.radius();
        let discrim: f32 = b * b - a * c;


        if (discrim >= 0.0) {
            let mut t = (-b - discrim.sqrt()) / a;
            if (t > t_min && t < t_nearest) {
                *payload = self.hit_sphere(t, ray, sphere, sphere_idx);
                return true;
            }

            t = (-b + discrim.sqrt()) / a;
            if (t > t_min && t < t_nearest) {
                *payload = self.hit_sphere(t, ray, sphere, sphere_idx);
                return true;
            }
        }
        false
    }

    fn hit_sphere(&self, t: f32, ray: Ray, sphere: Sphere, idx: u32) -> HitPayload {
        // make the hitPayload struct
        // note that decision here is that normals ALWAYS point out of the sphere
        // thus, to test whether a ray is intersecting the sphere from the inside vs the outside,
        // the dot product of the ray direction and the normal is evaluated;  if negative, ray comes
        // from outside; if positive, ray comes from within
        let p = ray.origin + t * ray.direction;
        let mut n = (p - sphere.center.xyz()).normalize();

        HitPayload {t, p, n, idx}
    }

    fn get_scatter_ray(&self, in_ray: Ray,
                       mat_idx: u32,
                       hit: HitPayload, rng_state: &mut GpuRng)
                       -> Ray {

        let origin = hit.p;
        let mat_type: u32 = self.materials[mat_idx as usize].material_type();
        let mut direction = Vec3::ZERO;

        match mat_type {
            0 => {
                let random_bounce = rng_state.rngNextVec3InUnitSphere().normalize();

                direction = hit.n + random_bounce;
                if direction.length() < 0.0001 {
                    direction = hit.n;
                }
            }
            1 => {
                let random_bounce = rng_state.rngNextVec3InUnitSphere().normalize();

                let fuzz: f32 = self.materials[mat_idx as usize].fuzz();
                direction = self.reflect(in_ray.direction, hit.n) + fuzz * random_bounce;
            }
            2 => {
                let refract_idx: f32 = self.materials[mat_idx as usize].refract_index();
                let mut norm= hit.n.clone();
                let uv = in_ray.direction.normalize();
                let mut cos_theta = norm.dot(-uv).min(1.0); // as uv represents incoming, -uv is outgoing direction
                let mut eta_over_eta_prime: f32 = 0.0;

                if cos_theta >= 0.0 {
                    eta_over_eta_prime = 1.0 / refract_idx;
                } else {
                    eta_over_eta_prime = refract_idx;
                    norm *= -1.0;
                    cos_theta *= -1.0;
                }

                let reflectance: f32 = self.schlick(cos_theta, eta_over_eta_prime);
                let mut refract_direction = Vec3::ZERO;
                let cond= rng_state.rngNextFloat();

                if self.refract(uv, norm, eta_over_eta_prime, &mut refract_direction) {
                    if reflectance > cond {
                        direction = self.reflect(uv, norm);
                    } else {
                        direction = refract_direction;
                    }
                } else {
                    direction = self.reflect(uv, norm);
                }
            }
            _ => {}
        }
        Ray { origin, direction }
    }

    fn schlick(&self, cosine: f32, refraction_index: f32) -> f32 {
        let mut r0 = (1f32 - refraction_index) / (1f32 + refraction_index);
        r0 = r0 * r0;
        r0 + (1f32 - r0) * (1f32 - cosine).powi(5)
    }

    fn reflect(&self, r: Vec3, n: Vec3) -> Vec3 {
        r - 2.0 * r.dot(n) * n
    }

    fn refract(&self, uv: Vec3, n: Vec3, ri: f32, dir: &mut Vec3) -> bool {
        let cos_theta: f32 = uv.dot(n);
        let k: f32 = 1.0 - ri * ri * (1.0 - cos_theta * cos_theta);
        if k >= 0.0 {
            *dir = ri * uv - (ri * cos_theta + k.sqrt()) * n;
            return true;
        }
        false
    }
}