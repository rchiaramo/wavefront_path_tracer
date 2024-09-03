use crate::bvh::BVHNode;
use crate::gpu_buffer::GPUBuffer;
use crate::gpu_structs::{GPUSamplingParameters};
use crate::material::Material;
use crate::sphere::Sphere;
use wavefront_common::gpu_structs::GPUFrameBuffer;
use wavefront_common::camera_controller::{GPUCamera};
use glam::{Mat4, UVec2, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use rayon::iter::{ParallelIterator, IntoParallelIterator, IntoParallelRefIterator};
use wgpu::Queue;

const EPSILON: f32 = 0.001;

const PI: f32 = 3.1415927;
const FRAC_1_PI: f32 = 0.31830987;
const FRAC_PI_2: f32 = 1.5707964;
const USE_BVH: bool = true;

pub struct ComputeShader {
    spheres: Vec<Sphere>,
    materials: Vec<Material>,
    bvh_tree: Vec<BVHNode>,
    camera_data: GPUCamera,
    sampling_parameters: GPUSamplingParameters,
    inv_proj_matrix: [[f32;4];4],
    view_matrix: [[f32;4];4],
    frame_buffer: [u32;4],
    pixels: Vec<[f32;3]>,
    rngState: GPURNG,
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

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl ComputeShader {
    pub fn new(spheres: Vec<Sphere>,
               materials: Vec<Material>,
               bvh_tree: Vec<BVHNode>,
               camera_data: GPUCamera,
               inv_proj_matrix: [[f32;4];4],
               view_matrix: [[f32;4];4],
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
            inv_proj_matrix,
            view_matrix,
            frame_buffer: frame_buffer.into_array(),
            pixels,
            rngState: GPURNG::default(),
        }
    }

    pub fn queue_camera(&mut self, gpucamera: GPUCamera) {
        self.camera_data = gpucamera;
    }

    pub fn queue_proj(&mut self, proj_mat: [[f32;4];4]) {
        self.inv_proj_matrix = proj_mat;
    }

    pub fn queue_view(&mut self, view_mat: [[f32;4];4]) {
        self.view_matrix = view_mat;
    }

    pub fn queue_sampling(&mut self, sampling_parameters: GPUSamplingParameters) {
        self.sampling_parameters = sampling_parameters
    }

    pub fn queue_frame(&mut self, frame: GPUFrameBuffer) {
        self.frame_buffer = frame.into_array();
    }

    pub fn run_parallel_render(&mut self, queue: &Queue, size: (u32, u32), image_buffer: &mut GPUBuffer) {
        let image_size = (self.frame_buffer[0] as usize, self.frame_buffer[1] as usize);
        let mut image = vec![[0f32;3]; self.pixels.len()];

        let bands: Vec<(usize, &mut [[f32;3]])> = image.chunks_mut(size.0 as usize).enumerate().collect();
        bands.into_par_iter().for_each(|(i, row)| {
            let screen_pos = UVec2::new(0u32, i as u32);
            let mut rngState = GPURNG::initRng(screen_pos, image_size, self.frame_buffer[2]);
            self.main_cs_parallel(row, i, &mut rngState);
        });

        // if the accumulator = 1, clear the buffer first, otherwise add to it
        if self.sampling_parameters.clear_image() == 1 {
            for idx in 0..self.pixels.len() {
                self.pixels[idx] = image[idx];
            }
        } else {
            for idx in 0..self.pixels.len() {
                self.pixels[idx][0] += image[idx][0];
                self.pixels[idx][1] += image[idx][1];
                self.pixels[idx][2] += image[idx][2];
            }
        }
        image_buffer.queue_for_gpu(queue, bytemuck::cast_slice(self.pixels.as_slice()));
    }

    pub fn main_cs_parallel(&self, pixel_row: &mut [[f32;3]], row: usize, rngState: &mut GPURNG) {
        for x in 0..pixel_row.iter().len() {
            let mut pixel_color = Vec3::from_array(pixel_row[x]);

            for _i in 0..self.sampling_parameters.spf() {
                let ray = self.getRay_parallel(x as u32, row as u32, rngState);
                pixel_color += self.rayColor_parallel(ray, rngState);
            }

            pixel_row[x] = pixel_color.to_array();
        }
    }

    fn rayColor_parallel(&self, primaryRay: Ray, rngState: &mut GPURNG) -> Vec3 {
        // for every ray, we want to trace the ray through num_bounces
        // rayColor calls traceRay to get a hit, then calls it again
        // with new bounce ray
        let mut nextRay = primaryRay.clone();
        let mut throughput = Vec3::ONE;
        let mut pixel_color = Vec3::ZERO;
        for _i in 0 .. self.sampling_parameters.num_bounces() {
            let mut payLoad = HitPayload::default();

            if self.TraceRay(nextRay, &mut payLoad) {
                // depending on what kind of material, I need to find the scatter ray and the attenuation
                let mat_idx:u32 = self.spheres[payLoad.idx as usize].material_idx();
                nextRay = self.getScatterRay_parallel(nextRay, mat_idx, payLoad, rngState);

                throughput *= self.materials[mat_idx as usize].albedo().xyz();
            } else {
                let a: f32 = 0.5 * (primaryRay.direction.y + 1.0);
                pixel_color = throughput * ((1.0 - a) * Vec3::ONE + a * Vec3::new(0.5, 0.7, 1.0));
                break;
            }
        }

        return pixel_color;
    }

    pub fn run_render(&mut self, queue: &Queue, size: (u32, u32), image_buffer: &mut GPUBuffer) {
        for y in 0..size.1 {
            for x in 0..size.0 {
                let id = UVec3::new(x, y, 0);
                self.main_cs(id);
            }
        }
        image_buffer.queue_for_gpu(queue, bytemuck::cast_slice(self.pixels.as_slice()));
    }

    pub fn main_cs(&mut self, id: UVec3) {
        let idx = id.x as usize + (self.frame_buffer[0] * id.y) as usize;

        let image_size = (self.frame_buffer[0] as usize, self.frame_buffer[1] as usize);
        let screen_pos = id.xy();
        self.rngState = GPURNG::initRng(screen_pos, image_size, self.frame_buffer[2]);

        // if the accumulator = 0, zero out the image buffer
        if self.sampling_parameters.clear_image() == 1 {
            self.pixels[idx] = [0f32; 3];
        }
        let mut pixel_color = Vec3::from_array(self.pixels[idx]);

        for _i in 0..self.sampling_parameters.spf() {
            let ray = self.getRay(id.x, id.y);
            pixel_color += self.rayColor(ray);
        }

        self.pixels[idx] = pixel_color.to_array();
    }

    fn rayColor(&mut self, primaryRay: Ray) -> Vec3 {
        // for every ray, we want to trace the ray through num_bounces
        // rayColor calls traceRay to get a hit, then calls it again
        // with new bounce ray
        let mut nextRay = primaryRay.clone();
        let mut throughput = Vec3::ONE;
        let mut pixel_color = Vec3::ZERO;
        for _i in 0 .. self.sampling_parameters.num_bounces() {
            let mut payLoad = HitPayload::default();

            if self.TraceRay(nextRay, &mut payLoad) {
                // depending on what kind of material, I need to find the scatter ray and the attenuation
                let mat_idx:u32 = self.spheres[payLoad.idx as usize].material_idx();
                nextRay = self.getScatterRay(nextRay, mat_idx, payLoad);

                throughput *= self.materials[mat_idx as usize].albedo().xyz();
            } else {
                let a: f32 = 0.5 * (primaryRay.direction.y + 1.0);
                pixel_color = throughput * ((1.0 - a) * Vec3::ONE + a * Vec3::new(0.5, 0.7, 1.0));
                break;
            }
        }

        return pixel_color;
    }

    fn TraceRay(&self, ray: Ray, hit: &mut HitPayload) -> bool {
        // runs through objects in the scene and returns true if the ray hits one, and updates
        // the hitPayload with the closest hit

        let mut nearest_hit: f32 = 1e29;
        let sphere_count = self.spheres.len();
        let mut tempHitPayload = HitPayload::default();

        if USE_BVH {
            // this is where I will implement the BVH tree search rather than using a full primitive search
            let mut stack = [0usize; 32];
            let mut stack_pointer = 0usize;
            let mut node_index = 0usize;

            while true {
                if self.bvh_tree[node_index].prim_count > 0 {
                    // this is a leaf and has primitives, so check to see if primitives are hit
                    for idx in 0..self.bvh_tree[node_index].prim_count {
                        let mut newHitPayload = HitPayload::default();
                        let i = self.bvh_tree[node_index].left_first;
                        if self.hit(ray, i + idx, 0.001, nearest_hit, &mut newHitPayload) {
                            nearest_hit = newHitPayload.t;
                            tempHitPayload = newHitPayload;
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
                let mut newHitPayload= HitPayload::default();

                // I could update this code so that hit only determines if a hit happened and, if it did,
                // modifies the nearest_hit_t and stores the nearest_index
                if self.hit(ray, i as u32, 0.001, nearest_hit, &mut newHitPayload) {
                    nearest_hit = newHitPayload.t;
                    tempHitPayload = newHitPayload;
                }
            }
        }
        // then after looping through the objects, we will know the nearest_hit_t and the index; we could call
        // for the payload then (as opposed to filling it out every time we hit a closer sphere)
        if nearest_hit < 1e29 {
            *hit = tempHitPayload;
            return true;
        }
        return false;
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

    fn hit(&self, ray: Ray, sphereIdx: u32, t_min: f32, t_nearest: f32, payload: & mut HitPayload) -> bool {
        // checks if the ray intersects the sphere given by sphereIdx; if so, returns true and modifies
        // a hitPayload to give the details of the hit
        let sphere: Sphere = self.spheres[sphereIdx as usize];
        let sphere_center = sphere.center.xyz();
        let a: f32 = ray.direction.dot(ray.direction);
        let b: f32 = ray.direction.dot(ray.origin - sphere_center);
        let c: f32 = (ray.origin - sphere_center).dot(ray.origin - sphere_center) -
            sphere.radius() * sphere.radius();
        let discrim: f32 = b * b - a * c;


        if (discrim >= 0.0) {
            let mut t = (-b - discrim.sqrt()) / a;
            if (t > t_min && t < t_nearest) {
                *payload = self.hitSphere(t, ray, sphere, sphereIdx);
                return true;
            }

            t = (-b + discrim.sqrt()) / a;
            if (t > t_min && t < t_nearest) {
                *payload = self.hitSphere(t, ray, sphere, sphereIdx);
                return true;
            }
        }
        return false;
    }

    fn hitSphere(&self, t: f32, ray: Ray, sphere: Sphere, idx: u32) -> HitPayload {
        // make the hitPayload struct
        // note that decision here is that normals ALWAYS point out of the sphere
        // thus, to test whether a ray is intersecting the sphere from the inside vs the outside,
        // the dot product of the ray direction and the normal is evaluated;  if negative, ray comes
        // from outside; if positive, ray comes from within
        let p = ray.origin + t * ray.direction;
        let mut n = (p - sphere.center.xyz()).normalize();

        return HitPayload {t, p, n, idx}
    }

    fn getRay_parallel(&self, x: u32, y: u32, rngState: &mut GPURNG) -> Ray {
        let mut offset = rngState.rngNextVec3InUnitDisk();

        let mut point = Vec2::new((x as f32 + offset.x) / self.frame_buffer[0] as f32,
                                  1.0 - (y as f32 + offset.y) / self.frame_buffer[1] as f32);
        point = 2.0 * point - 1.0;
        let mut projPoint = Mat4::from_cols_array_2d(&self.inv_proj_matrix) * Vec4::new(point.x, point.y, 1.0, 1.0);
        projPoint = projPoint / projPoint.w;
        projPoint = projPoint.xyz().extend(0.0);

        let mut origin = self.camera_data.position().xyz();

        if self.camera_data.defocus_radius() > 0.0 {
            offset = rngState.rngNextVec3InUnitDisk();

            let pLens= (self.camera_data.defocus_radius() * offset).extend(1.0);
            let mut lensOrigin = Mat4::from_cols_array_2d(&self.view_matrix) * pLens;
            lensOrigin = lensOrigin / lensOrigin.w;
            origin = lensOrigin.xyz();

            let tf = self.camera_data.focus_distance() / projPoint.z;
            projPoint = tf * projPoint - pLens;
        }

        let rayDir = Mat4::from_cols_array_2d(&self.view_matrix) * projPoint.with_w(0.0);
        let direction = rayDir.xyz().normalize();

        Ray { origin, direction }
    }

    fn getScatterRay_parallel(&self, inRay: Ray,
                     mat_idx: u32,
                     hit: HitPayload, rngState: &mut GPURNG)
                     -> Ray {

        let origin = hit.p;
        let mat_type: u32 = self.materials[mat_idx as usize].material_type();
        let mut direction = Vec3::ZERO;

        match mat_type {
            0 => {
                let randomBounce= rngState.rngNextVec3InUnitSphere().normalize();

                direction = hit.n + randomBounce;
                if direction.length() < 0.0001 {
                    direction = hit.n;
                }
            }
            1 => {
                let randomBounce= rngState.rngNextVec3InUnitSphere().normalize();

                let fuzz: f32 = self.materials[mat_idx as usize].fuzz();
                direction = self.reflect(inRay.direction, hit.n) + fuzz * randomBounce;
            }
            2 => {
                let refract_idx: f32 = self.materials[mat_idx as usize].refract_index();
                let mut norm= hit.n.clone();
                let uv = inRay.direction.normalize();
                let mut cosTheta = norm.dot(-uv).min(1.0); // as uv represents incoming, -uv is outgoing direction
                let mut etaOverEtaPrime: f32 = 0.0;

                if cosTheta >= 0.0 {
                    etaOverEtaPrime = 1.0 / refract_idx;
                } else {
                    etaOverEtaPrime = refract_idx;
                    norm *= -1.0;
                    cosTheta *= -1.0;
                }

                let reflectance: f32 = self.schlick(cosTheta, etaOverEtaPrime);
                let mut refractDirection = Vec3::ZERO;
                let cond= rngState.rngNextFloat();

                if self.refract(uv, norm, etaOverEtaPrime, &mut refractDirection) {
                    if reflectance > cond {
                        direction = self.reflect(uv, norm);
                    } else {
                        direction = refractDirection;
                    }
                } else {
                    direction = self.reflect(uv, norm);
                }
            }
            _ => {}
        }
        Ray { origin, direction }
    }

    fn getRay(&mut self, x: u32, y: u32) -> Ray {
        let mut offset = self.rngState.rngNextVec3InUnitDisk();

        let mut point = Vec2::new((x as f32 + offset.x) / self.frame_buffer[0] as f32,
                                  1.0 - (y as f32 + offset.y) / self.frame_buffer[1] as f32);
        point = 2.0 * point - 1.0;
        let mut projPoint = Mat4::from_cols_array_2d(&self.inv_proj_matrix) * Vec4::new(point.x, point.y, 1.0, 1.0);
        projPoint = projPoint / projPoint.w;
        projPoint = projPoint.xyz().extend(0.0);

        let mut origin = self.camera_data.position().xyz();

        if self.camera_data.defocus_radius() > 0.0 {
            offset = self.rngState.rngNextVec3InUnitDisk();

            let pLens= (self.camera_data.defocus_radius() * offset).extend(1.0);
            let mut lensOrigin = Mat4::from_cols_array_2d(&self.view_matrix) * pLens;
            lensOrigin = lensOrigin / lensOrigin.w;
            origin = lensOrigin.xyz();

            let tf = self.camera_data.focus_distance() / projPoint.z;
            projPoint = tf * projPoint - pLens;
        }

        let rayDir = Mat4::from_cols_array_2d(&self.view_matrix) * projPoint.with_w(0.0);
        let direction = rayDir.xyz().normalize();

        Ray { origin, direction }
    }

    fn getScatterRay(&mut self, inRay: Ray,
                     mat_idx: u32,
                     hit: HitPayload)
        -> Ray {

        let origin = hit.p;
        let mat_type: u32 = self.materials[mat_idx as usize].material_type();
        let mut direction = Vec3::ZERO;

        match mat_type {
            0 => {
                let randomBounce= self.rngState.rngNextVec3InUnitSphere().normalize();

                direction = hit.n + randomBounce;
                if direction.length() < 0.0001 {
                    direction = hit.n;
                }
            }
            1 => {
                let randomBounce= self.rngState.rngNextVec3InUnitSphere().normalize();

                let fuzz: f32 = self.materials[mat_idx as usize].fuzz();
                direction = self.reflect(inRay.direction, hit.n) + fuzz * randomBounce;
            }
            2 => {
                let refract_idx: f32 = self.materials[mat_idx as usize].refract_index();
                let mut norm= hit.n.clone();
                let uv = inRay.direction.normalize();
                let mut cosTheta = norm.dot(-uv).min(1.0); // as uv represents incoming, -uv is outgoing direction
                let mut etaOverEtaPrime: f32 = 0.0;

                if cosTheta >= 0.0 {
                    etaOverEtaPrime = 1.0 / refract_idx;
                } else {
                    etaOverEtaPrime = refract_idx;
                    norm *= -1.0;
                    cosTheta *= -1.0;
                }

                let reflectance: f32 = self.schlick(cosTheta, etaOverEtaPrime);
                let mut refractDirection = Vec3::ZERO;
                let cond= self.rngState.rngNextFloat();

                if self.refract(uv, norm, etaOverEtaPrime, &mut refractDirection) {
                       if reflectance > cond {
                           direction = self.reflect(uv, norm);
                       } else {
                            direction = refractDirection;
                        }
                } else {
                    direction = self.reflect(uv, norm);
                }
            }
            _ => {}
    }
        Ray { origin, direction }
    }

    fn schlick(&self, cosine: f32, refractionIndex: f32) -> f32 {
        let mut r0 = (1f32 - refractionIndex) / (1f32 + refractionIndex);
        r0 = r0 * r0;
        return r0 + (1f32 - r0) * (1f32 - cosine).powi(5)
    }

    fn reflect(&self, r: Vec3, n: Vec3) -> Vec3 {
        return r - 2.0 * r.dot(n) * n;
    }

    fn refract(&self, uv: Vec3, n: Vec3, ri: f32, dir: &mut Vec3) -> bool {
        let cosTheta: f32 = uv.dot(n);
        let k: f32 = 1.0 - ri * ri * (1.0 - cosTheta * cosTheta);
        if k >= 0.0 {
            *dir = ri * uv - (ri * cosTheta + k.sqrt()) * n;
            return true;
        }
        return false;
    }
}

#[derive(Default)]
struct GPURNG {
    state: u32,
}

impl GPURNG {
    fn initRng(pixel: UVec2, resolution: (usize, usize), frame: u32) -> Self {
        // the pixel.dot is a fancy way of taking the (i,j) point and converting it to the index
        // jenkinsHash is probably unnecessary
        let seed = pixel.dot(UVec2::new(1, resolution.0 as u32)) ^ Self::jenkinsHash(frame);
        Self { state: Self::jenkinsHash(seed) }
    }


    fn rngNextInUnitHemisphere(&mut self) -> Vec3 {
        let r1 = self.rngNextFloat();
        let r2 = self.rngNextFloat();

        let phi = 2.0 * PI * r1;
        let sinTheta = (1.0 - r2 * r2).sqrt();

        let x = phi.cos() * sinTheta;
        let y = phi.sin() * sinTheta;
        let z = r2;

        Vec3::new(x, y, z)
    }

    fn rngNextVec3InUnitDisk(&mut self) -> Vec3 {
        // r^2 is distributed as U(0, 1).
        let r = self.rngNextFloat().sqrt();
        let alpha = 2.0 * PI * self.rngNextFloat();

        let x = r * alpha.cos();
        let y = r * alpha.sin();

        Vec3::new(x, y, 0.0)
    }

    pub fn rngNextVec3InUnitSphere(&mut self) -> Vec3 {
        // probability density is uniformly distributed over r^3
        let r = self.rngNextFloat().powf(0.33333f32);
        let cos_theta = 2.0 * self.rngNextFloat() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * self.rngNextFloat();

        let x = r * sin_theta * phi.cos();
        let y = r * sin_theta * phi.sin();
        let z = r * cos_theta;

        Vec3::new(x, y, z)
    }

    pub fn rngNextUintInRange(&mut self, min: u32, max: u32) -> u32 {
        self.rngNextInt();
        return min + (self.state) % (max - min);
    }

    pub fn rngNextFloat(&mut self) -> f32 {
        self.rngNextInt();
        return self.state as f32 * 2.3283064365387e-10;
    }

    pub fn rngNextInt(&mut self) {
        // PCG hash RXS-M-XS
        let oldState = (self.state.wrapping_mul(747796405)).wrapping_add(2891336453); // LCG
        let word = ((oldState >> ((oldState >> 28) + 4)) ^ oldState).wrapping_mul(277803737); // RXS-M
        self.state = (word >> 22) ^ word; // XS
    }

    fn jenkinsHash(input: u32) -> u32 {
        let mut x = input;

        x = x.wrapping_add(x.wrapping_shl(10));
        x ^= x >> 6;
        x = x.wrapping_add(x.wrapping_shl(3));
        x ^= x >> 11;
        x = x.wrapping_add(x.wrapping_shl(15));

        x
    }
}
