use glam::{Mat4, Vec2, Vec4, Vec4Swizzles};
use wavefront_common::camera_controller::GPUCamera;
use crate::gpu_rng::GpuRng;
use crate::ray::Ray;

pub struct RayGenShaderGPUBuffers {
    rng_state: GpuRng,
    image_size: (u32, u32),
    inv_proj_matrix: Mat4,
    view_matrix: Mat4,
    camera: GPUCamera,
}

impl RayGenShaderGPUBuffers {
    pub fn new(rng_state: GpuRng,
               image_size: (u32, u32),
               inv_proj_matrix: Mat4,
               view_matrix: Mat4,
               camera: GPUCamera) -> RayGenShaderGPUBuffers {
        Self { rng_state, image_size, inv_proj_matrix, view_matrix, camera }
    }
}

pub struct RayGenShader {
    gpu_buffers: Option<RayGenShaderGPUBuffers>,
}

impl RayGenShader {
    pub fn new() -> RayGenShader {
        RayGenShader {
            gpu_buffers: None,
        }
    }

    pub fn prepare_ray_gen_shader(&mut self,
                                  ray_gen_gpu_buffers: RayGenShaderGPUBuffers) -> () {
        self.gpu_buffers = Some(ray_gen_gpu_buffers);
    }

    pub fn run(&mut self) -> Vec::<Ray> {
        let mut rng_state = &mut self.gpu_buffers.as_mut().unwrap().rng_state;
        let (width, height) = self.gpu_buffers.as_ref().unwrap().image_size;
        let inv_proj_matrix = self.gpu_buffers.as_ref().unwrap().inv_proj_matrix;
        let view_matrix = self.gpu_buffers.as_ref().unwrap().view_matrix;
        let camera = self.gpu_buffers.as_ref().unwrap().camera;

        let mut offset = rng_state.rng_next_vec3_unit_disk();
        let mut ray_queue = Vec::<Ray>::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let mut point = Vec2::new((x as f32 + offset.x) / width as f32,
                                          1.0 - (y as f32 + offset.y) / height as f32);
                point = 2.0 * point - 1.0;
                let mut proj_point = inv_proj_matrix * Vec4::new(point.x, point.y, 1.0, 1.0);
                proj_point = proj_point / proj_point.w;
                proj_point = proj_point.xyz().extend(0.0);

                let mut origin = camera.position().xyz();

                if camera.defocus_radius() > 0.0 {
                    offset = rng_state.rngNextVec3InUnitDisk();

                    let p_lens = (camera.defocus_radius() * offset).extend(1.0);
                    let mut lens_origin = view_matrix * p_lens;
                    lens_origin = lens_origin / lens_origin.w;
                    origin = lens_origin.xyz();

                    let tf = camera.focus_distance() / proj_point.z;
                    proj_point = tf * proj_point - p_lens;
                }

                let ray_dir = view_matrix * proj_point.with_w(0.0);
                let direction = ray_dir.xyz().normalize();
                ray_queue.push(Ray { origin, direction });
            }
        }
        ray_queue
    }
}