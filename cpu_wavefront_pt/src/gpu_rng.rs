use glam::{UVec2, Vec3};
use wavefront_common::constants::{PI};

#[derive(Default, Copy, Clone, Debug)]
pub struct GpuRng {
    state: u32,
}

impl GpuRng {
    pub fn init_rng(pixel: UVec2, resolution: (usize, usize), frame: u32) -> Self {
        // the pixel.dot is a fancy way of taking the (i,j) point and converting it to the index
        // jenkins_hash is probably unnecessary
        let seed = pixel.dot(UVec2::new(1, resolution.0 as u32)) ^ Self::jenkins_hash(frame);
        Self { state: Self::jenkins_hash(seed) }
    }


    pub fn rng_next_in_unit_hemisphere(&mut self) -> Vec3 {
        let r1 = self.rng_next_float();
        let r2 = self.rng_next_float();

        let phi = 2.0 * PI * r1;
        let sin_theta = (1.0 - r2 * r2).sqrt();

        let x = phi.cos() * sin_theta;
        let y = phi.sin() * sin_theta;
        let z = r2;

        Vec3::new(x, y, z)
    }

    pub fn rng_next_vec3_unit_disk(&mut self) -> Vec3 {
        // r^2 is distributed as U(0, 1).
        let r = self.rng_next_float().sqrt();
        let alpha = 2.0 * PI * self.rng_next_float();

        let x = r * alpha.cos();
        let y = r * alpha.sin();

        Vec3::new(x, y, 0.0)
    }

    pub fn rng_next_vec3_unit_sphere(&mut self) -> Vec3 {
        // probability density is uniformly distributed over r^3
        let r = self.rng_next_float().powf(0.33333f32);
        let cos_theta = 2.0 * self.rng_next_float() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * self.rng_next_float();

        let x = r * sin_theta * phi.cos();
        let y = r * sin_theta * phi.sin();
        let z = r * cos_theta;

        Vec3::new(x, y, z)
    }

    pub fn rng_next_uint_in_range(&mut self, min: u32, max: u32) -> u32 {
        self.rng_next_int();
        min + (self.state) % (max - min)
    }

    pub fn rng_next_float(&mut self) -> f32 {
        self.rng_next_int();
        self.state as f32 * 2.3283064365387e-10
    }

    pub fn rng_next_int(&mut self) {
        // PCG hash RXS-M-XS
        let old_state = (self.state.wrapping_mul(747796405)).wrapping_add(2891336453); // LCG
        let word = ((old_state >> ((old_state >> 28) + 4)) ^ old_state).wrapping_mul(277803737); // RXS-M
        self.state = (word >> 22) ^ word; // XS
    }

    fn jenkins_hash(input: u32) -> u32 {
        let mut x = input;

        x = x.wrapping_add(x.wrapping_shl(10));
        x ^= x >> 6;
        x = x.wrapping_add(x.wrapping_shl(3));
        x ^= x >> 11;
        x = x.wrapping_add(x.wrapping_shl(15));

        x
    }
}