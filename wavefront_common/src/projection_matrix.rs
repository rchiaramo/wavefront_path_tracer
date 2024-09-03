pub struct ProjectionMatrix {
    vfov_rad: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32
}

impl ProjectionMatrix {
    pub fn new(vfov_rad: f32, aspect_ratio: f32,
               z_near: f32, z_far: f32) -> Self {

        Self {
            vfov_rad,
            aspect_ratio,
            z_near,
            z_far
        }

    }

    pub fn p_inv(&self) -> [[f32; 4]; 4] {
        // compared to how I labeled h and w in prior iterations, I now
        // use them as would be commonly understood - h is the height of the
        // viewport and w is the width calculated using aspect ratio
        // the p_inv matrix has to multiply by these values to scale up
        let h = (self.vfov_rad / 2.0).tan();
        let w = h * self.aspect_ratio;
        let r = self.z_far / (self.z_far - self.z_near);

        let p_inv = [
            [w, 0.0, 0.0, 0.0],
            [0.0, h, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0 / (r * self.z_near)],
            [0.0, 0.0, 1.0, 1.0 / self.z_near]
        ];

        p_inv
    }
}