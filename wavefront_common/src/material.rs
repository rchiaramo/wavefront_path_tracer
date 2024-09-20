use glam::{Vec3, Vec4};

// material_type will be indexed as follows:
// 0 Lambertian; 1 Metal; 2 Dielectric

pub enum MaterialType {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Material {
    albedo: Vec4,
    fuzz: f32,
    refract_index: f32,
    material_type: u32,
    _buffer: u32,
}

unsafe impl bytemuck::Pod for Material {}
unsafe impl bytemuck::Zeroable for Material {}

impl Material {
    pub fn Lambertian(albedo: Vec3) -> Self {
        Self { albedo: albedo.extend(1.0), fuzz:0.0, refract_index:0.0, material_type: 0, _buffer: 0 }
    }

    pub fn Metal(albedo: Vec3, fuzz: f32) -> Self {
        Self { albedo: albedo.extend(1.0), fuzz: fuzz.clamp(0.0, 1.0), refract_index:0.0, material_type: 1, _buffer: 0 }
    }

    pub fn Dielectric(refract_index: f32) -> Self {
        Self { albedo: Vec4::ONE, fuzz:0.0, refract_index, material_type: 2, _buffer: 0 }
    }

    pub fn albedo(&self) -> Vec4 {
        self.albedo
    }

    pub fn fuzz(&self) -> f32 {
        self.fuzz
    }

    pub fn refract_index(&self) -> f32 {
        self.refract_index
    }

    pub fn material_type(&self) -> u32 {
        self.material_type
    }
}