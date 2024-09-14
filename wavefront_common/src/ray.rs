use bytemuck::Zeroable;
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub inverse_direction: Vec3,
}

unsafe impl bytemuck::Zeroable for Ray {}
unsafe impl bytemuck::Pod for Ray {}

impl Default for Ray {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            direction: Vec3::ZERO,
            inverse_direction: Vec3::ZERO,
        }
    }
}