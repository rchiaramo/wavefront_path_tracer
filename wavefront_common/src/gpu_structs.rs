use glam::{Vec4};
use crate::camera::Camera;
use crate::camera_controller::CameraController;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUFrameBuffer {
    width: u32,
    height: u32,
    frame: u32,
    sample_number: u32
}

impl GPUFrameBuffer {
    pub fn new(width: u32, height: u32, frame: u32) -> Self {
        Self {
            width,
            height,
            frame,
            sample_number: 0
        }
    }
    pub fn into_array(&self) -> [u32; 4] {
        [self.width, self.height, self.frame, self.sample_number]
    }

    pub fn set_sample_number(&mut self, sample_number: u32) {
        self.sample_number = sample_number;
    }
}