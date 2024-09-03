use glam::{Vec4};
use crate::parameters::SamplingParameters;
use crate::camera::Camera;
use crate::camera_controller::CameraController;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUSamplingParameters {
    samples_per_frame: u32,
    num_bounces: u32,
    clear_image_buffer: u32,
    _buffer: u32,
}

// sampling parameters are set at initialization but can also be changed by the user via GUI
impl GPUSamplingParameters {
    pub fn get_gpu_sampling_params(sampling_parameters: &SamplingParameters)
                                   -> GPUSamplingParameters
    {
        GPUSamplingParameters {
            samples_per_frame: sampling_parameters.samples_per_frame,
            num_bounces: sampling_parameters.num_bounces,
            clear_image_buffer: sampling_parameters.clear_image_buffer,
            _buffer: 0u32
        }
    }
    pub fn spf(&self) -> u32 { self. samples_per_frame}
    pub fn num_bounces(&self) -> u32 { self.num_bounces }
    pub fn clear_image(&self) -> u32 { self.clear_image_buffer }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUFrameBuffer {
    width: u32,
    height: u32,
    frame: u32,
    accumulated_samples: u32
}

impl GPUFrameBuffer {
    pub fn new(width: u32, height: u32, frame: u32, accumulated_samples: u32) -> Self {
        Self {
            width,
            height,
            frame,
            accumulated_samples
        }
    }
    pub fn into_array(&self) -> [u32; 4] {
        [self.width, self.height, self.frame, self.accumulated_samples]
    }
}