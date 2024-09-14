use crate::camera_controller::CameraController;
use crate::gpu_structs::GPUFrameBuffer;

#[derive(Copy, Clone, PartialEq)]
pub struct SamplingParameters {
    pub samples_per_frame: u32,
    pub num_bounces: u32,
    pub clear_image_buffer: u32,
    pub samples_per_pixel: u32
}

impl SamplingParameters {
    pub fn new(samples_per_frame: u32, num_bounces: u32, clear_image_buffer: u32, samples_per_pixel: u32) -> Self {
        Self {
            samples_per_frame,
            num_bounces,
            clear_image_buffer,
            samples_per_pixel
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub struct RenderParameters {
    camera_controller: CameraController,
    pub sampling_parameters: SamplingParameters,
    viewport_size: (u32, u32)
}

impl RenderParameters {
    pub fn new(camera_controller: CameraController, sampling_parameters: SamplingParameters, viewport_size: (u32, u32)) -> Self {
        Self {
            camera_controller,
            sampling_parameters,
            viewport_size,
        }
    }

    pub fn set_viewport(&mut self, size: (u32, u32)) {
        self.viewport_size = size;
    }

    pub fn get_viewport(&self) -> (u32, u32) {
        self.viewport_size
    }

    pub fn camera_controller(&self) -> &CameraController {
        & self.camera_controller
    }

    pub fn sampling_parameters(&self) -> &SamplingParameters { &self.sampling_parameters }

    pub fn update_camera_controller(&mut self, camera_controller: CameraController) {
        self.camera_controller = camera_controller
    }

}

pub struct RenderProgress {
    frame: u32,
    samples_per_frame: u32,
    samples_per_pixel: u32,
    num_bounces: u32,
    accumulated_samples: u32,
}

impl RenderProgress {
    pub fn new(spf: u32, spp: u32, nb: u32) -> Self {
        Self {
            frame: 0,
            samples_per_frame: spf,
            samples_per_pixel: spp,
            num_bounces: nb,
            accumulated_samples: 0
        }
    }

    pub fn reset(&mut self) {
        self.accumulated_samples = 0;
    }

    pub fn get(&self, width: u32, height: u32) -> GPUFrameBuffer {
        GPUFrameBuffer::new(width, height, self.frame, self.accumulated_samples)
    }

    pub fn current_frame(&self) -> u32 {
        self.frame
    }

    pub fn get_next_frame(&mut self, rp: &mut RenderParameters) -> GPUFrameBuffer {
        // if accumulated samples is 0, there's been something that triggered a reset
        let current_progress = self.accumulated_samples;
        let delta_samples = rp.sampling_parameters.samples_per_frame;

        // update the samples per frame if user changed, but never if spf = 0
        if delta_samples != 0 && delta_samples != self.samples_per_frame {
            self.samples_per_frame = delta_samples;
        }

        // update the samples per pixel if user changed them
        if rp.sampling_parameters.samples_per_pixel != self.samples_per_pixel {
            self.samples_per_pixel = rp.sampling_parameters.samples_per_pixel;
        }

        // update the number of bounces per ray if user changed them
        if rp.sampling_parameters.num_bounces != self.num_bounces {
            self.num_bounces = rp.sampling_parameters.num_bounces;
        }

        let updated_progress = current_progress + delta_samples;
        let (width, height) = rp.get_viewport();
        let mut frame = 0;
        let mut accumulated_samples = 0;

        if self.accumulated_samples == 0 {
            rp.sampling_parameters = SamplingParameters::new(
                self.samples_per_frame,
                self.num_bounces,
                1,
                self.samples_per_pixel
            );
            frame = 1;
            self.frame = 1;
            accumulated_samples = delta_samples;
            self.accumulated_samples = accumulated_samples;
        } else if updated_progress > self.samples_per_pixel {
            rp.sampling_parameters = SamplingParameters::new(
              0,
              self.num_bounces,
              0,
              self.samples_per_pixel
            );
            self.frame += 1;
            frame = self.frame;
            accumulated_samples = current_progress;
        } else {
            rp.sampling_parameters = SamplingParameters::new(
                // here we need to use the sampling parameters spf as at the end of the render
                // it is zero; we want it to stay that way
                rp.sampling_parameters.samples_per_frame,
                self.num_bounces,
                0,
                self.samples_per_pixel
            );
            self.frame += 1;
            frame = self.frame;
            self.accumulated_samples = updated_progress;
            accumulated_samples = updated_progress;
        }

        GPUFrameBuffer::new(width, height, frame, accumulated_samples)
    }

    pub fn progress(&self) -> f32 {
        self.accumulated_samples as f32 / self.samples_per_pixel as f32
    }
}



