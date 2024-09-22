use crate::camera_controller::CameraController;
use crate::gpu_structs::GPUFrameBuffer;

pub const SPP: u32 = 15;
pub const SPF: u32 = 1;

#[derive(Copy, Clone, PartialEq)]
pub struct RenderParameters {
    camera_controller: CameraController,
    viewport_size: (u32, u32),
    resized: bool,
    camera_changed: bool,
}

impl RenderParameters {
    pub fn new(camera_controller: CameraController, viewport_size: (u32, u32)) -> Self {
        Self {
            camera_controller,
            viewport_size,
            resized: false,
            camera_changed: false,
        }
    }

    pub fn changed (&self) -> bool {
        self.resized || self.camera_changed
    }

    pub fn resized (&self) -> bool {
        self.resized
    }

    pub fn camera_changed(&self) -> bool {
        self.camera_changed
    }

    pub fn set_viewport(&mut self, size: (u32, u32)) {
        self.viewport_size = size;
        self.resized = true;
    }

    pub fn viewport_size(&self) -> (u32, u32) {
        self.viewport_size
    }

    pub fn reset(&mut self) {
        self.resized = false;
        self.camera_changed = false;
    }

    pub fn camera_controller(&self) -> &CameraController {
        & self.camera_controller
    }

    pub fn update_camera_controller(&mut self, camera_controller: CameraController) {
        self.camera_controller = camera_controller;
        self.camera_changed = true;
    }
}

pub struct RenderProgress {
    frame: u32,
    accumulated_samples: u32,
}

impl RenderProgress {
    pub fn new() -> Self {
        Self {
            frame: 0,
            accumulated_samples: 0
        }
    }

    pub fn get(&self, width: u32, height: u32) -> GPUFrameBuffer {
        GPUFrameBuffer::new(width, height, self.frame)
    }

    pub fn get_next_frame(&mut self, rp: &mut RenderParameters) -> GPUFrameBuffer {
        let (width, height) = rp.viewport_size();
        self.frame += 1;

        GPUFrameBuffer::new(width, height, self.frame)
    }

    pub fn incr_accumulated_samples(&mut self, delta: u32) {
        self.accumulated_samples += delta;
    }

    pub fn reset(&mut self) {
        self.accumulated_samples = 0;
        self.frame = 0;
    }

    pub fn progress(&self) -> f32 {
        self.accumulated_samples as f32 / SPP as f32
    }

    pub fn accumulated_samples(&self) -> u32 {
        self.accumulated_samples
    }
}



