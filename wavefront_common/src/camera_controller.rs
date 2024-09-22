use std::f32::consts::{FRAC_PI_2, PI};
use std::time::Duration;
use glam::{Vec3, Vec4};
use imgui::Key::P;
use crate::camera::Camera;

#[derive(Copy, Clone, PartialEq)]
pub struct CameraController {
    camera: Camera,
    vfov_rad: f32,
    defocus_angle_rad: f32,
    focus_distance: f32,
    z_near: f32,
    z_far: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_right: f32,
    amount_left: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    speed: f32,
    sensitivity: f32
}


impl CameraController {
    const SAFE_FRAC_PI:f32 = PI - 0.001;

    pub fn new(camera: Camera, vfov: f32, defocus_angle: f32, focus_distance: f32,
               z_near:f32, z_far: f32, speed: f32, sensitivity: f32) -> Self {
        Self {
            camera,
            vfov_rad: vfov.to_radians(),
            defocus_angle_rad: defocus_angle.to_radians(),
            focus_distance,
            z_near,
            z_far,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_right: 0.0,
            amount_left: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            speed,
            sensitivity
        }
    }

    pub fn vfov_rad(&self) -> f32 {
        self.vfov_rad
    }
    pub fn set_vfov(&mut self, vfov:f32) { self.vfov_rad = vfov.to_radians() }

    pub fn dof(&self) -> (f32, f32) {
        (self.defocus_angle_rad, self.focus_distance)
    }
    pub fn set_defocus_angle(&mut self, da:f32) { self.defocus_angle_rad = da.to_radians() }
    pub fn set_focus_distance(&mut self, fd:f32) { self.focus_distance = fd }

    pub fn get_clip_planes(&self) -> (f32, f32) { (self.z_near, self.z_far) }

    pub fn get_GPU_camera(&self) -> GPUCamera {
        GPUCamera::new(&self.camera, self.defocus_angle_rad, self.focus_distance)
    }

    pub fn get_view_matrix(&self) -> [[f32;4];4] {
        self.camera.view_transform()
    }

    pub fn process_mouse(&mut self, delta: [f32; 2]) {
        self.rotate_horizontal = delta[0];
        self.rotate_vertical = delta[1];
    }

    pub fn move_up(&mut self, dir: u32) {
        if dir == 1 {
            self.amount_up = 1.0;
        } else {
            self.amount_up = 0.0;
        }
    }

    pub fn move_down(&mut self, dir: u32) {
        if dir == 1 {
            self.amount_down = 1.0;
        } else {
            self.amount_down = 0.0;
        }
    }

    pub fn move_forward(&mut self, dir: u32) {
        if dir == 1 {
            self.amount_forward = 1.0;
        } else {
            self.amount_forward = 0.0;
        }
    }

    pub fn move_backwards(&mut self, dir: u32) {
        if dir == 1 {
            self.amount_backward = 1.0;
        } else {
            self.amount_backward = 0.0;
        }
    }

    pub fn move_right(&mut self, dir: u32) {
        if dir == 1 {
            self.amount_right = 1.0;
        } else {
            self.amount_right = 0.0;
        }
    }

    pub fn move_left(&mut self, dir: u32) {
        if dir == 1 {
            self.amount_left = 1.0;
        } else {
            self.amount_left = 0.0;
        }
    }

    pub fn update_camera(&mut self, dt: f32) {
        // Move forward/backward and left/right
        let (sin_yaw, cos_yaw) = self.camera.yaw.sin_cos();

        let forward = Vec3::new(sin_yaw, 0.0, cos_yaw);
        let right = Vec3::new(-cos_yaw, 0.0, sin_yaw);

        self.camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        self.camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;


        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        self.camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        self.camera.yaw -= self.rotate_horizontal * self.sensitivity * dt;
        self.camera.pitch -= self.rotate_vertical * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non-cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if self.camera.pitch < -Self::SAFE_FRAC_PI {
            self.camera.pitch = -Self::SAFE_FRAC_PI;
        } else if self.camera.pitch > Self::SAFE_FRAC_PI {
            self.camera.pitch = Self::SAFE_FRAC_PI;
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GPUCamera {
    camera_position: Vec4,
    pitch: f32,
    yaw: f32,
    defocus_radius: f32,
    focus_distance: f32,
}
unsafe impl bytemuck::Pod for GPUCamera {}
unsafe impl bytemuck::Zeroable for GPUCamera {}

impl GPUCamera {
    pub fn new(camera: &Camera, defocus_angle_rad: f32, focus_distance: f32) -> GPUCamera {
        let defocus_radius = focus_distance * (0.5 * defocus_angle_rad).tan();
        let (camera_position, pitch, yaw) = camera.get_camera();

        GPUCamera {
            camera_position: camera_position.extend(1.0),
            pitch,
            yaw,
            defocus_radius,
            focus_distance,
        }
    }

    pub fn position(&self) -> Vec4 { self.camera_position }
    pub fn defocus_radius(&self) -> f32 { self.defocus_radius }
    pub fn focus_distance(&self) -> f32 { self.focus_distance }
}