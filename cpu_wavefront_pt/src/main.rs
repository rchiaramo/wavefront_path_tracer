mod app;
mod path_tracer;
mod compute_shader;
mod ray_gen_shader;
mod gpu_rng;
mod wavefront_path_integrator;
mod ray;
mod gpu;

use std::sync::Arc;
use wavefront_common::bvh;
use wavefront_common::camera::Camera;
use wavefront_common::camera_controller::CameraController;
use wavefront_common::gpu_buffer;
use wavefront_common::gpu_structs;
use wavefront_common::gui;
use wavefront_common::material;
use wavefront_common::parameters;
use wavefront_common::parameters::{RenderParameters, SamplingParameters};
use wavefront_common::scene;
use wavefront_common::sphere;
use glam::Vec3;

use crate::app::App;
use wavefront_common::scene::Scene;
use winit::error::EventLoopError;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::Window;

fn main() -> Result<(), EventLoopError> {
    env_logger::init();

    let scene = Scene::book_one_final();
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 1.0),       //look from
        Vec3::new(0.0, 0.0, -1.0));     //look at
    let camera = Camera::book_one_final_camera();
    let camera_controller
        = CameraController::new(camera,
                                20.0,
                                0.6,
                                10.0,
                                0.1,
                                100.0,
                                4.0,
                                0.1);
    let screen_size = (960, 540); // (960, 540) (1920, 1080) (2880, 1620) (3840, 2160)
    let sampling_parameters = SamplingParameters::new(1,
                                                      50,
                                                      1,
                                                      100);
    let render_parameters = RenderParameters::new(camera_controller, sampling_parameters, screen_size);

    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(scene, render_parameters);
    event_loop.run_app(&mut app)
}
