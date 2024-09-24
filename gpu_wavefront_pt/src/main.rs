mod app;
mod path_tracer;
mod query_gpu;
mod generate_ray;
mod shade;
mod display;
mod miss;
mod extend;
mod accumulate;
mod kernel;

use glam::Vec3;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};
use wavefront_common::camera::Camera;
use wavefront_common::camera_controller::CameraController;
use wavefront_common::parameters::RenderParameters;
use wavefront_common::scene::Scene;

use crate::app::App;

fn main() -> Result<(), EventLoopError> {
    env_logger::init();

    let scene = Scene::book_one_final();
    // let camera = Camera::new(Vec3::new(0.0, 0.0, 1.0),
    //                          Vec3::new(0.0, 0.0, -1.0));
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
    let screen_size = (2880, 1620); // (1920, 1080) (3840, 2160)

    let render_parameters
        = RenderParameters::new(camera_controller, screen_size);
    
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(scene, render_parameters);
    event_loop.run_app(&mut app)
}