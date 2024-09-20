use crate::path_tracer::PathTracer;
use crate::query_gpu::{Queries, QueryResults};

use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};
use wavefront_common::frames_per_second::FramesPerSecond;
use wavefront_common::parameters::{RenderParameters, SamplingParameters};
use wavefront_common::scene::Scene;
use wavefront_common::gui::GUI;
use wavefront_common::wgpu_state::WgpuState;


pub struct App {
    window: Option<Arc<Window>>,
    path_tracer: Option<PathTracer>,
    gui: Option<GUI>,
    scene: Scene,
    render_parameters: RenderParameters,
    sampling_parameters: SamplingParameters,
    last_render_time: Instant,
    frames_per_second: FramesPerSecond,
}

impl App {
    pub fn new(scene: Scene, render_parameters: RenderParameters, sampling_parameters: SamplingParameters) -> Self {
        Self {
            window: None,
            path_tracer: None,
            gui: None,
            scene,
            render_parameters,
            sampling_parameters,
            last_render_time: Instant::now(),
            frames_per_second: FramesPerSecond::new()
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let size = self.render_parameters.viewport_size();
        if self.window.is_none() {
            let win_attr = Window::default_attributes()
                .with_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1))
                .with_title("GPU path tracer app");
            let window = Arc::new(
                event_loop.create_window(win_attr).unwrap());
            self.window = Some(window.clone());

            let max_viewport_resolution = window
                .available_monitors()
                .map(|monitor| -> u32 {
                    let viewport = monitor.size();
                    let size = (viewport.width, viewport.height);
                    size.0 * size.1
                })
                .max()
                .expect("must have at least one monitor");

            let path_tracer = PathTracer::new(
                window.clone(),
                max_viewport_resolution,
                &mut self.scene,
                &self.render_parameters,
                &self.sampling_parameters
            );

            // let wgpu_state  = path_tracer.wgpu_state();
            // self.gui = GUI::new(&window, wgpu_state);
            self.path_tracer = Some(path_tracer);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, 
                    window_id: WindowId, event: WindowEvent) {
        let window = self.window.as_ref().unwrap();
        if window.id() != window_id { return; }

        let path_tracer = self.path_tracer.as_mut().unwrap();
        // let gui = self.gui.as_mut().unwrap();
        let mut rp = path_tracer.get_render_parameters();

        if !path_tracer.input(&event) {
            match event {
                WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                    ..
                } => {
                    event_loop.exit();
                }

                WindowEvent::Resized(new_size) => {
                    let (width, height) = (new_size.width, new_size.height);
                    rp.set_viewport((width, height));
                    path_tracer.resize(rp);
                }

                WindowEvent::RedrawRequested => {
                    // let now = Instant::now();
                    // let dt = now - self.last_render_time;
                    // self.last_render_time = now;
                    // self.frames_per_second.update(dt);
                    // let avg_fps= self.frames_per_second.get_avg_fps();
                    // let kernel_time= self.query_results.get_running_avg();

                    // gui.display_ui(window.as_ref(), path_tracer.progress(), & mut rp, avg_fps, kernel_time, dt);
                    path_tracer.update_render_parameters(rp);
                    path_tracer.run();
                }

                _ => {}
            }
        }
        // gui.platform.handle_event(gui.imgui.io_mut(), &window, window_id, &event);
        window.request_redraw();
    }
}