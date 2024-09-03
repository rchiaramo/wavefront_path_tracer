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
use wavefront_common::parameters::RenderParameters;
use wavefront_common::scene::Scene;
use wavefront_common::gui::GUI;


pub struct App<'a> {
    window: Option<Arc<Window>>,
    wgpu_state: Option<WgpuState<'a>>,
    path_tracer: Option<PathTracer>,
    gui: Option<GUI>,
    query_results: QueryResults,
    cursor_position: winit::dpi::PhysicalPosition<f64>,
    scene: Scene,
    render_parameters: RenderParameters,
    last_render_time: Instant,
    frames_per_second: FramesPerSecond,
}

impl<'a> App<'a> {
    pub fn new(scene: Scene, render_parameters: RenderParameters) -> Self {
        Self {
            window: None,
            wgpu_state: None,
            path_tracer: None,
            gui: None,
            query_results: Default::default(),
            cursor_position: Default::default(),
            scene,
            render_parameters,
            last_render_time: Instant::now(),
            frames_per_second: FramesPerSecond::new()
        }
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let size = self.render_parameters.get_viewport();
        if self.window.is_none() {
            let win_attr = Window::default_attributes()
                .with_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1))
                .with_title("GPU path tracer app");
            let window = Arc::new(
                event_loop.create_window(win_attr).unwrap());
            self.window = Some(window.clone());

            self.wgpu_state = WgpuState::new(window.clone());

            let max_viewport_resolution = window
                .available_monitors()
                .map(|monitor| -> u32 {
                    let viewport = monitor.size();
                    let size = (viewport.width, viewport.height);
                    size.0 * size.1
                })
                .max()
                .expect("must have at least one monitor");

            if let Some(state) = &self.wgpu_state {
                self.path_tracer =
                    PathTracer::new(&state.device,
                                    max_viewport_resolution,
                                    &mut self.scene,
                                    &self.render_parameters);
                self.gui = GUI::new(&window, &state.surface_config, &state.device, &state.queue);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, 
                    window_id: WindowId, event: WindowEvent) {
        let window = self.window.as_ref().unwrap();
        if window.id() != window_id { return; }

        let path_tracer = self.path_tracer.as_mut().unwrap();
        let state = self.wgpu_state.as_mut().unwrap();
        let gui = self.gui.as_mut().unwrap();
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
                    state.resize((width, height));
                    path_tracer.update_render_parameters(rp);
                }
                
                WindowEvent::CursorMoved { position, ..} => {
                    self.cursor_position = position;
                }

                // state below is NOT wgpu state as declared above
                WindowEvent::MouseInput { state, ..
                } => {
                    if state.is_pressed() {
                        println!("cursor position {:?}", self.cursor_position);
                    }
                }

                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now - self.last_render_time;
                    self.last_render_time = now;
                    self.frames_per_second.update(dt);
                    let avg_fps= self.frames_per_second.get_avg_fps();
                    let kernel_time= self.query_results.get_running_avg();
                    gui.display_ui(window.as_ref(), path_tracer.progress(), & mut rp, avg_fps, kernel_time, dt);

                    path_tracer.update_render_parameters(rp);
                    path_tracer.update_buffers(&state.queue);
                    let mut queries = Queries::new(&state.device, QueryResults::NUM_QUERIES);
                    path_tracer.run_compute_kernel(&state.device, &state.queue, &mut queries);
                    path_tracer.run_display_kernel(
                        &mut state.surface,
                        &state.device,
                        &state.queue,
                        gui
                    );
                    let raw_results = queries.wait_for_results(&state.device);
                    // println!("Raw timestamp buffer contents: {:?}", &raw_results);
                    self.query_results.process_raw_results(&state.queue, raw_results);
                }

                _ => {}
            }
        }
        gui.platform.handle_event(gui.imgui.io_mut(), &window, window_id, &event);
        window.request_redraw();
    }
}

pub struct WgpuState<'a> {
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<'a> WgpuState<'a> {
    pub fn new(window: Arc<Window>) -> Option<WgpuState<'a>> {
        pollster::block_on(WgpuState::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Option<WgpuState<'a>> {
        let size = {
            let viewport = window.inner_size();
            (viewport.width, viewport.height)
        };

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            }
        );

        let surface = instance.create_surface(
            Arc::clone(&window)).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await?;

        // Check timestamp features.
        let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        // if features.contains(wgpu::Features::BGRA8UNORM_STORAGE)
        // {
        //     println!("Adapter has bgra8unorm storage");
        // } else {
        //     println!("Adapter does not have this storage");
        // }
        // if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        //     println!("Adapter supports timestamp queries.");
        // } else {
        //     println!("Adapter does not support timestamp queries, aborting.");
        // }
        // let timestamps_inside_passes = features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        // if timestamps_inside_passes {
        //     println!("Adapter supports timestamp queries within encoders.");
        // } else {
        //     println!("Adapter does not support timestamp queries within encoders.");
        // }

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: features, // wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 512_u32 << 20,
                    ..Default::default()
                },
                label: None,
                memory_hints: Default::default(),
            },
            None,
        ).await.unwrap();

        let surface_capabilities = surface.get_capabilities(&adapter);

        // I need to figure out why Bgra8Unorm looks best

        // let surface_format = surface_capabilities.formats.iter()
        //     .find(|f| f.is_srgb())
        //     .copied()
        //     .unwrap_or(surface_capabilities.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: size.0,
            height: size.1,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };

        Some(Self {
            surface,
            surface_config,
            device,
            queue,
        })
    }
    
    fn resize(&mut self, new_size: (u32, u32))
    {
        self.surface_config.width = new_size.0;
        self.surface_config.height = new_size.1;
        self.surface.configure(&self.device, &self.surface_config);
    }
}

