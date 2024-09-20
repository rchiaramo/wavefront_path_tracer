use std::rc::Rc;
use std::time::{Duration, Instant};
use imgui::{FontSource, MouseCursor};
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use wgpu::{Queue, SurfaceConfiguration};
use winit::window::Window;
use crate::parameters::{RenderParameters, SamplingParameters};
use crate::wgpu_state::WgpuState;

pub struct GUI {
    wgpu_state: Rc<WgpuState>,
    pub platform: WinitPlatform,
    pub imgui: imgui::Context,
    pub imgui_renderer: Renderer,
    last_cursor: Option<MouseCursor>,
}

impl GUI {
    pub fn new(window: &Window, wgpu_state: Rc<WgpuState>)
        -> Option<Self> {

        let mut imgui = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );
        imgui.set_ini_filename(std::path::PathBuf::from("imgui.ini"));

        let hidpi_factor = window.scale_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);
        let texture_format = wgpu_state.surface_config.borrow().format;
        let renderer_config = RendererConfig {
            texture_format,
            ..Default::default()
        };
        let device = wgpu_state.device();
        let queue = wgpu_state.queue();
        let mut imgui_renderer
            = Renderer::new(&mut imgui, device, queue, renderer_config);

        Some(Self {
            wgpu_state,
            platform,
            imgui,
            imgui_renderer,
            last_cursor: None,
        })
    }

    pub fn display_ui(&mut self, window: &Window, progress: f32, rp: & mut RenderParameters,
                      avg_fps:f32, compute_kernel_time: f32, dt: Duration) {
        self.imgui.io_mut().update_delta_time(dt);

        let mut cc = rp.camera_controller().clone();
        let mut fov = cc.vfov_rad().to_degrees();
        let (defocus_angle_rad, mut focus_distance) = cc.dof();
        let mut defocus_angle = defocus_angle_rad.to_degrees();

        let mut samples_per_pixel = 0;
        let mut samples_per_frame = 0;
        let mut num_bounces = 0;

        {
            self.platform
                .prepare_frame(self.imgui.io_mut(), &window)
                .expect("WinitPlatform::prepare_frame failed");

            let ui = self.imgui.frame();

            // if the right mouse button is held down and we move the mouse, we can orient the camera
            let mouse_down = ui.io().mouse_down;
            if mouse_down[1] {
                let mouse_delta = ui.io().mouse_delta;
                cc.process_mouse(mouse_delta);
            }
            // move up/down
            if ui.is_key_pressed(imgui::Key::E) {
                cc.move_up(1);
            }
            if ui.is_key_released(imgui::Key::E) {
                cc.move_up(0);
            }
            if ui.is_key_pressed(imgui::Key::Q) {
                cc.move_down(1);
            }
            if ui.is_key_released(imgui::Key::Q) {
                cc.move_down(0);
            }
            // move forward/backwards
            if ui.is_key_pressed(imgui::Key::W) {
                cc.move_forward(1);
            }
            if ui.is_key_released(imgui::Key::W) {
                cc.move_forward(0);
            }
            if ui.is_key_pressed(imgui::Key::S) {
                cc.move_backwards(1);
            }
            if ui.is_key_released(imgui::Key::S) {
                cc.move_backwards(0);
            }
            // move left/right
            if ui.is_key_pressed(imgui::Key::D) {
                cc.move_right(1);
            }
            if ui.is_key_released(imgui::Key::D) {
                cc.move_right(0);
            }
            if ui.is_key_pressed(imgui::Key::A) {
                cc.move_left(1);
            }
            if ui.is_key_released(imgui::Key::A) {
                cc.move_left(0);
            }

            cc.update_camera(dt.as_secs_f32());

            {
                let window = ui.window("Parameters");
                window
                    .size([300.0, 300.0], imgui::Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!(
                            "Render progress: {:.1} %  Avg FPS: {:.2}  Avg compute kernel time: {:.2}ns ",
                            progress * 100.0, avg_fps, compute_kernel_time
                        ));

                        ui.separator();

                        ui.text("Camera parameters");
                        ui.slider(
                            "vfov",
                            10.0,
                            90.0,
                            &mut fov,
                        );

                        ui.slider(
                            "defocus radius",
                            0.0,
                            1.0,
                            &mut defocus_angle,
                        );

                        ui.slider(
                            "focus distance",
                            5.0,
                            20.0,
                            &mut focus_distance,
                        );

                        ui.separator();
                        ui.text("Sampling parameters");
                        ui.slider(
                            "Samples per frame",
                            1,
                            10,
                            &mut samples_per_frame,
                        );

                        ui.slider(
                            "Samples per pixel",
                            10,
                            1000,
                            &mut samples_per_pixel,
                        );

                        ui.slider(
                            "num bounces",
                            5,
                            100,
                            &mut num_bounces,
                        );
                    });
            }

            if self.last_cursor != ui.mouse_cursor() {
                self.last_cursor = ui.mouse_cursor();
                self.platform.prepare_render(&ui, &window);
            }
            cc.set_vfov(fov);
            cc.set_defocus_angle(defocus_angle);
            cc.set_focus_distance(focus_distance);
            rp.update_camera_controller(cc);
        }
    }
}
