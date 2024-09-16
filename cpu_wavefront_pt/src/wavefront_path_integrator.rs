use std::sync::Arc;
use glam::{Mat4, UVec2, Vec3};
use rayon::iter::IntoParallelIterator;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BufferAddress, BufferUsages, Device, Queue, RenderPipeline, ShaderStages, Surface, TextureFormat};
use winit::event::WindowEvent;
use winit::window::Window;
use wavefront_common::bvh::BVHTree;
use wavefront_common::camera::Camera;
use wavefront_common::camera_controller::GPUCamera;
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::gpu_structs::{GPUFrameBuffer, GPUSamplingParameters};
use wavefront_common::gui::GUI;
use wavefront_common::parameters::{RenderParameters, RenderProgress};
use wavefront_common::projection_matrix::ProjectionMatrix;
use wavefront_common::scene::Scene;
use wavefront_common::wgpu_state::WgpuState;
use crate::compute_shader::ComputeShader;
use crate::gpu::GPU;
use crate::gpu_rng::GpuRng;
use crate::ray::Ray;
use crate::ray_gen_shader::{RayGenShader, RayGenShaderGPUBuffers};

pub struct WavefrontPathIntegrator<'a> {
    wgpu_state: WgpuState<'a>,
    gpu_state: GPU,
    compute_shader: ComputeShader,
    ray_gen_shader: RayGenShader,
    camera: GPUCamera,
    max_depth: u32,
    samples_per_pixel: u32,
    samples_per_frame: u32,
    scanlines_per_pass: u32,
    image_buffer: GPUBuffer,
    frame_buffer: GPUBuffer,
    camera_buffer: GPUCamera,
    sampling_parameters_buffer: GPUSamplingParameters,
    projection_buffer: [[f32;4];4],
    view_buffer: [[f32;4];4],
    display_bind_group: BindGroup,
    display_pipeline: RenderPipeline,
    render_parameters: RenderParameters,
    last_render_parameters: RenderParameters,
    render_progress: RenderProgress,
}

impl<'a> WavefrontPathIntegrator<'a> {
    pub fn new(window: Arc<Window>, max_window_size: u32,
               scene: &mut Scene,
               rp: &RenderParameters)
               -> Self {
        let wgpu_state = WgpuState::new(window);
        let device = wgpu_state.device();
        let window_size = window.inner_size();

        // create the image_buffer that the compute shader will use to store image
        // we make this array as big as the largest possible window on resize
        let image = vec![[0.0f32; 3]; max_window_size as usize];
        let image_buffer =
            GPUStorageBuffer::new_from_bytes(device,
                                             BufferUsages::STORAGE,
                                             0u32,
                                             bytemuck::cast_slice(image.as_slice()),
                                             Some("image buffer"));

        let gpu_state = GPU::new(max_window_size as usize);

        // create the frame_buffer
        let frame_buffer = GPUStorageBuffer::new(device,
                                                 BufferUsages::UNIFORM,
                                                 16 as BufferAddress,
                                                 1u32,
                                                 Some("frame buffer"));

        // group image and frame buffers into image bind group
        // for the display shader
        let display_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label: Some("display bind group layout"),
                entries: &[
                    image_buffer.layout(ShaderStages::FRAGMENT, true),
                    frame_buffer.layout(ShaderStages::FRAGMENT, true)
                ],
            }
        );

        let display_bind_group = device.create_bind_group(
            &BindGroupDescriptor {
                label: Some("display bind group"),
                layout: &display_bind_group_layout,
                entries: &[
                    image_buffer.binding(),
                    frame_buffer.binding()
                ],
            }
        );

        // create the bvh_tree that corresponds to the scene
        let mut bvh_tree = BVHTree::new(scene.spheres.len());
        bvh_tree.build_bvh_tree(&mut scene.spheres);

        let spheres_buffer = scene.spheres.clone();
        let materials_buffer = scene.materials.clone();
        let bvh_buffer = bvh_tree.nodes;

        // create the parameters bind group to interact with GPU during runtime
        // this will include the camera, and the sampling parameters
        let render_parameters = rp.clone();
        let camera_controller = render_parameters.camera_controller();
        let (width, height) = render_parameters.viewport_size();
        let ar = width as f32 / height as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let projection_buffer = ProjectionMatrix::new(
            camera_controller.vfov_rad(), ar, z_near, z_far).p_inv();

        let view_buffer = camera_controller.get_view_matrix();

        let gpu_sampling_parameters_buffer
            = GPUSamplingParameters::get_gpu_sampling_params(render_parameters.sampling_parameters());

        let camera_buffer = camera_controller.get_GPU_camera();

        let last_render_parameters = render_parameters.clone();

        let spf = render_parameters.sampling_parameters().samples_per_frame;
        let spp = render_parameters.sampling_parameters().samples_per_pixel;
        let nb = render_parameters.sampling_parameters().num_bounces;
        let render_progress = RenderProgress::new(spf, spp, nb);

        let ray_gen_shader = RayGenShader::new();

        let compute_shader = ComputeShader::new(spheres_buffer,
                                                materials_buffer,
                                                bvh_buffer,
                                                camera_buffer,
                                                gpu_sampling_parameters_buffer,
                                                GPUFrameBuffer::new(width,
                                                                    height,
                                                                    1,
                                                                    0),
                                                max_window_size);

        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../../wavefront_common/shaders/display_shader.wgsl")
        );

        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("display pipeline layout"),
                bind_group_layouts: &[&display_bind_group_layout],
                push_constant_ranges: &[],
            });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display pipeline"),
            layout: Some(&display_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TextureFormat::Bgra8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            wgpu_state,
            image_buffer,
            frame_buffer,
            camera_buffer,
            sampling_parameters_buffer: gpu_sampling_parameters_buffer,
            projection_buffer,
            view_buffer,
            display_bind_group,
            display_pipeline,
            render_parameters,
            last_render_parameters,
            render_progress,
            compute_shader,
            ray_gen_shader,
            camera: camera_buffer,
            max_depth: 0,
            samples_per_pixel: spp,
            samples_per_frame: 0,
            scanlines_per_pass: 1,
            gpu_state,
        }
    }

    pub fn wgpu_state(&self) -> &WgpuState {
        &self.wgpu_state
    }

    pub fn render(&mut self) {
        let (width, height) = self.render_parameters.viewport_size();
        let mut wavefront_depth = 0;
        // fundamental rendering loop
        for sample_index in 0..self.samples_per_frame {
            for y in (0..width * height).step_by(self.scanlines_per_pass as usize) {
                // clear the ray queue
                let rng_state =
                    GpuRng::init_rng(UVec2::new(0, y),
                                     (width as usize, height as usize),
                                     sample_index);
                self.prepare_ray_gen_shader(RayGenShaderGPUBuffers::new(
                    rng_state.clone(),(width, self.scanlines_per_pass),
                    Mat4::from_cols_array_2d(&self.projection_buffer),
                    Mat4::from_cols_array_2d(&self.view_buffer),
                    self.camera_buffer)
                );

                self.gpu_state.ray_buffer_mut() = self.ray_gen_shader.run();

                while wavefront_depth < self.max_depth {
                    // get next ray queue
                    // reset all the relevant queues
                    // generate ray samples

                    let image = self.compute_shader.run(self.gpu_state.ray_buffer());
                    self.image_buffer.queue_for_gpu(self.wgpu_state.queue(), bytemuck::cast_slice(image.as_slice()));
                    // Sample Medium interaction(wavefront_depth)
                    // Handle escaped rays
                    // handle emissive intersection
                    // if wavefront_depth == wavefront_max_depth { break; }
                    // eval materials and BSDFs
                    // trace shadow rays
                    // sample subsurface
                    wavefront_depth += 1;
                }
                // update film
            }
        }
    }

    pub fn run_display_kernel(&mut self, gui: &mut GUI)
    {
        // on cpu version there is no compute kernel frame buffer so we have to update it here
        let (width, height) = self.render_parameters.viewport_size();
        let frame = self.render_progress.get(width, height);
        self.frame_buffer.queue_for_gpu(self.wgpu_state.queue(), bytemuck::cast_slice(&[frame]));

        let output = self.wgpu_state.surface().get_current_texture().unwrap();
        let view = output.texture.create_view(
            &wgpu::TextureViewDescriptor::default());

        let mut encoder = self.wgpu_state.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("display kernel encoder"),
            });

        {
            let mut display_pass = encoder.begin_render_pass(
                &wgpu::RenderPassDescriptor {
                    label: Some("display render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None
                });
            display_pass.set_pipeline(&self.display_pipeline);
            display_pass.set_bind_group(0, &self.display_bind_group, &[]);
            display_pass.draw(0..6, 0..1);

            gui.imgui_renderer.render(
                gui.imgui.render(), self.wgpu_state.queue(), self.wgpu_state.device(), &mut display_pass
            ).expect("failed to render gui");
        }
        self.wgpu_state.queue().submit(Some(encoder.finish()));
        output.present();
    }

    pub fn progress(&self) -> f32 {
        self.render_progress.progress()
    }

    pub fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn get_render_parameters(&self) -> RenderParameters {
        self.render_parameters.clone()
    }

    pub fn update_render_parameters(&mut self, render_parameters: RenderParameters) {
        self.render_parameters = render_parameters
    }

    pub fn update_buffers(&mut self, _queue: &Queue) {
        // if rp is the same as the stored buffer, no need to do anything
        if self.render_parameters == self.last_render_parameters {
            return;
        }

        let camera_controller = self.render_parameters.camera_controller();
        // update the projection matrix
        let (w,h) = self.render_parameters.viewport_size();
        let ar = w as f32 / h as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        self.projection_buffer = ProjectionMatrix::new(camera_controller.vfov_rad(), ar, z_near, z_far).p_inv();

        // update the view matrix
        self.view_buffer = camera_controller.get_view_matrix();

        // update the camera
        self.camera_buffer = camera_controller.get_GPU_camera();

        self.compute_shader.queue_camera(self.camera_buffer);

        self.render_progress.reset();
    }
}
