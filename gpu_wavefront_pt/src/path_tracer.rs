use std::sync::Arc;
use glam::{Mat4, UVec2};
use crate::query_gpu::Queries;
use wavefront_common::bvh::BVHTree;
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::gpu_structs::{GPUSamplingParameters};
use wavefront_common::gui::GUI;
use wavefront_common::parameters::{RenderParameters, RenderProgress};
use wavefront_common::projection_matrix::ProjectionMatrix;
use wavefront_common::scene::Scene;
use wavefront_common::ray::Ray;
use wgpu::{BufferAddress, BufferUsages, ComputePassTimestampWrites, Device, Queue, ShaderStages, Surface, TextureFormat};
use winit::event::WindowEvent;
use wavefront_common::wgpu_state::WgpuState;
use crate::compute_rest::ComputeRestKernel;
use crate::display::DisplayKernel;
use crate::generate_ray::GenerateRayKernel;

pub struct PathTracer<'a> {
    wgpu_state: WgpuState<'a>,
    image_buffer: GPUBuffer,
    frame_buffer: GPUBuffer,
    ray_buffer: GPUBuffer,
    spheres_buffer: GPUBuffer,
    materials_buffer: GPUBuffer,
    bvh_buffer: GPUBuffer,
    camera_buffer: GPUBuffer,
    projection_buffer: GPUBuffer,
    view_buffer: GPUBuffer,
    sampling_parameters_buffer: GPUBuffer,
    generate_ray_kernel: GenerateRayKernel,
    compute_rest_kernel: ComputeRestKernel,
    display_kernel: DisplayKernel,
    render_parameters: RenderParameters,
    last_render_parameters: RenderParameters,
    render_progress: RenderProgress
}

impl<'a> PathTracer<'a> {
    pub fn new(window: Arc<winit::window::Window>,
               max_window_size: u32,
               scene: &mut Scene,
               rp: &RenderParameters)
        -> Self {
        // create the connection to the GPU
        let wgpu_state = WgpuState::new(window);
        let device = wgpu_state.device();

        // create the image_buffer that the compute shader will use to store image
        // we make the buffers as big as the largest possible window on resize
        let image = vec![[0.0f32; 3]; max_window_size as usize];
        let image_buffer = 
            GPUBuffer::new_from_bytes(device,
                                             BufferUsages::STORAGE,
                                             bytemuck::cast_slice(image.as_slice()),
                                             Some("image buffer"));

        // create the frame_buffer
        let frame_buffer = GPUBuffer::new(device,
                                                 BufferUsages::UNIFORM,
                                                 16 as BufferAddress,
                                                 1u32,
                                                 Some("frame buffer"));

        // create the ray_buffer
        let rays = vec![Ray::default(); max_window_size as usize];
        let ray_buffer =
            GPUBuffer::new_from_bytes(device,
                                             BufferUsages::STORAGE,
                                             bytemuck::cast_slice(rays.as_slice()),
                                             Some("ray buffer"));

        // create the scene and the bvh_tree that corresponds to it
        let mut bvh_tree= BVHTree::new(scene.spheres.len());
        bvh_tree.build_bvh_tree(&mut scene.spheres);

        let spheres_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE,
                                                              bytemuck::cast_slice(scene.spheres.as_slice()),
                                                              Some("spheres buffer"));
        let materials_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE,
                                                                bytemuck::cast_slice(scene.materials.as_slice()),
                                                                Some("materials buffer"));
        let bvh_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE,
                                                          bytemuck::cast_slice(bvh_tree.nodes.as_slice()),
                                                          Some("bvh_tree buffer"));
        
        // create the parameters bind group to interact with GPU during runtime
        // this will include the camera controller, the sampling parameters, and the window size
        let render_parameters= rp.clone();
        let camera_controller = render_parameters.camera_controller();
        let (width, height) = render_parameters.get_viewport();
        let ar = width as f32 / height as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let proj_mat = ProjectionMatrix::new(
            camera_controller.vfov_rad(), ar, z_near,z_far).p_inv();
        let view_mat = camera_controller.get_view_matrix();
        let gpu_sampling_params
            = GPUSamplingParameters::get_gpu_sampling_params(render_parameters.sampling_parameters());

        let gpu_camera = camera_controller.get_GPU_camera();

        let camera_buffer = GPUBuffer::new_from_bytes(device,
                                                             BufferUsages::UNIFORM,
                                                             bytemuck::cast_slice(&[gpu_camera]),
                                                             Some("camera buffer"));

        let sampling_parameters_buffer = GPUBuffer::new_from_bytes(device,
                                                                          BufferUsages::UNIFORM,
                                                                          bytemuck::cast_slice(&[gpu_sampling_params]),
                                                                          Some("sampling parameters buffer"));

        let projection_buffer = GPUBuffer::new_from_bytes(device,
                                                                 BufferUsages::UNIFORM,
                                                                 bytemuck::cast_slice(&[proj_mat]),
                                                                 Some("projection buffer"));

        let view_buffer = GPUBuffer::new_from_bytes(device,
                                                           BufferUsages::UNIFORM,
                                                           bytemuck::cast_slice(&[view_mat]),
                                                           Some("view buffer"));
        
        // set the viewport of last parameters to something different so that on the first
        // pass, last is different from current render parameters
        let mut last_render_parameters = render_parameters.clone();
        last_render_parameters.set_viewport((0,0));

        let spf = render_parameters.sampling_parameters().samples_per_frame;
        let spp= render_parameters.sampling_parameters().samples_per_pixel;
        let nb = render_parameters.sampling_parameters().num_bounces;
        let render_progress = RenderProgress::new(spf, spp, nb);

        let generate_ray_kernel
            = GenerateRayKernel::new(device,
                                     &ray_buffer,
                                     &frame_buffer,
                                     &camera_buffer,
                                     &projection_buffer,
                                     &view_buffer);

        let compute_rest_kernel
            = ComputeRestKernel::new(device,
                                     &image_buffer,
                                     &frame_buffer,
                                     &ray_buffer,
                                     &spheres_buffer,
                                     &materials_buffer,
                                     &bvh_buffer,
                                     &camera_buffer,
                                     &sampling_parameters_buffer);

        let display_kernel = DisplayKernel::new(device, &image_buffer, &frame_buffer);

        Self {
            wgpu_state,
            image_buffer,
            frame_buffer,
            ray_buffer,
            spheres_buffer,
            materials_buffer,
            bvh_buffer,
            camera_buffer,
            projection_buffer,
            view_buffer,
            sampling_parameters_buffer,
            generate_ray_kernel,
            compute_rest_kernel,
            display_kernel,
            render_parameters,
            last_render_parameters,
            render_progress
        }
    }

    pub fn wgpu_state(&self) -> &WgpuState {
        &self.wgpu_state
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
        self.render_parameters = render_parameters;
        if render_parameters.get_viewport() != self.last_render_parameters.get_viewport() {
            self.wgpu_state.resize(render_parameters.get_viewport());
        }
    }

    pub fn update_buffers(&mut self) {
        // if rp is the same as the stored buffer, no need to do anything
        if self.render_parameters == self.last_render_parameters {
            return;
        }

        let queue = self.wgpu_state.queue();

        let camera_controller = self.render_parameters.camera_controller();
        // update the projection matrix
        let (w,h) = self.render_parameters.get_viewport();
        let ar = w as f32 / h as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let proj_mat = ProjectionMatrix::new(camera_controller.vfov_rad(), ar, z_near, z_far).p_inv();
        self.projection_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[proj_mat]));

        // update the view matrix
        let view_mat = camera_controller.get_view_matrix();
        self.view_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[view_mat]));

        // update the camera
        let gpu_camera = self.render_parameters.camera_controller().get_GPU_camera();
        self.camera_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[gpu_camera]));

        self.render_progress.reset();
    }

    pub fn run(&mut self) {
        self.update_buffers();
        let (width, height) = self.render_parameters.get_viewport();
        let device = self.wgpu_state.device();
        let queue = self.wgpu_state.queue();
        let mut queries = Queries::new(device, 2);

        // fundamental rendering loop

        self.generate_ray_kernel.run(device,
                                     queue,
                                     (width, height),
                                     queries);

    }
}