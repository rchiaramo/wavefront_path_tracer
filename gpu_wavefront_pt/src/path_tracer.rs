use std::sync::Arc;
use glam::{Mat4, UVec2};
use imgui::Key::B;
use crate::query_gpu::Queries;
use wavefront_common::bvh::BVHTree;
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::gpu_structs::{GPUSamplingParameters};
use wavefront_common::gui::GUI;
use wavefront_common::parameters::{RenderParameters, RenderProgress, SamplingParameters};
use wavefront_common::projection_matrix::ProjectionMatrix;
use wavefront_common::scene::Scene;
use wavefront_common::ray::Ray;
use wgpu::{BufferAddress, BufferUsages, ComputePassTimestampWrites, Device, Queue, ShaderStages, Surface, TextureFormat};
use winit::event::WindowEvent;
use wavefront_common::wgpu_state::WgpuState;
use crate::shade::ShadeKernel;
use crate::display::DisplayKernel;
use crate::extend::ExtendKernel;
use crate::generate_ray::GenerateRayKernel;
use crate::miss::MissKernel;

pub struct PathTracer<'a> {
    wgpu_state: WgpuState<'a>,
    image_buffer: GPUBuffer,
    frame_buffer: GPUBuffer,
    ray_buffer: GPUBuffer,
    miss_buffer: GPUBuffer,
    hit_buffer: GPUBuffer,
    counter_buffer: GPUBuffer,
    counter_read_buffer: GPUBuffer,
    spheres_buffer: GPUBuffer,
    materials_buffer: GPUBuffer,
    bvh_buffer: GPUBuffer,
    camera_buffer: GPUBuffer,
    projection_buffer: GPUBuffer,
    view_buffer: GPUBuffer,
    sampling_parameters_buffer: GPUBuffer,
    generate_ray_kernel: GenerateRayKernel,
    extend_kernel: ExtendKernel,
    compute_rest_kernel: ShadeKernel,
    miss_kernel: MissKernel,
    display_kernel: DisplayKernel,
    render_parameters: RenderParameters,
    sampling_parameters: SamplingParameters,
    render_progress: RenderProgress
}

impl<'a> PathTracer<'a> {
    pub fn new(window: Arc<winit::window::Window>,
               max_window_size: u32,
               scene: &mut Scene,
               rp: &RenderParameters,
               sp: &SamplingParameters)
        -> Self {
        // create the connection to the GPU
        let wgpu_state = WgpuState::new(window);
        let device = wgpu_state.device();

        // create the image_buffer that the compute shader will use to store image
        // we make the buffers as big as the largest possible window on resize
        let image = vec![[0.0f32; 3]; max_window_size as usize];
        let image_buffer = 
            GPUBuffer::new_from_bytes(device,
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                      bytemuck::cast_slice(image.as_slice()),
                                      Some("image buffer"));

        // create the frame_buffer
        let frame_buffer = GPUBuffer::new(device,
                                          BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                          16 as BufferAddress,
                                          Some("frame buffer"));

        // create the ray_buffer
        let rays = vec![Ray::default(); max_window_size as usize];
        let ray_buffer =
            GPUBuffer::new_from_bytes(device,
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                      bytemuck::cast_slice(rays.as_slice()),
                                      Some("ray buffer"));

        // create the miss buffer
        let misses = vec!(0u32; max_window_size as usize);
        let miss_buffer =
            GPUBuffer::new_from_bytes(device,
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                      bytemuck::cast_slice(misses.as_slice()),
                                      Some("misses buffer"));

        // create the hit buffer
        let hit_buffer
            = GPUBuffer::new(device,
                             BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                             32 * max_window_size as BufferAddress,
                             Some("hit buffer"));

        // create the counter buffer
        let counter = vec![0u32; 16];
        let counter_buffer =
            GPUBuffer::new_from_bytes(device,
                                      BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                                      bytemuck::cast_slice(counter.as_slice()),
                                      Some("counter_buffer"));

        let counter_read_buffer =
            GPUBuffer::new(device,
                           BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                           4 * 16 as BufferAddress,
                           Some("counter read buffer"));

        // create the scene and the bvh_tree that corresponds to it
        let mut bvh_tree= BVHTree::new(scene.spheres.len());
        bvh_tree.build_bvh_tree(&mut scene.spheres);

        let spheres_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                                              bytemuck::cast_slice(scene.spheres.as_slice()),
                                                              Some("spheres buffer"));
        let materials_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                                                bytemuck::cast_slice(scene.materials.as_slice()),
                                                                Some("materials buffer"));
        let bvh_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                                          bytemuck::cast_slice(bvh_tree.nodes.as_slice()),
                                                          Some("bvh_tree buffer"));
        
        // create the parameters bind group to interact with GPU during runtime
        // this will include the camera controller, the sampling parameters, and the window size
        let render_parameters= rp.clone();
        let camera_controller = render_parameters.camera_controller();
        let (width, height) = render_parameters.viewport_size();
        let ar = width as f32 / height as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let proj_mat = ProjectionMatrix::new(
            camera_controller.vfov_rad(), ar, z_near,z_far).p_inv();
        let view_mat = camera_controller.get_view_matrix();
        let sampling_parameters = sp.clone();
        let gpu_sampling_params
            = GPUSamplingParameters::get_gpu_sampling_params(&sampling_parameters);

        let gpu_camera = camera_controller.get_GPU_camera();

        let camera_buffer = GPUBuffer::new_from_bytes(device,
                                                             BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                             bytemuck::cast_slice(&[gpu_camera]),
                                                             Some("camera buffer"));

        let sampling_parameters_buffer = GPUBuffer::new_from_bytes(device,
                                                                          BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                                          bytemuck::cast_slice(&[gpu_sampling_params]),
                                                                          Some("sampling parameters buffer"));

        let projection_buffer = GPUBuffer::new_from_bytes(device,
                                                                 BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                                 bytemuck::cast_slice(&[proj_mat]),
                                                                 Some("projection buffer"));

        let view_buffer = GPUBuffer::new_from_bytes(device,
                                                           BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                           bytemuck::cast_slice(&[view_mat]),
                                                           Some("view buffer"));

        let render_progress = RenderProgress::new();

        let generate_ray_kernel
            = GenerateRayKernel::new(device,
                                     &ray_buffer,
                                     &frame_buffer,
                                     &camera_buffer,
                                     &projection_buffer,
                                     &view_buffer);

        let extend_kernel
            = ExtendKernel::new(device,
                                &frame_buffer,
                                &ray_buffer,
                                &miss_buffer,
                                &hit_buffer,
                                &counter_buffer,
                                &spheres_buffer,
                                &bvh_buffer);

        let compute_rest_kernel
            = ShadeKernel::new(device,
                               &image_buffer,
                               &frame_buffer,
                               &ray_buffer,
                               &hit_buffer,
                               &counter_buffer,
                               &spheres_buffer,
                               &materials_buffer,
                               &bvh_buffer,
                               &camera_buffer,
                               &sampling_parameters_buffer);

        let miss_kernel = MissKernel::new(device, &image_buffer, &ray_buffer, &miss_buffer);

        let display_kernel = DisplayKernel::new(device, &image_buffer, &frame_buffer);

        Self {
            wgpu_state,
            image_buffer,
            frame_buffer,
            ray_buffer,
            miss_buffer,
            hit_buffer,
            counter_buffer,
            counter_read_buffer,
            spheres_buffer,
            materials_buffer,
            bvh_buffer,
            camera_buffer,
            projection_buffer,
            view_buffer,
            sampling_parameters_buffer,
            generate_ray_kernel,
            extend_kernel,
            compute_rest_kernel,
            miss_kernel,
            display_kernel,
            render_parameters,
            sampling_parameters,
            render_progress
        }
    }

    pub fn wgpu_state(&'a self) -> &'a WgpuState {
        &self.wgpu_state
    }

    pub fn progress(&self) -> f32 {
        self.render_progress.progress(self.sampling_parameters.samples_per_pixel)
    }

    pub fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn get_render_parameters(&self) -> RenderParameters {
        self.render_parameters.clone()
    }

    pub fn resize(&mut self, rp: RenderParameters) {
        self.wgpu_state.resize(rp.viewport_size());
        self.update_render_parameters(rp);

    }
    pub fn update_render_parameters(&mut self, render_parameters: RenderParameters) {
        self.render_parameters = render_parameters;
    }

    pub fn update_buffers(&mut self) {
        let queue = self.wgpu_state.queue();

        // if nothing changed, no need to do anything
        if !self.render_parameters.changed() {
            let sampling_parameters = SamplingParameters::new(1,50, 0, 500);
            let gpu_sampling_parameters
                = GPUSamplingParameters::get_gpu_sampling_params(&sampling_parameters);
            self.sampling_parameters_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[gpu_sampling_parameters]));
            return;
        }

        // otherwise something changed
        let sampling_parameters = SamplingParameters::new(1,50, 1, 500);
        let gpu_sampling_parameters
            = GPUSamplingParameters::get_gpu_sampling_params(&sampling_parameters);
        self.sampling_parameters_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[gpu_sampling_parameters]));

        let camera_controller = self.render_parameters.camera_controller();

        // if the camera position or orientation changed, update the view matrix and the camera itself
        // if the window was resized or vfov altered, update the projection matrix
        // right now, I'll just update both all the time

        // update the view matrix
        let view_mat = camera_controller.get_view_matrix();
        self.view_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[view_mat]));

        // update the camera
        let gpu_camera = self.render_parameters.camera_controller().get_GPU_camera();
        self.camera_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[gpu_camera]));

        // update the projection matrix
        let (w, h) = self.render_parameters.viewport_size();
        let ar = w as f32 / h as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let proj_mat = ProjectionMatrix::new(camera_controller.vfov_rad(), ar, z_near, z_far).p_inv();
        self.projection_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[proj_mat]));

        // reset all flags after changes made
        self.render_parameters.reset();
        self.render_progress.reset();
    }

    pub fn run(&mut self) {
        self.update_buffers();

        if self.render_progress.accumulated_samples() < self.sampling_parameters.samples_per_pixel {
            // always update the frame
            let mut frame = self.render_progress.get_next_frame(&mut self.render_parameters);
            let samples_per_frame = self.sampling_parameters.samples_per_frame;
            for sample_number in 0..samples_per_frame {
                {
                    let device = self.wgpu_state.device();
                    let queue = self.wgpu_state.queue();

                    frame.set_sample_number(sample_number);
                    self.frame_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[frame]));

                    let (width, height) = self.render_parameters.viewport_size();

                    // fundamental rendering loop
                    // remember this gets called every frame and renders the whole scene,
                    // accumulating pixel color
                    // therefore, we generate new initial rays every time here
                    self.generate_ray_kernel.run(device,
                                                 queue,
                                                 (width / 8, height / 4));

                    // the wavefront loop will start here
                    self.extend_kernel.run(device,
                                           queue,
                                           (width / 8, height / 4),
                                           &self.counter_buffer,
                                           &self.counter_read_buffer);

                    self.counter_read_buffer.name()
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, |_| ());
                    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

                    let counter: Vec<u32>  = {
                        let counter_view = self
                            .counter_read_buffer.name()
                            .slice(..)
                            .get_mapped_range();
                        bytemuck::cast_slice(&counter_view).to_vec()
                    };

                    let num_misses = counter[0];
                    let num_hits = counter[1];
                    self.counter_read_buffer.name().unmap();

                    self.compute_rest_kernel.run(device,
                                                 queue,
                                                 (num_hits / 64, 64),
                                                 &self.counter_buffer,
                                                 &self.counter_read_buffer);

                    self.render_progress.incr_accumulated_samples(samples_per_frame);


                    self.miss_kernel.run(&device, &queue, (num_misses / 64, 64));
                    self.counter_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[0u32;4]));
                }
            }
            // sloppy solution right now, but I'm double using the sample_number for the display shader
            // by storing the total accumulated samples in that variable
            let queue = self.wgpu_state.queue();
            frame.set_sample_number(self.render_progress.accumulated_samples());
            self.frame_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[frame]));

            self.display_kernel.run(&mut self.wgpu_state);
        }
    }
}