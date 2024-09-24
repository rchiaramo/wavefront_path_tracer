use std::rc::Rc;
use std::sync::Arc;
use wavefront_common::bvh::BVHTree;
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::parameters::{RenderParameters, RenderProgress, SPF, SPP};
use wavefront_common::projection_matrix::ProjectionMatrix;
use wavefront_common::scene::Scene;
use wavefront_common::ray::Ray;
use wgpu::{BufferUsages, Device, Maintain};
use winit::event::WindowEvent;
use wavefront_common::wgpu_state::WgpuState;
use crate::accumulate::AccumulateKernel;
use crate::shade::ShadeKernel;
use crate::display::DisplayKernel;
use crate::extend::ExtendKernel;
use crate::generate_ray::GenerateRayKernel;
use crate::miss::MissKernel;

pub struct PathTracer {
    wgpu_state: Rc<WgpuState>,
    image_buffer: GPUBuffer,
    accumulated_image_buffer: GPUBuffer,
    frame_buffer: GPUBuffer,
    ray_buffer: GPUBuffer,
    extension_ray_buffer: GPUBuffer,
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
    generate_ray_kernel: GenerateRayKernel,
    extend_kernel: ExtendKernel,
    shade_kernel: ShadeKernel,
    miss_kernel: MissKernel,
    accumulate_kernel: AccumulateKernel,
    display_kernel: DisplayKernel,
    render_parameters: RenderParameters,
    render_progress: RenderProgress
}

impl PathTracer {
    pub fn new(window: Arc<winit::window::Window>,
               max_window_size: u32,
               scene: &mut Scene,
               rp: &RenderParameters)
        -> Self {
        // create the connection to the GPU
        let wgpu_state = Rc::new(WgpuState::new(window));

        // create the image_buffer that the compute shader will use to store image
        // we make the buffers as big as the largest possible window on resize
        let mut image = vec![[1.0f32; 3]; max_window_size as usize];
        let image_buffer = 
            GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                      bytemuck::cast_slice(image.as_slice()),
                                      Some("image buffer"));

        image = vec![[0f32;3]; max_window_size as usize];
        let accumulated_image_buffer =
            GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                      bytemuck::cast_slice(image.as_slice()),
                                      Some("accumulated image buffer"));

        // create the frame_buffer
        let frame_buffer = GPUBuffer::new(Rc::clone(&wgpu_state),
                                          BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                          16,
                                          Some("frame buffer"));

        // create the ray_buffer
        let rays = vec![Ray::default(); max_window_size as usize];
        let ray_buffer =
            GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                      bytemuck::cast_slice(rays.as_slice()),
                                      Some("ray buffer"));

        let extension_ray_buffer
            = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                        bytemuck::cast_slice(rays.as_slice()),
                                        Some("extension ray buffer"));

        // create the miss buffer
        let misses = vec!(0u32; max_window_size as usize);
        let miss_buffer =
            GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                      BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                      bytemuck::cast_slice(misses.as_slice()),
                                      Some("misses buffer"));

        // create the hit buffer
        let hit_buffer
            = GPUBuffer::new(Rc::clone(&wgpu_state),
                             BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                             (32 * max_window_size) as usize,
                             Some("hit buffer"));

        // create the counter buffer
        let counter = vec![0u32; 16];
        let counter_buffer =
            GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                      BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                                      bytemuck::cast_slice(counter.as_slice()),
                                      Some("counter_buffer"));

        let counter_read_buffer =
            GPUBuffer::new(Rc::clone(&wgpu_state),
                           BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                           4 * 16,
                           Some("counter read buffer"));

        // create the scene and the bvh_tree that corresponds to it
        let mut bvh_tree= BVHTree::new(scene.spheres.len());
        bvh_tree.build_bvh_tree(&mut scene.spheres);

        let spheres_buffer = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state), BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                                       bytemuck::cast_slice(scene.spheres.as_slice()),
                                                       Some("spheres buffer"));
        let materials_buffer = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state), BufferUsages::STORAGE | BufferUsages::COPY_DST,
                                                         bytemuck::cast_slice(scene.materials.as_slice()),
                                                         Some("materials buffer"));
        let bvh_buffer = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state), BufferUsages::STORAGE | BufferUsages::COPY_DST,
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

        let gpu_camera = camera_controller.get_GPU_camera();

        let camera_buffer = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                                      BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                      bytemuck::cast_slice(&[gpu_camera]),
                                                      Some("camera buffer"));

        let projection_buffer = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                                          BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                          bytemuck::cast_slice(&[proj_mat]),
                                                          Some("projection buffer"));

        let view_buffer = GPUBuffer::new_from_bytes(Rc::clone(&wgpu_state),
                                                    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                                                    bytemuck::cast_slice(&[view_mat]),
                                                    Some("view buffer"));

        let render_progress = RenderProgress::new();

        let generate_ray_kernel
            = GenerateRayKernel::new(Rc::clone(&wgpu_state),
                                     &ray_buffer,
                                     &frame_buffer,
                                     &camera_buffer,
                                     &projection_buffer,
                                     &view_buffer);

        let extend_kernel
            = ExtendKernel::new(Rc::clone(&wgpu_state),
                                &frame_buffer,
                                &ray_buffer,
                                &miss_buffer,
                                &hit_buffer,
                                &counter_buffer,
                                &spheres_buffer,
                                &bvh_buffer);

        let shade_kernel
            = ShadeKernel::new(Rc::clone(&wgpu_state),
                               &image_buffer,
                               &frame_buffer,
                               &ray_buffer,
                               &extension_ray_buffer,
                               &hit_buffer,
                               &counter_buffer,
                               &spheres_buffer,
                               &materials_buffer
        );

        let miss_kernel
            = MissKernel::new(Rc::clone(&wgpu_state), &image_buffer,
                              &ray_buffer, &miss_buffer,
                              &counter_buffer);

        let accumulate_kernel
            = AccumulateKernel::new(Rc::clone(&wgpu_state),
                                    &image_buffer,
                                    &accumulated_image_buffer);

        let display_kernel
            = DisplayKernel::new(Rc::clone(&wgpu_state), &accumulated_image_buffer, &frame_buffer);

        Self {
            wgpu_state,
            image_buffer,
            accumulated_image_buffer,
            frame_buffer,
            ray_buffer,
            extension_ray_buffer,
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
            generate_ray_kernel,
            extend_kernel,
            shade_kernel,
            miss_kernel,
            accumulate_kernel,
            display_kernel,
            render_parameters,
            render_progress
        }
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

    pub fn resize(&mut self, rp: RenderParameters) {
        self.wgpu_state.resize(rp.viewport_size());
        self.update_render_parameters(rp);

    }
    pub fn update_render_parameters(&mut self, render_parameters: RenderParameters) {
        self.render_parameters = render_parameters;
    }

    pub fn update_buffers(&mut self) {
        // if nothing changed, no need to do anything
        if !self.render_parameters.changed() {
            return;
        }

        // otherwise something changed

        // clear the accumulated image buffer
        let size = self.accumulated_image_buffer.size() / 12;
        self.accumulated_image_buffer.queue_for_gpu(bytemuck::cast_slice(&vec![[0f32; 3]; size]));

        // get the possibly updated camera_controller
        let camera_controller = self.render_parameters.camera_controller();

        // if the camera position or orientation changed, update the view matrix and the camera itself
        // if the window was resized or vfov altered, update the projection matrix
        // right now, I'll just update both all the time

        // update the view matrix
        let view_mat = camera_controller.get_view_matrix();
        self.view_buffer.queue_for_gpu(bytemuck::cast_slice(&[view_mat]));

        // update the camera
        let gpu_camera = self.render_parameters.camera_controller().get_GPU_camera();
        self.camera_buffer.queue_for_gpu(bytemuck::cast_slice(&[gpu_camera]));

        // update the projection matrix
        let (w, h) = self.render_parameters.viewport_size();
        let ar = w as f32 / h as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let proj_mat = ProjectionMatrix::new(camera_controller.vfov_rad(), ar, z_near, z_far).p_inv();
        self.projection_buffer.queue_for_gpu(bytemuck::cast_slice(&[proj_mat]));

        // reset all flags after changes made
        self.render_parameters.reset();
        self.render_progress.reset();
    }

    pub fn run(&mut self) {
        self.update_buffers();

        let workgroup_size = |x:u32| {
            let x_over_32 = x.div_ceil(32);
            let y = (x_over_32 as f32).sqrt().ceil() as u32;
            let fac = (1..y).rev().find(|z| { x_over_32 % z == 0 }).unwrap();
            if (x_over_32 / fac) >= (1<<16) { (y, y) } else {
                (fac, x_over_32 / fac)
            }
        };

        if self.render_progress.accumulated_samples() < SPP {
            // always update the frame
            let mut frame = self.render_progress.get_next_frame(&mut self.render_parameters);
            let samples_per_frame = SPF;
            for sample_number in 0..samples_per_frame {
                frame.set_sample_number(sample_number);
                self.frame_buffer.queue_for_gpu(bytemuck::cast_slice(&[frame]));

                // fundamental rendering loop
                // remember this gets called every frame and renders the whole scene,
                // accumulating pixel color
                // therefore, we generate new initial rays every time here

                // clear the image buffer
                let size = self.image_buffer.size() / 12;
                self.image_buffer.queue_for_gpu(bytemuck::cast_slice(&vec![[1f32; 3]; size]));

                // clear the ray buffers
                self.ray_buffer.clear_buffer();
                self.extension_ray_buffer.clear_buffer();

                // set the counter buffer initially to run through the entire image
                let (width, height) = self.render_parameters.viewport_size();
                let mut counter = vec![0u32; 16];
                counter[2] = width * height;
                self.counter_buffer.queue_for_gpu(bytemuck::cast_slice(&counter));

                self.generate_ray_kernel.run((width / 8, height / 8));

                // the wavefront loop starts here; this is like the old num_bounces
                let mut wavefront = 0;
                let mut extend_size = workgroup_size(width * height);
                while wavefront < 50 {

                    self.extend_kernel.run(extend_size);

                    self.wgpu_state.copy_buffer_to_buffer(&self.counter_buffer, &self.counter_read_buffer);
                    let mut counter = self.wgpu_state.read_buffer(&self.counter_read_buffer);

                    let num_misses = counter[0];
                    let num_hits = counter[1];
                    if num_misses < 50 { println!("wavefront {wavefront}"); break }

                    // after we know number of hits and misses, reset the ray counter
                    counter[2] = 0;
                    self.counter_buffer.queue_for_gpu(bytemuck::cast_slice(&counter));

                    // now do shading, get extension rays, and handle misses
                    self.shade_kernel.run(workgroup_size(num_hits));
                    self.miss_kernel.run(workgroup_size(num_misses));

                    // find the number of extension rays
                    self.wgpu_state.copy_buffer_to_buffer(&self.counter_buffer, &self.counter_read_buffer);
                    counter = self.wgpu_state.read_buffer(&self.counter_read_buffer);
                    let num_extension = counter[2];

                    // clear the ray buffer and copy the extension_ray_buffer into it; then clear the extension_ray_buffer
                    self.wgpu_state.copy_buffer_to_buffer(&self.extension_ray_buffer, &self.ray_buffer);

                    extend_size = workgroup_size(num_extension);

                    self.counter_buffer.queue_for_gpu(bytemuck::cast_slice(&[0u32, 0, num_extension, 0]));
                    wavefront += 1;

                }
                let gr_timing = self.generate_ray_kernel.get_timing();
                let er_timing = self.extend_kernel.get_timing();
                let sh_timing = self.shade_kernel.get_timing();
                let miss_timing = self.miss_kernel.get_timing();
                // sloppy solution right now, but I'm double using the sample_number for the display shader
                // by storing the total accumulated samples in that variable
                self.accumulate_kernel.run(workgroup_size(width * height));
                self.render_progress.incr_accumulated_samples(1);
                println!("sample: {} total: {} gr: {gr_timing} er: {er_timing} sh: {sh_timing} miss: {miss_timing}",
                         self.render_progress.accumulated_samples(), gr_timing+er_timing+sh_timing+miss_timing);
                frame.set_sample_number(self.render_progress.accumulated_samples());
                self.frame_buffer.queue_for_gpu(bytemuck::cast_slice(&[frame]));
            }
        }
        self.display_kernel.run();
    }
}