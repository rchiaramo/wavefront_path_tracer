use crate::bvh::BVHTree;
use crate::compute_shader::ComputeShader;
use crate::gpu_buffer::GPUStorageBuffer;
use crate::gpu_structs::{GPUSamplingParameters};
use crate::gui::GUI;
use crate::parameters::{RenderParameters, RenderProgress};
use crate::scene::Scene;
use wavefront_common::camera_controller::{GPUCamera};
use wavefront_common::gpu_structs::GPUFrameBuffer;
use wavefront_common::projection_matrix::ProjectionMatrix;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BufferAddress, BufferUsages, Device, Queue, RenderPipeline, ShaderStages, Surface, TextureFormat};
use winit::event::WindowEvent;

pub struct PathTracer {
    image_buffer: GPUStorageBuffer,
    frame_buffer: GPUStorageBuffer,
    camera_buffer: GPUCamera,
    sampling_parameters_buffer: GPUSamplingParameters,
    projection_buffer: [[f32;4];4],
    view_buffer: [[f32;4];4],
    display_bind_group: BindGroup,
    display_pipeline: RenderPipeline,
    render_parameters: RenderParameters,
    last_render_parameters: RenderParameters,
    render_progress: RenderProgress,
    compute_shader: ComputeShader
}

impl PathTracer {
    pub fn new(device: &Device,
               max_window_size: u32,
               window_size: (u32, u32),
               scene: &mut Scene,
               rp: &RenderParameters)
        -> Option<Self> {
        // create the image_buffer that the compute shader will use to store image
        // we make this array as big as the largest possible window on resize
        let image = vec![[0.0f32; 3]; max_window_size as usize];
        let image_buffer =
            GPUStorageBuffer::new_from_bytes(device,
                                             BufferUsages::STORAGE,
                                             0u32,
                                             bytemuck::cast_slice(image.as_slice()),
                                             Some("image buffer"));

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
        let mut bvh_tree= BVHTree::new(scene.spheres.len());
        bvh_tree.build_bvh_tree(&mut scene.spheres);
        
        let spheres_buffer = scene.spheres.clone();
        let materials_buffer = scene.materials.clone();
        let bvh_buffer = bvh_tree.nodes;

        // create the parameters bind group to interact with GPU during runtime
        // this will include the camera, and the sampling parameters
        let render_parameters= rp.clone();
        let camera_controller = render_parameters.camera_controller();
        let (width, height) = render_parameters.get_viewport();
        let ar = width as f32 / height as f32;
        let (z_near, z_far) = camera_controller.get_clip_planes();
        let projection_buffer = ProjectionMatrix::new(
            camera_controller.vfov_rad(), ar, z_near,z_far).p_inv();

        let view_buffer = camera_controller.get_view_matrix();

        let gpu_sampling_parameters_buffer
            = GPUSamplingParameters::get_gpu_sampling_params(render_parameters.sampling_parameters());

        let camera_buffer = camera_controller.get_GPU_camera();

        let last_render_parameters = render_parameters.clone();

        let spf = render_parameters.sampling_parameters().samples_per_frame;
        let spp= render_parameters.sampling_parameters().samples_per_pixel;
        let nb = render_parameters.sampling_parameters().num_bounces;
        let render_progress = RenderProgress::new(spf, spp, nb);
        
        let compute_shader = ComputeShader::new(spheres_buffer,
                                                materials_buffer,
                                                bvh_buffer,
                                                camera_buffer,
                                                projection_buffer,
                                                view_buffer,
                                                gpu_sampling_parameters_buffer);

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
            multisample: wgpu::MultisampleState{
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Some(Self {
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
            compute_shader
        })

    }



    pub fn run_compute_kernel(&mut self, _device: &Device, queue: &Queue) { //, queries: &mut Queries) {
        let size = self.render_parameters.get_viewport();

        // on cpu version, all compute kernel buffers have to be "queued" by copying them to the
        // compute_shader structure
        let frame = self.render_progress.get_next_frame(&mut self.render_parameters);
        self.compute_shader.queue_frame(frame);

        self.last_render_parameters = self.get_render_parameters();


        let gpu_sampling_parameters
            = GPUSamplingParameters::get_gpu_sampling_params(self.render_parameters.sampling_parameters());
        self.sampling_parameters_buffer = gpu_sampling_parameters;
        self.compute_shader.queue_sampling(self.sampling_parameters_buffer.clone());

        // self.compute_shader.run_render(queue, size, &mut self.image_buffer);
        self.compute_shader.run_parallel_render(queue, size, &mut self.image_buffer);
    }


}