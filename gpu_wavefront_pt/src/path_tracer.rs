use crate::query_gpu::Queries;
use wavefront_common::bvh::BVHTree;
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::gpu_structs::{GPUSamplingParameters};
use wavefront_common::gui::GUI;
use wavefront_common::parameters::{RenderParameters, RenderProgress};
use wavefront_common::projection_matrix::ProjectionMatrix;
use wavefront_common::scene::Scene;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BufferAddress, BufferUsages, ComputePassTimestampWrites, Device, Queue, RenderPipeline, ShaderStages, Surface, TextureFormat};
use winit::event::WindowEvent;

pub struct PathTracer {
    image_buffer: GPUBuffer,
    frame_buffer: GPUBuffer,
    image_bind_group: BindGroup,
    spheres_buffer: GPUBuffer,
    materials_buffer: GPUBuffer,
    bvh_buffer:  GPUBuffer,
    scene_bind_group: BindGroup,
    camera_buffer: GPUBuffer,
    sampling_parameters_buffer: GPUBuffer,
    projection_buffer: GPUBuffer,
    view_buffer: GPUBuffer,
    parameters_bind_group: BindGroup,
    compute_shader_pipeline: wgpu::ComputePipeline,
    display_bind_group: BindGroup,
    display_pipeline: RenderPipeline,
    render_parameters: RenderParameters,
    last_render_parameters: RenderParameters,
    render_progress: RenderProgress
}

impl PathTracer {
    pub fn new(device: &Device,
               max_window_size: u32,
               scene: &mut Scene,
               rp: &RenderParameters)
        -> Option<Self> {
        // create the image_buffer that the compute shader will use to store image
        // we make this array as big as the largest possible window on resize
        let image = vec![[0.0f32; 3]; max_window_size as usize];
        let image_buffer = 
            GPUBuffer::new_from_bytes(device, 
                                      BufferUsages::STORAGE, 
                                      0u32, 
                                      bytemuck::cast_slice(image.as_slice()), 
                                      Some("image buffer"));
        
        // create the frame_buffer
        let frame_buffer = GPUBuffer::new(device, 
                                          BufferUsages::UNIFORM,
                                          16 as BufferAddress,
                                          1u32,
                                          Some("frame buffer"));
        
        // group image and frame buffers into image bind group
        let image_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("image bind group layout"),
                entries: &[image_buffer.layout(ShaderStages::COMPUTE, false), 
                    frame_buffer.layout(ShaderStages::COMPUTE, false)
                ],
            });
        
        let image_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("image bind group"),
            layout: &image_bind_group_layout,
            entries: &[image_buffer.binding(), frame_buffer.binding()],
        });
        
        // create the scene and the bvh_tree that corresponds to it
        let mut bvh_tree= BVHTree::new(scene.spheres.len());
        bvh_tree.build_bvh_tree(&mut scene.spheres);
        
        let spheres_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE,
                                                0u32,
                                                bytemuck::cast_slice(scene.spheres.as_slice()),
                                                Some("spheres buffer"));
        let materials_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE,
                                                  1u32,
                                                  bytemuck::cast_slice(scene.materials.as_slice()),
                                                  Some("materials buffer"));
        let bvh_buffer = GPUBuffer::new_from_bytes(device, BufferUsages::STORAGE,
                                                  2u32,
                                                  bytemuck::cast_slice(bvh_tree.nodes.as_slice()),
                                                  Some("bvh_tree buffer"));
        
        // the scene bind group will hold the primitives, the materials, and the bvh_tree
        let scene_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("scene bind group layout"),
                entries: &[spheres_buffer.layout(ShaderStages::COMPUTE, true),
                    materials_buffer.layout(ShaderStages::COMPUTE, true),
                    bvh_buffer.layout(ShaderStages::COMPUTE, true)],
            });
        
        let scene_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("scene bind group"),
            layout: &scene_bind_group_layout,
            entries: &[spheres_buffer.binding(), materials_buffer.binding(), bvh_buffer.binding()],
        });
        
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
                                                      0u32,
                                                      bytemuck::cast_slice(&[gpu_camera]),
                                                      Some("camera buffer"));

        let sampling_parameters_buffer = GPUBuffer::new_from_bytes(device,
                                                                   BufferUsages::UNIFORM,
                                                                   1u32, 
                                                                   bytemuck::cast_slice(&[gpu_sampling_params]), 
                                                                   Some("sampling parameters buffer"));

        let projection_buffer = GPUBuffer::new_from_bytes(device,
                                                          BufferUsages::UNIFORM,
                                                          2u32,
                                                          bytemuck::cast_slice(&[proj_mat]),
                                                          Some("projection buffer"));

        let view_buffer = GPUBuffer::new_from_bytes(device,
                                                 BufferUsages::UNIFORM,
                                                 3u32,
                                                 bytemuck::cast_slice(&[view_mat]),
                                                 Some("view buffer"));
        
        let parameters_bind_group_layout = 
            device.create_bind_group_layout(&BindGroupLayoutDescriptor{
                label: Some("parameters bind group layout"),
                entries: &[camera_buffer.layout(ShaderStages::COMPUTE, false), 
                    sampling_parameters_buffer.layout(ShaderStages::COMPUTE, false),
                    projection_buffer.layout(ShaderStages::COMPUTE, false),
                    view_buffer.layout(ShaderStages::COMPUTE, false)
                ],
        });
        
        let parameters_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("parameters bind group"),
            layout: &parameters_bind_group_layout,
            entries: &[camera_buffer.binding(),
                sampling_parameters_buffer.binding(),
                projection_buffer.binding(),
                view_buffer.binding()
            ],
        });

        let last_render_parameters = render_parameters.clone();

        let spf = render_parameters.sampling_parameters().samples_per_frame;
        let spp= render_parameters.sampling_parameters().samples_per_pixel;
        let nb = render_parameters.sampling_parameters().num_bounces;
        let render_progress = RenderProgress::new(spf, spp, nb);

        // create the compute pipeline
        let ray_tracer_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute shader pipeline layout"),
                bind_group_layouts: &[
                    &image_bind_group_layout,
                    &scene_bind_group_layout,
                    &parameters_bind_group_layout
                ],
                push_constant_ranges: &[],
            }
        );

        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/compute_megakernel.wgsl")
        );
        
        // if I want to pass in override values, I can do it here:
        // let mut id:HashMap<String, f64> = HashMap::new();
        // id.insert("stackSize".to_string(), (bvh_tree.nodes.len() - 1) as f64);
        
        let compute_shader_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("compute shader pipeline"),
                layout: Some(&ray_tracer_pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                // PipelineCompilationOptions {
                //     constants: None, //&id,
                //     zero_initialize_workgroup_memory: false,
                //     vertex_pulling_transform: false,
                // },
                cache: None,
            }
        );

        // create the display shader
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
            label: Some("display Pipeline"),
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
            image_bind_group,
            spheres_buffer,
            materials_buffer,
            bvh_buffer,
            scene_bind_group,
            camera_buffer,
            sampling_parameters_buffer,
            projection_buffer,
            view_buffer,
            parameters_bind_group,
            compute_shader_pipeline,
            display_bind_group,
            display_pipeline,
            render_parameters,
            last_render_parameters,
            render_progress
        })

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

    pub fn update_buffers(&mut self, queue: &Queue) {
        // if rp is the same as the stored buffer, no need to do anything
        if self.render_parameters == self.last_render_parameters {
            return;
        }

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

    pub fn run_compute_kernel(&mut self, device: &Device, queue: &Queue, queries: &mut Queries) {
        let size = self.render_parameters.get_viewport();

        let frame = self.render_progress.get_next_frame(&mut self.render_parameters);
        self.last_render_parameters = self.get_render_parameters();
        self.frame_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[frame]));

        let gpu_sampling_parameters
            = GPUSamplingParameters::get_gpu_sampling_params(self.render_parameters.sampling_parameters());
        self.sampling_parameters_buffer.queue_for_gpu(queue, bytemuck::cast_slice(&[gpu_sampling_parameters]));

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("compute kernel encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute pass"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &queries.set,
                    beginning_of_pass_write_index: Some(queries.next_unused_query),
                    end_of_pass_write_index: Some(queries.next_unused_query + 1),
                })
            });
            queries.next_unused_query += 2;
            compute_pass.set_pipeline(&self.compute_shader_pipeline);
            compute_pass.set_bind_group(0, &self.image_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.scene_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.parameters_bind_group, &[]);
            compute_pass.dispatch_workgroups(size.0 / 4, size.1 / 4, 1);

        }
        queries.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    pub fn run_display_kernel(&mut self, surface: &mut Surface,
                  device: &Device, queue: &Queue, gui: &mut GUI) {

        let output = surface.get_current_texture().unwrap();
        let view = output.texture.create_view(
            &wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(
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
                gui.imgui.render(), queue, device, &mut display_pass
            ).expect("failed to render gui");
        }
        queue.submit(Some(encoder.finish()));
        output.present();
    }
}