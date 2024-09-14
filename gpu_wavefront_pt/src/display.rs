use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, RenderPipeline, ShaderStages, Surface, TextureFormat};
use wavefront_common::gpu_buffer::GPUBuffer;
use crate::query_gpu::Queries;

pub struct DisplayKernel {
    display_bind_group: BindGroup,
    pipeline: RenderPipeline
}

impl DisplayKernel {
    // on initialization, a kernel needs to:
    // load a shader
    // create bind group layout and bind group
    // create a pipeline layout and a pipeline
    pub fn new(device: &Device,
               image_buffer: &GPUBuffer,
               frame_buffer: &GPUBuffer) -> Self {

        // load the kernel
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../../wavefront_common/shaders/display_shader.wgsl")
        );

        // create the bind group, here just the image and the frame buffers
        let display_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                label: Some("display bind group layout"),
                entries: &[
                    image_buffer.layout(ShaderStages::FRAGMENT, 0,true),
                    frame_buffer.layout(ShaderStages::FRAGMENT, 1, true)
                ],
            }
        );

        let display_bind_group = device.create_bind_group(
            &BindGroupDescriptor {
                label: Some("display bind group"),
                layout: &display_bind_group_layout,
                entries: &[
                    image_buffer.binding(0),
                    frame_buffer.binding(1)
                ],
            }
        );

        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("display pipeline layout"),
                bind_group_layouts: &[&display_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        Self {
            display_bind_group,
            pipeline
        }
    }

    // when executing, a kernel needs to:
    // possibly get a view (display kernel)
    // create an encoder
    // create a _pass
    // set the pipeline
    // set the bind groups
    // do the version of execute (dispatch workgroups vs draw)
    // submit the encoder through the queue
    // possibly present the output (display kernel)

    pub fn run(&self, surface: &mut Surface, device: &Device, queue: &Queue) {
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
            display_pass.set_pipeline(&self.pipeline);
            display_pass.set_bind_group(0, &self.display_bind_group, &[]);
            display_pass.draw(0..6, 0..1);

            // gui.imgui_renderer.render(
            //     gui.imgui.render(), queue, device, &mut display_pass
            // ).expect("failed to render gui");
        }
        queue.submit(Some(encoder.finish()));
        output.present();
    }
}

