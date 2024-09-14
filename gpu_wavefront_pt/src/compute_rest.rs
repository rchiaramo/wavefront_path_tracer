use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use crate::query_gpu::Queries;

pub struct ComputeRestKernel {
    image_buffer_bind_group: BindGroup,
    scene_buffer_bind_group: BindGroup,
    parameters_buffer_bind_group: BindGroup,
    pipeline: ComputePipeline
}

impl ComputeRestKernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(device: &Device,
               image_buffer: &GPUBuffer,
               frame_buffer: &GPUBuffer,
               ray_buffer: &GPUBuffer,
               sphere_buffer: &GPUBuffer,
               material_buffer: &GPUBuffer,
               bvh_buffer: &GPUBuffer,
               camera_buffer: &GPUBuffer,
               sampling_parameters_buffer: &GPUBuffer) -> Self {
        // load the kernel
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/compute_rest.wgsl"));

        // create the various bind groups
        // group image, frame, and ray buffers into image bind group

        let image_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("image buffer bind group layout"),
                entries: &[image_buffer.layout(ShaderStages::COMPUTE, 0, false),
                    frame_buffer.layout(ShaderStages::COMPUTE, 1,false),
                    ray_buffer.layout(ShaderStages::COMPUTE, 2, true)
                ],
            });

        let image_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("image buffer bind group"),
            layout: &image_buffer_bind_group_layout,
            entries: &[image_buffer.binding(0),
                frame_buffer.binding(1),
                ray_buffer.binding(2)
            ],
        });

        // the scene bind group will hold the primitives, the materials, and the bvh_tree
        let scene_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("scene buffer bind group layout"),
                entries: &[sphere_buffer.layout(ShaderStages::COMPUTE, 0,true),
                    material_buffer.layout(ShaderStages::COMPUTE, 1,  true),
                    bvh_buffer.layout(ShaderStages::COMPUTE, 2,true)],
            });

        let scene_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("scene buffer bind group"),
            layout: &scene_buffer_bind_group_layout,
            entries: &[sphere_buffer.binding(0),
                material_buffer.binding(1),
                bvh_buffer.binding(2)],
        });

        // put everything else into parameters buffer bind group
        let parameters_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("parameters buffer bind group layout"),
                entries: &[
                    camera_buffer.layout(ShaderStages::COMPUTE, 0, true),
                    sampling_parameters_buffer.layout(ShaderStages::COMPUTE, 1, true),
                ],
            });

        let parameters_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("parameters bind group"),
            layout: &parameters_buffer_bind_group_layout,
            entries: &[
                camera_buffer.binding(0),
                sampling_parameters_buffer.binding(1),
            ],
        });

        // create the pipeline
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute rest shader pipeline layout"),
                bind_group_layouts: &[
                    &image_buffer_bind_group_layout,
                    &scene_buffer_bind_group_layout,
                    &parameters_buffer_bind_group_layout
                ],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("compute rest shader pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }
        );

        Self {
            image_buffer_bind_group,
            scene_buffer_bind_group,
            parameters_buffer_bind_group,
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
    pub fn run(&self, device: &Device, queue: &Queue, workgroup_size: (u32, u32), mut queries: Queries) {

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("compute rest kernel encoder"),
            });

        {
            let mut compute_rest_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute rest pass"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &queries.set,
                    beginning_of_pass_write_index: Some(queries.next_unused_query),
                    end_of_pass_write_index: Some(queries.next_unused_query + 1),
                })
            });
            queries.next_unused_query += 2;
            compute_rest_pass.set_pipeline(&self.pipeline);
            compute_rest_pass.set_bind_group(0, &self.image_buffer_bind_group, &[]);
            compute_rest_pass.set_bind_group(1, &self.scene_buffer_bind_group, &[]);
            compute_rest_pass.set_bind_group(2, &self.parameters_buffer_bind_group, &[]);
            compute_rest_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

        }
        queries.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

}

