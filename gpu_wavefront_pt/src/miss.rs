use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use crate::query_gpu::Queries;

pub struct MissKernel {
    miss_buffer_bind_group: BindGroup,
    pipeline: ComputePipeline
}

impl MissKernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(device: &Device,
               image_buffer: &GPUBuffer,
               ray_buffer: &GPUBuffer,
               miss_buffer: &GPUBuffer) -> Self {
        // load the kernel
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/miss_kernel.wgsl"));

        // create the bind group
        let miss_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("miss buffer bind group layout"),
                entries: &[image_buffer.layout(ShaderStages::COMPUTE, 0, false),
                    ray_buffer.layout(ShaderStages::COMPUTE, 1, true),
                    miss_buffer.layout(ShaderStages::COMPUTE, 2,false)
                ],
            });
        let miss_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("ray buffer bind group"),
            layout: &miss_buffer_bind_group_layout,
            entries: &[image_buffer.binding(0),
                ray_buffer.binding(1),
                miss_buffer.binding(2)],
        });

        // create the pipeline
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("miss shader pipeline layout"),
                bind_group_layouts: &[
                    &miss_buffer_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("miss shader pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }
        );

        Self {
            miss_buffer_bind_group,
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

    pub fn run(&self, device: &Device, queue: &Queue, workgroup_size: (u32, u32), mut _queries: Queries) {

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("miss kernel encoder"),
            });

        {
            let mut miss_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("miss pass"),
                timestamp_writes: None
                // timestamp_writes: Some(ComputePassTimestampWrites {
                //     query_set: &queries.set,
                //     beginning_of_pass_write_index: Some(queries.next_unused_query),
                //     end_of_pass_write_index: Some(queries.next_unused_query + 1),
                // })
            });
            // queries.next_unused_query += 2;
            miss_pass.set_pipeline(&self.pipeline);
            miss_pass.set_bind_group(0, &self.miss_buffer_bind_group, &[]);
            miss_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

        }
        // queries.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

}

