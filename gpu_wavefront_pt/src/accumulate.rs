use std::rc::Rc;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::wgpu_state::WgpuState;
use crate::query_gpu::{Queries, QueryResults};

pub struct AccumulateKernel {
    wgpu_state: Rc<WgpuState>,
    buffer_bind_group: BindGroup,
    pipeline: ComputePipeline,
    timing_query: Queries
}

impl AccumulateKernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(wgpu_state: Rc<WgpuState>,
               image_buffer: &GPUBuffer,
               accumultated_image_buffer: &GPUBuffer) -> Self {
        // load the kernel
        let device = wgpu_state.device();
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/accumulate.wgsl"));

        // create the bind group
        let buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("accumulate buffer bind group layout"),
                entries: &[image_buffer.layout(ShaderStages::COMPUTE, 0, false),
                    accumultated_image_buffer.layout(ShaderStages::COMPUTE, 1, false),
                ],
            });
        let buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("accumulate buffer bind group"),
            layout: &buffer_bind_group_layout,
            entries: &[image_buffer.binding(0),
                accumultated_image_buffer.binding(1)],
        });

        // create the pipeline
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("accumulate kernel pipeline layout"),
                bind_group_layouts: &[
                    &buffer_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("accumulate shader pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }
        );

        Self {
            wgpu_state: Rc::clone(&wgpu_state),
            buffer_bind_group,
            pipeline,
            timing_query: Queries::new(device, QueryResults::NUM_QUERIES)
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

    pub fn run(&self, workgroup_size: (u32, u32)) {

        let device = self.wgpu_state.device();
        let queue = self.wgpu_state.queue();
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("accumulate kernel encoder"),
            });

        {
            let mut miss_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accumulate pass"),
                timestamp_writes: None
                // timestamp_writes: Some(ComputePassTimestampWrites {
                //     query_set: &queries.set,
                //     beginning_of_pass_write_index: Some(queries.next_unused_query),
                //     end_of_pass_write_index: Some(queries.next_unused_query + 1),
                // })
            });
            // queries.next_unused_query += 2;
            miss_pass.set_pipeline(&self.pipeline);
            miss_pass.set_bind_group(0, &self.buffer_bind_group, &[]);
            miss_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

        }
        // queries.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

}

