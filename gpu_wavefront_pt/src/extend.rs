use std::rc::Rc;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::wgpu_state::WgpuState;
use crate::query_gpu::{Queries, QueryResults};

pub struct ExtendKernel {
    wgpu_state: Rc<WgpuState>,
    extend_shader_bind_group: BindGroup,
    scene_buffer_bind_group: BindGroup,
    pipeline: ComputePipeline,
    timing_query: Queries,
}

impl ExtendKernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(wgpu_state: Rc<WgpuState>,
               frame_buffer: &GPUBuffer,
               ray_buffer: &GPUBuffer,
               miss_buffer: &GPUBuffer,
               hit_buffer: &GPUBuffer,
               counter_buffer: &GPUBuffer,
               sphere_buffer: &GPUBuffer,
               bvh_buffer: &GPUBuffer) -> Self {
        // load the kernel
        let device = wgpu_state.device();
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/extend.wgsl"));

        // create the various bind groups
        // group frame, ray, hit, miss, and counter buffers together

        let extend_shader_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("extend shader bind group layout"),
                entries: &[
                    ray_buffer.layout(ShaderStages::COMPUTE, 0, true),
                    counter_buffer.layout(ShaderStages::COMPUTE, 1, false),
                    miss_buffer.layout(ShaderStages::COMPUTE, 2, false),
                    hit_buffer.layout(ShaderStages::COMPUTE, 3, false),
                ],
            });

        let extend_shader_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("extend shader bind group"),
            layout: &extend_shader_bind_group_layout,
            entries: &[
                ray_buffer.binding(0),
                counter_buffer.binding(1),
                miss_buffer.binding(2),
                hit_buffer.binding(3),
            ],
        });

        // the scene bind group will hold the primitives and the bvh_tree
        let scene_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("scene buffer bind group layout"),
                entries: &[sphere_buffer.layout(ShaderStages::COMPUTE, 0,true),
                    bvh_buffer.layout(ShaderStages::COMPUTE, 1,true)],
            });

        let scene_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("scene buffer bind group"),
            layout: &scene_buffer_bind_group_layout,
            entries: &[sphere_buffer.binding(0),
                bvh_buffer.binding(1)],
        });

        // create the pipeline
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("extend shader pipeline layout"),
                bind_group_layouts: &[
                    &extend_shader_bind_group_layout,
                    &scene_buffer_bind_group_layout
                ],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("extend shader pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }
        );

        Self {
            wgpu_state: Rc::clone(&wgpu_state),
            extend_shader_bind_group,
            scene_buffer_bind_group,
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
    pub fn run(&self,
               workgroup_size: (u32, u32),
               counter_buffer: &GPUBuffer, read_buffer: &GPUBuffer) {

        let device = self.wgpu_state.device();
        let queue = self.wgpu_state.queue();

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("extend kernel encoder"),
            });

        {
            let mut extend_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute rest pass"),
                timestamp_writes: None
                // timestamp_writes: Some(ComputePassTimestampWrites {
                //     query_set: &queries.set,
                //     beginning_of_pass_write_index: Some(queries.next_unused_query),
                //     end_of_pass_write_index: Some(queries.next_unused_query + 1),
                // })
            });
            // queries.next_unused_query += 2;
            extend_pass.set_pipeline(&self.pipeline);
            extend_pass.set_bind_group(0, &self.extend_shader_bind_group, &[]);
            extend_pass.set_bind_group(1, &self.scene_buffer_bind_group, &[]);
            extend_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

        }
        // queries.resolve(&mut encoder);
        encoder.copy_buffer_to_buffer(counter_buffer.name(), 0,
                                      read_buffer.name(), 0,
                                      read_buffer.name().size());
        queue.submit(Some(encoder.finish()));
    }

}

