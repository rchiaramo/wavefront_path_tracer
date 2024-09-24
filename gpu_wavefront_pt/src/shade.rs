use std::rc::Rc;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::wgpu_state::WgpuState;
use crate::query_gpu::{Queries, QueryResults};

pub struct ShadeKernel {
    wgpu_state: Rc<WgpuState>,
    image_buffer_bind_group: BindGroup,
    scene_buffer_bind_group: BindGroup,
    pipeline: ComputePipeline,
    timing_query: Queries,
    query_results: QueryResults,
}

impl ShadeKernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(wgpu_state: Rc<WgpuState>,
               image_buffer: &GPUBuffer,
               frame_buffer: &GPUBuffer,
               ray_buffer: &GPUBuffer,
               extension_ray_buffer: &GPUBuffer,
               hit_buffer: &GPUBuffer,
               counter_buffer: &GPUBuffer,
               sphere_buffer: &GPUBuffer,
               material_buffer: &GPUBuffer) -> Self {
        // load the kernel
        let device = wgpu_state.device();
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/shade.wgsl"));

        // create the various bind groups
        // group image, frame, and ray buffers into image bind group

        let image_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("image buffer bind group layout"),
                entries: &[image_buffer.layout(ShaderStages::COMPUTE, 0, false),
                    frame_buffer.layout(ShaderStages::COMPUTE, 1,false),
                    ray_buffer.layout(ShaderStages::COMPUTE, 2, false),
                    extension_ray_buffer.layout(ShaderStages::COMPUTE, 3, false),
                    hit_buffer.layout(ShaderStages::COMPUTE, 4, false),
                    counter_buffer.layout(ShaderStages::COMPUTE, 5, false)
                ],
            });

        let image_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("image buffer bind group"),
            layout: &image_buffer_bind_group_layout,
            entries: &[image_buffer.binding(0),
                frame_buffer.binding(1),
                ray_buffer.binding(2),
                extension_ray_buffer.binding(3),
                hit_buffer.binding(4),
                counter_buffer.binding(5)
            ],
        });

        // the scene bind group will hold the primitives, the materials, and the bvh_tree
        let scene_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("scene buffer bind group layout"),
                entries: &[sphere_buffer.layout(ShaderStages::COMPUTE, 0,true),
                    material_buffer.layout(ShaderStages::COMPUTE, 1,  true)],
            });

        let scene_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("scene buffer bind group"),
            layout: &scene_buffer_bind_group_layout,
            entries: &[sphere_buffer.binding(0),
                material_buffer.binding(1)
                ]
        });

        // create the pipeline
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("shade kernel shader pipeline layout"),
                bind_group_layouts: &[
                    &image_buffer_bind_group_layout,
                    &scene_buffer_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("shade kernel pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }
        );

        Self {
            wgpu_state: Rc::clone(&wgpu_state),
            image_buffer_bind_group,
            scene_buffer_bind_group,
            pipeline,
            timing_query: Queries::new(device, QueryResults::NUM_QUERIES),
            query_results: QueryResults::new()
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
    pub fn run(&mut self,
               workgroup_size: (u32, u32)) {

        let device = self.wgpu_state.device();
        let queue = self.wgpu_state.queue();
        self.timing_query.next_unused_query = 0;

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("compute rest kernel encoder"),
            });

        encoder.write_timestamp(&self.timing_query.set, self.timing_query.next_unused_query);
        self.timing_query.next_unused_query += 1;
        {
            let mut compute_rest_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute rest pass"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &self.timing_query.set,
                    beginning_of_pass_write_index: Some(self.timing_query.next_unused_query),
                    end_of_pass_write_index: Some(self.timing_query.next_unused_query + 1),
                })
            });
            self.timing_query.next_unused_query += 2;
            compute_rest_pass.set_pipeline(&self.pipeline);
            compute_rest_pass.set_bind_group(0, &self.image_buffer_bind_group, &[]);
            compute_rest_pass.set_bind_group(1, &self.scene_buffer_bind_group, &[]);
            compute_rest_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

        }
        encoder.write_timestamp(&self.timing_query.set, self.timing_query.next_unused_query);
        self.timing_query.next_unused_query += 1;
        self.timing_query.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    pub fn get_timing(&mut self) -> f32 {
        self.query_results.process_raw_results(&self.wgpu_state.queue(),
                                               self.timing_query.wait_for_results(&self.wgpu_state.device()));
        self.query_results.get_running_avg()
    }
}

