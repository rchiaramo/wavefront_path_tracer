use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use crate::query_gpu::Queries;

pub struct GenerateRayKernel {
    ray_buffer_bind_group: BindGroup,
    parameters_buffer_bind_group: BindGroup,
    pipeline: ComputePipeline
}

impl GenerateRayKernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(device: &Device,
               ray_buffer: &GPUBuffer,
               frame_buffer: &GPUBuffer,
               camera_buffer: &GPUBuffer,
               proj_matrix_buffer: &GPUBuffer,
               view_mat_buffer: &GPUBuffer) -> Self {
        // load the kernel
        let shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/generate_rays.wgsl"));

        // create the various bind groups
        let ray_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("ray buffer bind group layout"),
                entries: &[ray_buffer.layout(ShaderStages::COMPUTE, 0,false)
                ],
            });
        let ray_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("ray buffer bind group"),
            layout: &ray_buffer_bind_group_layout,
            entries: &[ray_buffer.binding(0)],
        });

        let parameters_buffer_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor{
                label: Some("parameters buffer bind group layout"),
                entries: &[frame_buffer.layout(ShaderStages::COMPUTE, 0, true),
                    camera_buffer.layout(ShaderStages::COMPUTE, 1, true),
                    proj_matrix_buffer.layout(ShaderStages::COMPUTE, 2, true),
                    view_mat_buffer.layout(ShaderStages::COMPUTE, 3, true),
                ],
            });

        let parameters_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
            label: Some("parameters bind group"),
            layout: &parameters_buffer_bind_group_layout,
            entries: &[frame_buffer.binding(0),
                camera_buffer.binding(1),
                proj_matrix_buffer.binding(2),
                view_mat_buffer.binding(3)
            ],
        });

        // create the pipeline
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute shader pipeline layout"),
                bind_group_layouts: &[
                    &ray_buffer_bind_group_layout,
                    &parameters_buffer_bind_group_layout
                ],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("generate rays shader pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }
        );

        Self {
            ray_buffer_bind_group,
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
                label: Some("generate ray kernel encoder"),
            });


        {
            let mut generate_rays_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate rays pass"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &queries.set,
                    beginning_of_pass_write_index: Some(queries.next_unused_query),
                    end_of_pass_write_index: Some(queries.next_unused_query + 1),
                })
            });
            queries.next_unused_query += 2;
            generate_rays_pass.set_pipeline(&self.pipeline);
            generate_rays_pass.set_bind_group(0, &self.ray_buffer_bind_group, &[]);
            generate_rays_pass.set_bind_group(2, &self.parameters_buffer_bind_group, &[]);
            generate_rays_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

        }
        queries.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

}

