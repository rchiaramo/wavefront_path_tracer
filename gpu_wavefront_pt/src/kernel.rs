use std::borrow::Cow;
use std::fmt::format;
use std::{env, fs};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderModuleDescriptor, ShaderSource, ShaderStages};
use wavefront_common::gpu_buffer::GPUBuffer;
use wavefront_common::wgpu_state::WgpuState;
use crate::query_gpu::{Queries, QueryResults};

pub struct Kernel {
    wgpu_state: Rc<WgpuState>,
    bind_groups: Vec<BindGroup>,
    pipeline: ComputePipeline,
    timing_query: Queries,
    query_results: QueryResults,
}

impl Kernel {
    // on initialization, a kernel needs to:
    // create bind group layout and bind group
    // load a shader
    // create a pipeline layout and a pipeline
    pub fn new(name: &str,
               wgpu_state: Rc<WgpuState>,
               buffers: Vec<Vec<&GPUBuffer>>,
               read_state: Vec<Vec<bool>>) -> Self {

        let device = wgpu_state.device();
        let path = format!("gpu_wavefront_pt/shaders/{name}.wgsl");

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(fs::read_to_string(path).unwrap().as_str())),
        });

        let num_bind_groups = buffers.len();
        let mut bind_group_layouts = Vec::<BindGroupLayout>::with_capacity(num_bind_groups);
        let mut bind_groups = Vec::<BindGroup>::with_capacity(num_bind_groups);

        for i in 0..num_bind_groups {
            let num_entries = buffers[i].len();
            let mut layout_entries = Vec::<BindGroupLayoutEntry>::with_capacity(num_entries);
            let mut binding_entries = <Vec::<BindGroupEntry>>::with_capacity(num_entries);
            for j in 0..num_entries {
                layout_entries.push(buffers[i][j].layout(ShaderStages::COMPUTE, j as u32, read_state[i][j]));
                binding_entries.push(buffers[i][j].binding(j as u32));
            }
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor{
                label: None,
                entries: layout_entries.as_slice(),
            });
            bind_groups.push(device.create_bind_group(&BindGroupDescriptor{
                label: None,
                layout: &bind_group_layout,
                entries: &binding_entries,
            }));
            bind_group_layouts.push(bind_group_layout);
        }

        let mut bind_group_layout_refs = Vec::<&BindGroupLayout>::with_capacity(num_bind_groups);
        for i in 0..num_bind_groups {
            bind_group_layout_refs.push(&bind_group_layouts[i]);
        }

        // create the pipeline
        let label = format!("{name} compute shader pipeline");
        let pipeline_layout = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: Some(label.as_str()),
            bind_group_layouts: bind_group_layout_refs.as_slice(),
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some(label.as_str()),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });


        Self {
            wgpu_state: Rc::clone(&wgpu_state),
            bind_groups,
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

    pub fn run(&mut self, workgroup_size: (u32, u32)) {
        let device = self.wgpu_state.device();
        let queue = self.wgpu_state.queue();
        self.timing_query.next_unused_query = 0;

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("generate ray kernel encoder"),
            });

        encoder.write_timestamp(&self.timing_query.set, self.timing_query.next_unused_query);
        self.timing_query.next_unused_query += 1;
        {
            let mut generate_rays_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate rays pass"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &self.timing_query.set,
                    beginning_of_pass_write_index: Some(self.timing_query.next_unused_query),
                    end_of_pass_write_index: Some(self.timing_query.next_unused_query + 1),
                })
            });
            self.timing_query.next_unused_query += 2;
            generate_rays_pass.set_pipeline(&self.pipeline);
            for (i, group) in self.bind_groups.iter().enumerate() {
                generate_rays_pass.set_bind_group(i as u32, &group, &[]);
            }
            generate_rays_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);

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

