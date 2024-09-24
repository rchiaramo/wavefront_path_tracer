// use std::rc::Rc;
// use wgpu::{BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, ComputePassTimestampWrites, ComputePipeline, Device, Queue, ShaderStages};
// use wavefront_common::gpu_buffer::GPUBuffer;
// use wavefront_common::wgpu_state::WgpuState;
// use crate::query_gpu::{Queries, QueryResults};
//
// pub struct Kernel {
//     wgpu_state: Rc<WgpuState>,
//     bind_groups: Vec<BindGroup>,
//     pipeline: ComputePipeline,
//     timing_query: Queries,
//     query_results: QueryResults,
// }
//
// impl Kernel {
//     // on initialization, a kernel needs to:
//     // create bind group layout and bind group
//     // load a shader
//     // create a pipeline layout and a pipeline
//     pub fn new(name: &str,
//                wgpu_state: Rc<WgpuState>,
//                buffer_list: Vec<&GPUBuffer>,
//                lay_list: Vec<(usize, usize, bool)>
//     ) -> Self {
//         // load the kernel
//         let device = wgpu_state.device();
//         let shader = device.create_shader_module(
//             wgpu::include_wgsl!(&format!("../shaders/{name}.wgsl")[..]));
//
//         let bind_group_layouts = self.make_bind_group_layouts(buffer_list, lay_list);
//         let bind_groups = self.make_bind_groups(&device, buffer_list, lay_list);
//
//         // create the various bind groups
//         let ray_buffer_bind_group_layout = device.create_bind_group_layout(
//             &BindGroupLayoutDescriptor{
//                 label: Some("ray buffer bind group layout"),
//                 entries: &[ray_buffer.layout(ShaderStages::COMPUTE, 0,false)
//                 ],
//             });
//         let ray_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
//             label: Some("ray buffer bind group"),
//             layout: &ray_buffer_bind_group_layout,
//             entries: &[ray_buffer.binding(0)],
//         });
//
//         let parameters_buffer_bind_group_layout = device.create_bind_group_layout(
//             &BindGroupLayoutDescriptor{
//                 label: Some("parameters buffer bind group layout"),
//                 entries: &[frame_buffer.layout(ShaderStages::COMPUTE, 0, true),
//                     camera_buffer.layout(ShaderStages::COMPUTE, 1, true),
//                     proj_matrix_buffer.layout(ShaderStages::COMPUTE, 2, true),
//                     view_mat_buffer.layout(ShaderStages::COMPUTE, 3, true),
//                 ],
//             });
//
//         let parameters_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor{
//             label: Some("parameters bind group"),
//             layout: &parameters_buffer_bind_group_layout,
//             entries: &[frame_buffer.binding(0),
//                 camera_buffer.binding(1),
//                 proj_matrix_buffer.binding(2),
//                 view_mat_buffer.binding(3)
//             ],
//         });
//
//         // create the pipeline
//         let pipeline_layout = device.create_pipeline_layout(
//             &wgpu::PipelineLayoutDescriptor {
//                 label: Some("compute shader pipeline layout"),
//                 bind_group_layouts: &[
//                     &ray_buffer_bind_group_layout,
//                     &parameters_buffer_bind_group_layout
//                 ],
//                 push_constant_ranges: &[],
//             }
//         );
//
//         let pipeline = device.create_compute_pipeline(
//             &wgpu::ComputePipelineDescriptor {
//                 label: Some("generate rays shader pipeline"),
//                 layout: Some(&pipeline_layout),
//                 module: &shader,
//                 entry_point: "main",
//                 compilation_options: Default::default(),
//                 cache: None,
//             }
//         );
//
//         Self {
//             wgpu_state: Rc::clone(&wgpu_state),
//             ray_buffer_bind_group,
//             parameters_buffer_bind_group,
//             pipeline,
//             timing_query: Queries::new(device, QueryResults::NUM_QUERIES),
//             query_results: QueryResults::new()
//         }
//     }
//
//     fn make_bind_group_layouts(&self, buffer_list: Vec<&GPUBuffer>,
//                                lay_list: Vec<(usize, usize, bool)>) -> &[BindGroupLayout] {
//         let ray_buffer_bind_group_layout = device.create_bind_group_layout(
//             &BindGroupLayoutDescriptor{
//                 label: Some("ray buffer bind group layout"),
//                 entries: &[ray_buffer.layout(ShaderStages::COMPUTE, 0,false)
//                 ],
//             });
//     }
//
//     // when executing, a kernel needs to:
//     // possibly get a view (display kernel)
//     // create an encoder
//     // create a _pass
//     // set the pipeline
//     // set the bind groups
//     // do the version of execute (dispatch workgroups vs draw)
//     // submit the encoder through the queue
//     // possibly present the output (display kernel)
//
//     pub fn run(&mut self, workgroup_size: (u32, u32)) {
//         let device = self.wgpu_state.device();
//         let queue = self.wgpu_state.queue();
//         self.timing_query.next_unused_query = 0;
//
//         let mut encoder = device.create_command_encoder(
//             &wgpu::CommandEncoderDescriptor {
//                 label: Some("generate ray kernel encoder"),
//             });
//
//         encoder.write_timestamp(&self.timing_query.set, self.timing_query.next_unused_query);
//         self.timing_query.next_unused_query += 1;
//         {
//             let mut generate_rays_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//                 label: Some("generate rays pass"),
//                 timestamp_writes: Some(ComputePassTimestampWrites {
//                     query_set: &self.timing_query.set,
//                     beginning_of_pass_write_index: Some(self.timing_query.next_unused_query),
//                     end_of_pass_write_index: Some(self.timing_query.next_unused_query + 1),
//                 })
//             });
//             self.timing_query.next_unused_query += 2;
//             generate_rays_pass.set_pipeline(&self.pipeline);
//             generate_rays_pass.set_bind_group(0, &self.ray_buffer_bind_group, &[]);
//             generate_rays_pass.set_bind_group(1, &self.parameters_buffer_bind_group, &[]);
//             generate_rays_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);
//
//         }
//         encoder.write_timestamp(&self.timing_query.set, self.timing_query.next_unused_query);
//         self.timing_query.next_unused_query += 1;
//         self.timing_query.resolve(&mut encoder);
//         queue.submit(Some(encoder.finish()));
//     }
//
//     pub fn get_timing(&mut self) -> f32 {
//         self.query_results.process_raw_results(&self.wgpu_state.queue(),
//             self.timing_query.wait_for_results(&self.wgpu_state.device()));
//         self.query_results.get_running_avg()
//     }
// }
//
