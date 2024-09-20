use std::rc::Rc;
use wgpu::{BindGroupEntry, BindGroupLayoutEntry, BindingType, Buffer, BufferAddress, BufferBindingType, BufferUsages, Device, Queue, ShaderStages};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use crate::wgpu_state::WgpuState;

pub struct GPUBuffer {
    name: Buffer,
    usage: BufferUsages,
    wgpu_state: Rc<WgpuState>
}


impl GPUBuffer {
    pub fn new(wgpu_state: Rc<WgpuState>, usage: BufferUsages, size: BufferAddress, label: Option<&str>)
               -> Self {
        let device = wgpu_state.device();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            name: buffer,
            usage,
            wgpu_state
        }
    }

    pub fn new_from_bytes(wgpu_state: Rc<WgpuState>,
                          usage: BufferUsages,
                          data: &[u8],
                          label: Option<&str>) -> Self {
        let device = wgpu_state.device();
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label,
            contents: data,
            usage,
        });
        Self {
            name: buffer,
            usage,
            wgpu_state
        }
    }

    pub fn name(&self) -> &Buffer {
        &self.name
    }

    pub fn queue_for_gpu(&mut self, data: &[u8]) {
        let queue = self.wgpu_state.queue();
        queue.write_buffer(&self.name, 0, data);
    }


    pub fn layout(&self, visibility: ShaderStages, binding_idx: u32, read_only: bool) -> BindGroupLayoutEntry {
        let mut buffer_binding_type: BufferBindingType = Default::default();
        if self.usage.contains(BufferUsages::STORAGE) {
            buffer_binding_type = BufferBindingType::Storage { read_only };
        } else if self.usage.contains(BufferUsages::UNIFORM) {
            buffer_binding_type = BufferBindingType::Uniform;
        }

        BindGroupLayoutEntry {
            binding: binding_idx,
            visibility,
            ty: BindingType::Buffer {
                ty: buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn binding(&self, binding_idx: u32) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding: binding_idx,
            resource: self.name.as_entire_binding(),
        }
    }
}