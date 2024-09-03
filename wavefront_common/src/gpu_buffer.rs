use wgpu::{BindGroupEntry, BindGroupLayoutEntry, BindingType, Buffer, BufferAddress, BufferBindingType, BufferUsages, Device, Queue, ShaderStages};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub struct GPUBuffer {
    name: Buffer,
    usage: BufferUsages,
    binding_idx: u32,
}

impl GPUBuffer {
    pub fn new(device: &Device, usage: BufferUsages, size: BufferAddress, binding_idx: u32, label: Option<&str>)
               -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage: usage | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            name: buffer,
            usage,
            binding_idx
        }
    }

    pub fn new_from_bytes(device: &Device,
                          usage: BufferUsages,
                          binding_idx: u32,
                          data: &[u8],
                          label: Option<&str>) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label,
            contents: data,
            usage: usage | BufferUsages::COPY_DST,
        });
        Self {
            name: buffer,
            usage,
            binding_idx
        }
    }

    pub fn name(&self) -> &Buffer {
        &self.name
    }

    pub fn queue_for_gpu(&mut self, queue: &Queue, data: &[u8]) {
        queue.write_buffer(&self.name, 0, data);
    }


    pub fn layout(&self, visibility: ShaderStages, read_only: bool) -> BindGroupLayoutEntry {
        let mut buffer_binding_type: BufferBindingType = Default::default();
        match self.usage {
            BufferUsages::STORAGE => {
                buffer_binding_type = BufferBindingType::Storage { read_only };
            }
            BufferUsages::UNIFORM => {
                buffer_binding_type = BufferBindingType::Uniform;
            }
            _ => {}
        }
        BindGroupLayoutEntry {
            binding: self.binding_idx,
            visibility,
            ty: BindingType::Buffer {
                ty: buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn binding(&self) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding: self.binding_idx,
            resource: self.name.as_entire_binding(),
        }
    }
}