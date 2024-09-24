use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;
use wgpu::Surface;
use winit::window::Window;
use crate::gpu_buffer::GPUBuffer;

pub struct WgpuState {
    pub surface: RefCell<Surface<'static>>,
    pub surface_config: RefCell<wgpu::SurfaceConfiguration>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuState {
    pub fn new(window: Arc<Window>) -> WgpuState {
        pollster::block_on(WgpuState::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> WgpuState {
        let size = {
            let viewport = window.inner_size();
            (viewport.width, viewport.height)
        };

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            }
        );

        let surface: wgpu::Surface<'_> = instance.create_surface(
            Arc::clone(&window)).expect("Failed to create surface");

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await.expect("Failed to find an appropriate adapter");

        let features = adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        // if features.contains(wgpu::Features::BGRA8UNORM_STORAGE)
        // {
        //     println!("Adapter has bgra8unorm storage");
        // } else {
        //     println!("Adapter does not have this storage");
        // }
        // if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        //     println!("Adapter supports timestamp queries.");
        // } else {
        //     println!("Adapter does not support timestamp queries, aborting.");
        // }
        // let timestamps_inside_passes = features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        // if timestamps_inside_passes {
        //     println!("Adapter supports timestamp queries within encoders.");
        // } else {
        //     println!("Adapter does not support timestamp queries within encoders.");
        // }
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1024u32 << 20,
                    max_buffer_size: 1_u64 << 35,
                    ..Default::default()
                },
                label: None,
                memory_hints: Default::default(),
            },
            None,
        ).await.expect("Failed to create device");

        let surface_capabilities = surface.get_capabilities(&adapter);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: size.0,
            height: size.1,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };

        Self {
            surface: RefCell::new(surface),
            surface_config: RefCell::new(surface_config),
            device,
            queue,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize(&self, new_size: (u32, u32))
    {
        {
            let mut surf_conf = self.surface_config.borrow_mut();
            surf_conf.width = new_size.0;
            surf_conf.height = new_size.1;
        }
        let mut surface = self.surface.borrow_mut();
        surface.configure(&self.device, &self.surface_config.borrow());
    }

    pub fn copy_buffer_to_buffer(&self, from_buffer: &GPUBuffer, to_buffer: &GPUBuffer) {
        let device = self.device();
        let queue = self.queue();
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("swap encoder"),
            });
        encoder.clear_buffer(to_buffer.name(), 0, None);
        encoder.copy_buffer_to_buffer(from_buffer.name(),
                                      0,
                                      to_buffer.name(),
                                      0,
                                      to_buffer.name().size());
        encoder.clear_buffer(from_buffer.name(), 0, None);
        queue.submit(Some(encoder.finish()));
    }

    pub fn read_buffer(&self, from_buffer: &GPUBuffer) -> Vec::<u32> {
        from_buffer.name()
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.device().poll(wgpu::Maintain::wait()).panic_on_timeout();

        let counter: Vec<u32> = {
            let counter_view = from_buffer.name()
                .slice(..)
                .get_mapped_range();
            bytemuck::cast_slice(&counter_view).to_vec()
        };
        from_buffer.name().unmap();

        counter
    }
}