use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;
use wgpu::Surface;
use winit::window::Window;

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

        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
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
        let mut x = self.surface_config.borrow_mut().width;
        x = new_size.0;
        let mut y = self.surface_config.borrow_mut().height;
        y = new_size.1;
        self.surface.borrow_mut().configure(&self.device, &*self.surface_config.borrow());
    }
}