use std::sync::Arc;
use winit::window::Window;

pub struct WgpuState<'a> {
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<'a> WgpuState<'a> {
    pub fn new(window: Arc<Window>) -> WgpuState<'a> {
        pollster::block_on(WgpuState::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> WgpuState<'a> {
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

        let surface = instance.create_surface(
            Arc::clone(&window)).expect("Failed to create surface");

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await.expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1024_u32 << 20,
                    max_buffer_size: 1024_u64 << 20,
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
            surface,
            surface_config,
            device,
            queue,
        }
    }

    pub fn surface_config(&self) -> wgpu::SurfaceConfiguration {
        self.surface_config.clone()
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub fn surface(&self) -> &wgpu::Surface {
        &self.surface
    }

    pub fn resize(&mut self, new_size: (u32, u32))
    {
        self.surface_config.width = new_size.0;
        self.surface_config.height = new_size.1;
        self.surface.configure(&self.device, &self.surface_config);
    }
}