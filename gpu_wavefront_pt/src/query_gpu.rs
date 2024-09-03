use std::collections::VecDeque;
use wgpu::Queue;

pub struct Queries {
    pub set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    num_queries: u64,
    pub next_unused_query: u32,
}

#[derive(Default)]
pub struct QueryResults {
    compute_start_end_timestamps: [u64; 2],
    running_avg: VecDeque<f64>
}

impl QueryResults {
    // Queries:
    // * compute start
    // * compute end
    pub const NUM_QUERIES: u64 = 2;
    pub const RUNNING_AVG_LENGTH: usize = 10;

    pub fn new() -> Self {
        Self {
            compute_start_end_timestamps: [0u64; 2],
            running_avg: VecDeque::<f64>::with_capacity(Self::RUNNING_AVG_LENGTH)
        }
    }
    #[allow(clippy::redundant_closure)] // False positive
    pub fn process_raw_results(&mut self, queue: &Queue, timestamps: Vec<u64>) {
        assert_eq!(timestamps.len(), Self::NUM_QUERIES as usize);

        let period = queue.get_timestamp_period();
        let elapsed_us = |start, end: u64| { end.wrapping_sub(start) as f64 * period as f64 / (1000.0) };

        if self.running_avg.len() == Self::RUNNING_AVG_LENGTH {
            self.running_avg.pop_back();
        }
        self.running_avg.push_front(elapsed_us(timestamps[0], timestamps[1]));
    }

    pub fn get_running_avg(&self) -> f32 {
        let sum: f64 = self.running_avg.iter().sum();
        (sum / self.running_avg.len() as f64) as f32
    }

    #[cfg_attr(test, allow(unused))]
    pub fn print(&self, queue: &wgpu::Queue) {
        let period = queue.get_timestamp_period();
        let elapsed_ms = |start, end: u64| { end.wrapping_sub(start) as f64 * period as f64 / (1000000.0) };


        println!(
            "Elapsed time compute pass: {:.2} ms",
            elapsed_ms(
                self.compute_start_end_timestamps[0],
                self.compute_start_end_timestamps[1]
            ) as f32
        );
    }
}

impl Queries {
    pub(crate) fn new(device: &wgpu::Device, num_queries: u64) -> Self {
        Queries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: num_queries as _,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size: size_of::<u64>() as u64 * num_queries,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size: size_of::<u64>() as u64 * num_queries,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            num_queries,
            next_unused_query: 0,
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.set,
            // TODO(https://github.com/gfx-rs/wgpu/issues/3993): Musn't be larger than the number valid queries in the set.
            0..self.next_unused_query,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn wait_for_results(&self, device: &wgpu::Device) -> Vec<u64> {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(..(std::mem::size_of::<u64>() as wgpu::BufferAddress * self.num_queries))
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        timestamps
    }
}