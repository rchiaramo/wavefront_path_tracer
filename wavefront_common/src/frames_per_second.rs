use std::collections::VecDeque;
use std::time::Duration;

pub struct FramesPerSecond {
    time_history: VecDeque<f32>
}

impl FramesPerSecond {
    pub const RUNNING_AVG_LENGTH: usize = 10;

    pub fn new() -> Self {
        Self {
            time_history: VecDeque::<f32>::with_capacity(Self::RUNNING_AVG_LENGTH)
        }
    }

    pub fn update(&mut self, dt: Duration){
        self.time_history.push_front(dt.as_secs_f32());
        if self.time_history.len() > Self::RUNNING_AVG_LENGTH {
            self.time_history.pop_back();
        }
    }

    pub fn get_avg_fps(&self) -> f32 {
        let sum: f32 = self.time_history.iter().sum();
        self.time_history.len() as f32 / sum
    }
}