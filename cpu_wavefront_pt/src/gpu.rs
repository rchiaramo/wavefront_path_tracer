use std::cell::{Ref, RefCell, RefMut};
use crate::ray::Ray;

pub struct GPU {
    ray_buffer: RefCell<Vec<Ray>>
}

impl GPU {
    pub fn new(max_num_pixels: usize) -> GPU {
        let ray_buffer = RefCell::new(Vec::<Ray>::with_capacity(max_num_pixels));
        Self { ray_buffer }
    }

    pub fn ray_buffer_mut(&self) -> RefMut<'_, Vec<Ray>> {
        self.ray_buffer.borrow_mut()
    }

    pub fn ray_buffer(&self) -> Ref<Vec<Ray>> {
        self.ray_buffer.borrow()
    }
}