use crate::sphere::Sphere;
use glam::{Vec3};

const BINS: usize = 4096;

pub struct Bin {
    aabb_min: Vec3,
    aabb_max: Vec3,
    prim_count: u32,
}

impl Default for Bin {
    fn default() -> Self {
        Self {
            aabb_min: Vec3::INFINITY,
            aabb_max: Vec3::NEG_INFINITY,
            prim_count: 0
        }
    }
}

impl Bin {
    pub fn expand_bin(&mut self, min: Vec3, max: Vec3) {
        self.aabb_min = self.aabb_min.min(min);
        self.aabb_max = self.aabb_max.max(max);
        self.prim_count += 1;
    }

    pub fn get_area(&self) -> f32 {
        if !self.aabb_max.is_finite() {
            return 0.0
        }
        let extent = self.aabb_max - self.aabb_min;
        extent.x * extent.y + extent.y * extent.z + extent.z * extent.x
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BVHNode {
    pub aabb_min: Vec3,
    pub left_first: u32,
    pub aabb_max: Vec3,
    pub prim_count: u32
}

unsafe impl bytemuck::Pod for BVHNode {}
unsafe impl bytemuck::Zeroable for BVHNode {}

impl BVHNode {
    pub fn find_node_cost(&self) -> f32 {
        let extent = self.aabb_max - self.aabb_min;
        let area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;

        self.prim_count as f32 * area
    }

    pub fn update_node_bounds(&mut self, spheres: &[Sphere]) {
        let mut aabb_min = Vec3::INFINITY;
        let mut aabb_max = Vec3::NEG_INFINITY;
        //expand the aabb
        for i in 0 ..self.prim_count as usize { 
            let (sph_min, sph_max) =
                spheres[self.left_first as usize + i].get_aabb();
            aabb_min = aabb_min.min(sph_min);
            aabb_max = aabb_max.max(sph_max);
        }
        self.aabb_min = aabb_min;
        self.aabb_max = aabb_max;
    }

    // this function will return a tuple with (splitCost, bestAxis, planeValue)
    pub fn find_best_split_plane(&self, spheres: &[Sphere])
                                 -> (f32, usize, f32) {

        let extent = self.aabb_max - self.aabb_min;
        let start_idx = self.left_first as usize;
        let mut low_cost = f32::INFINITY;
        let mut best_axis = 0;
        let mut best_plane = 0.0;

        for axes in 0..3 {
            if extent[axes] < 0.00001 {
                continue;
            }

            let mut bins = Vec::<Bin>::with_capacity(BINS);
            for _i in 0..BINS {
                bins.push(Bin::default());
            }

            let scale = BINS as f32 / extent[axes];
            let min_bound = self.aabb_min[axes];
            // for each axis, populate the bins
            for i in 0..self.prim_count as usize {
                let bin_idx = (BINS - 1).min(
                    ((spheres[i + start_idx].center[axes] - min_bound) * scale) as usize);
                let (aabb_min, aabb_max) =
                    spheres[i + start_idx].get_aabb();
                bins[bin_idx].expand_bin(aabb_min, aabb_max);
            }

            // now calculate the cost
            // N bins means N-1 planes as we don't consider the 2 end planes
            // N bins also means N-1 left sided or right sided bins
            let mut left_count = [0u32; BINS - 1];
            let mut right_count = [0u32; BINS - 1];
            let mut left_area = [0.0f32; BINS - 1];
            let mut right_area = [0.0f32; BINS - 1];
            let mut left_sum_bin = Bin::default();
            let mut right_sum_bin = Bin::default();
            for idx in 0..BINS - 1 {
                left_sum_bin.prim_count += bins[idx].prim_count;
                left_count[idx] = left_sum_bin.prim_count;
                right_sum_bin.prim_count += bins[BINS - 1 - idx].prim_count;
                right_count[BINS - 2 - idx] = right_sum_bin.prim_count;

                left_sum_bin.expand_bin(bins[idx].aabb_min, bins[idx].aabb_max);
                left_sum_bin.prim_count -= 1;
                left_area[idx] = left_sum_bin.get_area();
                right_sum_bin.expand_bin(bins[BINS - 1 - idx].aabb_min, bins[BINS - 1 - idx].aabb_max);
                right_sum_bin.prim_count -= 1;
                right_area[BINS - 2 - idx] = right_sum_bin.get_area();
            }

            let scale = 1.0 / BINS as f32;
            for idx in 0..BINS - 1 {
                let cost = left_count[idx] as f32 * left_area[idx] +
                    right_count[idx] as f32 * right_area[idx];
                if cost < low_cost {
                    best_axis = axes;
                    best_plane = min_bound + extent[axes] * scale * (1.0 + idx as f32);
                    low_cost = cost;
                }
            }
        }

        (low_cost, best_axis, best_plane)
    }
}


pub struct BVHTree {
    pub nodes: Vec<BVHNode>,
}

impl BVHTree {
    pub fn new(num_primitives: usize) -> Self {
        Self { nodes: Vec::<BVHNode>::with_capacity(2 * num_primitives) }
    }

    pub fn build_bvh_tree(&mut self, spheres: &mut [Sphere]) {
        let prim_count = spheres.len() as u32;
        let mut node = BVHNode::default();
        node.left_first = 0;
        node.prim_count = prim_count;
        node.update_node_bounds(spheres);
        self.nodes.push(node);

        // push an empty node at index 1 as a placeholder that will never be used
        self.nodes.push(BVHNode::default());

        self.subdivide(0, spheres);
    }

    fn subdivide(&mut self, index: usize, spheres: &mut [Sphere]) {
        let (split_cost, best_axis, plane_val) =
            self.nodes[index].find_best_split_plane(spheres);
        let cost = self.nodes[index].find_node_cost();

        if cost <= split_cost {
            return;
        }

        let mut i = self.nodes[index].left_first as usize;
        let mut j = i + self.nodes[index].prim_count as usize - 1;

        while i <= j {
            if spheres[i].center[best_axis] < plane_val {
                i += 1;
            } else {
                spheres.swap(i,j);
                j -= 1;
            }
        }
        let left_count = i as u32 - self.nodes[index].left_first;
        if left_count == 0 || left_count == self.nodes[index].prim_count {
            return;
        }

        let node_idx = self.nodes.len();
        let mut left_node = BVHNode::default();
        left_node.left_first = self.nodes[index].left_first;
        left_node.prim_count = left_count;
        left_node.update_node_bounds(spheres);

        let mut right_node = BVHNode::default();
        right_node.left_first = i as u32;
        right_node.prim_count = self.nodes[index].prim_count - left_count;
        right_node.update_node_bounds(spheres);

        self.nodes[index].left_first = node_idx as u32;
        self.nodes[index].prim_count = 0;

        self.nodes.push(left_node);
        self.nodes.push(right_node);

        self.subdivide(node_idx, spheres);
        self.subdivide(node_idx + 1, spheres);
    }
}
