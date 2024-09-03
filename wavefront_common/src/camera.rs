use glam::{Vec3};

#[derive(Copy, Clone, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub pitch: f32,
    pub yaw: f32
}

impl Camera {
    pub fn new(look_from: Vec3, look_at: Vec3) -> Self {

        let position = look_from;
        let forwards = (look_at - position).normalize();

        let pitch = forwards.y.acos();
        let yaw = forwards.x.atan2(forwards.z);

        Self {
            position,
            pitch,
            yaw
        }
    }

    pub fn book_one_final_camera() -> Self {
        let look_at = Vec3::new(0.0, 0.0, 0.0);
        let look_from = Vec3::new(13.0, 2.0, 3.0);
        Self::new(look_from, look_at)
    }

    pub fn get_camera(&self) -> (Vec3, f32, f32) {
        (self.position, self.pitch, self.yaw)
    }
    pub fn update_camera(&mut self, camera: Camera) {
        self.position = camera.position;
        self.pitch = camera.pitch;
        self.yaw = camera.yaw;
    }

    pub fn view_transform(& self) -> [[f32; 4]; 4]
    {
        // look-at transformation with x-axis flipped to account for
        // rh world coordinates but lh camera coordinates
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let dir = Vec3::new(sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw);
        let right = dir.cross(Vec3::new(0.0, 1.0, 0.0));
        let up = right.cross(dir);
        let center = self.position;

        let world_from_camera = [
            [right.x, right.y, right.z, 0.0],
            [up.x, up.y, up.z, 0.0],
            [dir.x, dir.y, dir.z, 0.0],
            [center.x, center.y, center.z, 1.0]
        ];

        // if wfc is of form T*R, then inv is inv(T)*inv(T), which is why we have the dot
        // product now in the fourth column
        // let camera_from_world = Mat4::from_cols(
        //     Vec4::new(-right.x, new_up.x, dir.x, 0.0),
        //     Vec4::new(-right.y, new_up.y, dir.y, 0.0),
        //     Vec4::new(-right.z, new_up.z, dir.z, 0.0),
        //     Vec4::new(center.dot(right), -center.dot(new_up), -center.dot(dir), 1.0)
        // );

        world_from_camera
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angles() {
        let look_from = Vec3::new(13.0, 2.0, 3.0);
        let look_at = Vec3::new(0.0, 0.0, 0.0);
        let camera = Camera::new(look_from, look_at);
        println!("pitch: {}, yaw:{}",camera.pitch.to_degrees(), camera.yaw.to_degrees());
        let (sin_pitch, cos_pitch) = camera.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
        let dir = Vec3::new(sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw);
        println!("direction: {}", dir);
    }
}