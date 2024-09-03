use glam::{Vec3};
use crate::material::Material;
use crate::sphere::Sphere;
use crate::util_funcs::{random_f32, random_range_f32, random_vec3, random_vec3_range};

pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
}

impl Scene {
    pub fn new() -> Self {
        let mat_ground = Material::Lambertian(Vec3::new(0.8, 0.8, 0.0));
        let mat_center = Material::Lambertian(Vec3::new(0.1, 0.2, 0.5));
        let mat_left = Material::Dielectric(1.50);
        // let mat_left = Material::Metal(Vec3::new(0.8, 0.8, 0.8), 0.3);
        let mat_bubble = Material::Dielectric(1.00/1.50);
        let mat_right = Material::Metal(Vec3::new(0.8, 0.6, 0.2), 1.0);

        let mut materials = vec![mat_ground, mat_center, mat_left, mat_right, mat_bubble];

        let ground = Sphere::new(
            Vec3::new(0.0, -100.5, -1.0),
            100.0,
            0);
        let center = Sphere::new(
            Vec3::new(0.0, 0.0, -1.2),
            0.5,
            1);
        let left = Sphere::new(
            Vec3::new(-1.0, 0.0, -1.0),
            0.5,
            2);
        let bubble = Sphere::new(
            Vec3::new(-1.0, 0.0, -1.0),
            0.4,
            4);
        let right = Sphere::new(
            Vec3::new(1.0, 0.0, -1.0),
            0.5,
            3);

        let mut spheres = vec![ground, center, right, left, bubble];

        Self { spheres, materials }
    }

    pub fn book_one_final() -> Self {
        let mut spheres = Vec::<Sphere>::new();
        let mut materials = Vec::<Material>::new();
        // ground
        let ground_mat = Material::Lambertian(Vec3::new(0.5, 0.5, 0.5));
        materials.push(ground_mat);

        let ground = Sphere::new(
            Vec3::new(0.0, -1000.0, 0.0),
            1000.0,
            0);
        spheres.push(ground);

        // random marbles
        for a in  -11 .. 11 {
            for b in -11 .. 11 {
                let choose_mat = random_f32();
                let center = Vec3::new(a as f32 + 0.9 * random_f32(), 0.2,
                                        b as f32 + 0.9 * random_f32());

                if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9 {

                    if choose_mat < 0.8 {
                        // diffuse
                        let albedo = random_vec3() * random_vec3();
                        let sphere_material = Material::Lambertian(albedo);
                        materials.push(sphere_material);
                        spheres.push(Sphere::new(center, 0.2, (materials.len() - 1) as u32));
                    } else if choose_mat < 0.95 {
                        // metal
                        let albedo = random_vec3_range(0.5, 1.0);
                        let fuzz = random_range_f32(0.0,0.5);
                        let sphere_material = Material::Metal(albedo, fuzz);
                        materials.push(sphere_material);
                        spheres.push(Sphere::new(center, 0.2, (materials.len() - 1) as u32));
                    } else {
                        // glass
                        let sphere_material = Material::Dielectric(1.5);
                        materials.push(sphere_material);
                        spheres.push(Sphere::new(center, 0.2, (materials.len() - 1) as u32));
                    }
                }
            }
        }

        // Big spheres
        let dia_mat = Material::Dielectric(1.50);
        materials.push(dia_mat);
        spheres.push(Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0, (materials.len() - 1) as u32));

        let lamb_mat = Material::Lambertian(Vec3::new(0.4, 0.2, 0.1));
        materials.push(lamb_mat);
        spheres.push(Sphere::new(Vec3::new(-4.0, 1.0, 0.0), 1.0, (materials.len() - 1) as u32));

        let met_mat = Material::Metal(Vec3::new(0.7, 0.6, 0.5), 0.0);
        materials.push(met_mat);
        spheres.push(Sphere::new(Vec3::new(4.0, 1.0, 0.0), 1.0, (materials.len() - 1) as u32));

        Self { spheres, materials }
    }

}