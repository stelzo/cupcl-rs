use crate::*;
use ndarray::{ArrayBase, OwnedRepr};
use petal_clustering::{Dbscan, Fit};
use petal_neighbors::distance::Euclidean;


pub struct VoxelDownsampleParameters {
    pub voxel_size: f32, // size of the voxel
}

impl Default for VoxelDownsampleParameters {
    fn default() -> Self {
        Self {
            voxel_size: 0.1,
        }
    }
}

#[allow(unused)]
pub fn voxel_downsample(input: &PointCloud, parameters: VoxelDownsampleParameters) -> PointCloud {
    input.clone() // TODO implement
}

pub struct EuclideanClusterParameters {
    pub tolerance: f64, // distance between points to be considered in the same cluster
    pub min_points_per_cluster: usize, // minimum number of points in a cluster
}

impl Default for EuclideanClusterParameters {
    fn default() -> Self {
        Self {
            tolerance: 0.1,
            min_points_per_cluster: 10,
        }
    }
}

pub fn euclidean_cluster(
    cloud: &PointCloud,
    parameters: &EuclideanClusterParameters,
) -> Vec<PointCloud> {

    if cloud.buffer.len() < parameters.min_points_per_cluster {
        return vec![cloud.clone()];
    }

    let data: ArrayBase<OwnedRepr<f64>, _> = ArrayBase::from_shape_vec((cloud.buffer.len(), 3), cloud.as_slice().iter().map(|point| vec![point.x as f64, point.y as f64, point.z as f64]).flatten().collect()).unwrap();

    let clustering = Dbscan::new(parameters.tolerance, parameters.min_points_per_cluster, Euclidean::default()).fit(&data);

    clustering.0.iter().map(|cluster| {
        let mut cluster_points = Vec::with_capacity(cluster.1.len());
        for idx in cluster.1 {
            cluster_points.push(cloud.buffer[*idx].clone());
        }
        PointCloud::from_full_cloud(cluster_points)
    }).collect()
}

fn rotate_by_quaternion(p: (f32, f32, f32), q: (f32, f32, f32, f32)) -> (f32, f32, f32) {
    let u = (q.0, q.1, q.2);
    let s = q.3;

    let x = (
        u.0 * 2.0 * dot_float3(u, p),
        u.1 * 2.0 * dot_float3(u, p),
        u.2 * 2.0 * dot_float3(u, p),
    );
    let y = (
        p.0 * (s * s - dot_float3(u, u)),
        p.1 * (s * s - dot_float3(u, u)),
        p.2 * (s * s - dot_float3(u, u)),
    );
    let z = (
        cross_float3(u, p).0 * 2.0 * s,
        cross_float3(u, p).1 * 2.0 * s,
        cross_float3(u, p).2 * 2.0 * s,
    );

    (x.0 + y.0 + z.0, x.1 + y.1 + z.1, x.2 + y.2 + z.2)
}

fn dot_float3(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn cross_float3(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn transform_point(
    p: (f32, f32, f32),
    rotation: (f32, f32, f32, f32),
    translation: (f32, f32, f32),
) -> (f32, f32, f32) {
    let rotated = rotate_by_quaternion(p, rotation);
    (
        rotated.0 + translation.0,
        rotated.1 + translation.1,
        rotated.2 + translation.2,
    )
}

fn safe_angle(angle: f32) -> f32 {
    let factor = (angle / (2.0 * std::f32::consts::PI)).floor() as i32;
    angle - factor as f32 * 2.0 * std::f32::consts::PI
}

fn angle2d(v1: (f32, f32), v2: (f32, f32)) -> f32 {
    v1.1.atan2(v1.0) - v2.1.atan2(v2.0)
}

fn inside_range(p: (f32, f32, f32), min_dist: f32, max_dist: f32) -> bool {
    let dist = p.0 * p.0 + p.1 * p.1 + p.2 * p.2;
    dist >= min_dist * min_dist && dist <= max_dist * max_dist
}

fn inside_box(p: (f32, f32, f32), min: (f32, f32, f32), max: (f32, f32, f32)) -> bool {
    p.0 >= min.0 && p.0 <= max.0 && p.1 >= min.1 && p.1 <= max.1 && p.2 >= min.2 && p.2 <= max.2
}

fn inside_horizontal_fov(
    p: (f32, f32, f32),
    fov_right: f32,
    fov_left: f32,
    forward: (f32, f32),
) -> bool {
    let angle = angle2d(forward, (p.0, p.1));
    let fov_angle = safe_angle(fov_right - fov_left);
    let local_angle = safe_angle(angle - fov_left);
    local_angle <= fov_angle
}

fn point_inside_polygon_winding_number(
    p: (f32, f32, f32),
    polygon: &[(f32, f32)],
    polygon_size: usize,
) -> bool {
    let mut winding_number = 0;
    for i in 0..polygon_size {
        let v1 = polygon[i];
        let v2 = polygon[(i + 1) % polygon_size];

        let cont =
            p.1 < v1.1.min(v2.1) || p.1 > v1.1.max(v2.1) || p.0 > v1.0.max(v2.0) || v1.1 == v2.1;

        if cont {
            continue;
        }

        let y_slope = (v2.0 - v1.0) / (v2.1 - v1.1);
        let x_intercept = (p.1 - v1.1) * y_slope + v1.0;
        if v1.0 == v2.0 || p.0 <= x_intercept {
            winding_number += if v2.1 > v1.1 { 1 } else { -1 };
        }
    }

    winding_number != 0
}

pub fn passthrough_filter(input: PointCloud, params: &PassthroughFilterParameters) -> PointCloud {
    let res = input
        .buffer
        .into_iter()
        .filter(|point| {
            let p_t = transform_point(
                (point.x, point.y, point.z),
                params.rotation,
                params.translation,
            );

            let mut is_inside_range = inside_range(p_t, params.min_dist, params.max_dist);
            if params.invert_distance {
                is_inside_range = !is_inside_range;
            }


            let mut is_inside_box = inside_box(p_t, params.min, params.max);
            if params.invert_bounding_box {
                is_inside_box = !is_inside_box;
            }

            let mut is_inside_fov =
                inside_horizontal_fov(p_t, params.fov_right, params.fov_left, params.forward);
            if params.invert_fov {
                is_inside_fov = !is_inside_fov;
            }

            match &params.polygon {
                None => {
                    return !(!is_inside_range || !is_inside_box || (params.enable_horizontal_fov && !is_inside_fov));
                }
                Some(polygon) => {
                    let mut is_inside_polygon =
                        point_inside_polygon_winding_number(p_t, polygon, polygon.len());
                    if params.invert_polygon {
                        is_inside_polygon = !is_inside_polygon;
                    }

                    return is_inside_polygon;
                }
            }
        })
        .collect::<Vec<Point>>();

    PointCloud::from_full_cloud(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::*;

    #[test]
    fn cluster() {
        let cloud = read_pcl("test_files/cluster_sample.pcd".to_string());
        let pointcloud = PointCloud::from_full_cloud(cloud);

        let parameters = EuclideanClusterParameters {
            tolerance: 0.02,
            min_points_per_cluster: 50,
        };

        let clusters = euclidean_cluster(
            &pointcloud,
            &parameters,
        );
        assert!(!clusters.is_empty());

        for cluster in clusters {
            assert!(cluster.buffer.len() > parameters.min_points_per_cluster as usize);
        }
    }

    #[test]
    fn passthrough() {
        let mut cloud = generate_random_pointcloud(1000, 0.1, 1.0);

        let cloud_size = cloud.len();
        let pointcloud = PointCloud::from_full_cloud(cloud.clone());
        let params = PassthroughFilterParameters::default();
        let filtered = passthrough_filter( pointcloud, &params);
        let out_size = filtered.buffer.len();
        let back_to_vec = filtered.as_slice();
        assert_eq!(out_size, cloud_size);
        equal_list_with_linear_search(cloud.as_slice(), back_to_vec);

        cloud.push(Point::new(0.0, 0.0, 0.0, 0.0));
        cloud.push(Point::new(0.0, 0.0, 0.0, 0.0));

        let cloud_size = cloud.len();

        let pointcloud = PointCloud::from_full_cloud(cloud.clone());
        let mut params = PassthroughFilterParameters::default();

        params.min = (-1.0, -1.0, -1.0);
        params.max = (1.0, 1.0, 1.0);
        params.min_dist = 0.05;
        let filtered = passthrough_filter(pointcloud.clone(), &params);
        assert_eq!(filtered.buffer.len(), cloud_size - 2);

        params.invert_distance = true;
        let filtered = passthrough_filter(pointcloud.clone(), &params);
        assert_eq!(filtered.buffer.len(), 2);

        let cloud = vec![
            Point::new(0.5, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.6, 0.0, 0.0),
            Point::new(0.0, 0.7, 6.0, 0.0),
            Point::new(0.0, 2.0, 0.0, 0.0),
        ];
        let pointcloud = PointCloud::from_full_cloud(cloud.clone());
        let mut params = PassthroughFilterParameters::default();
        let polygon = vec![
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
        ];
        params.polygon = Some(polygon);
        let filtered = passthrough_filter(pointcloud.clone(), &params);
        assert_eq!(filtered.buffer.len(), 3);
    }

    #[test]
    fn fov_passthrough() {
        let pointcloud = PointCloud::from_full_cloud(
            vec![
                Point::new(1.0, 0.0, 0.0, 0.0),
                Point::new(-1.0, 0.0, 0.0, 0.0),
            ],
        );
        let mut params = PassthroughFilterParameters::default();
        params.fov_left = 90.0;
        params.fov_right = 180.0;
        params.enable_horizontal_fov = true;
        params.forward = (1.0, 0.0);

        let filtered = passthrough_filter(pointcloud, &params);
        assert_eq!(filtered.buffer.len(), 1);
    }
}
