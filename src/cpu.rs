use crate::*;
use ndarray::{ArrayBase, OwnedRepr};
use petal_clustering::{Dbscan, Fit};
use petal_neighbors::distance::Euclidean;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub struct VoxelDownsampleParameters {
    pub voxel_size: f32, // size of the voxel
    pub strategy: VoxelDownsampleStrategy,
}

impl Default for VoxelDownsampleParameters {
    fn default() -> Self {
        Self {
            voxel_size: 0.1,
            strategy: VoxelDownsampleStrategy::Center,
        }
    }
}

fn hash_func_3d_points(point: &Point, voxel_size: f32) -> usize {
    let x = (point.x / voxel_size).round() as usize;
    let y = (point.y / voxel_size).round() as usize;
    let z = (point.z / voxel_size).round() as usize;
    let mut key: usize = (x * 73856093) ^ (y * 471943) ^ (z * 83492791);
    key = key % 100000; // MOD 100000, 1000000007
    key
}

// hash point into grid cell and return the center of the cell as the new point
fn voxel_downsample_center(
    input: &PointCloud,
    parameters: VoxelDownsampleParameters,
) -> PointCloud {
    let mut hashes = rustc_hash::FxHashSet::default();
    let mut voxel_center = Vec::<Point>::new();

    for point in input.buffer.iter() {
        let hash = hash_func_3d_points(point, parameters.voxel_size);
        if !hashes.contains(&hash) {
            hashes.insert(hash);
            voxel_center.push(Point::new(
                (point.x / parameters.voxel_size).round() * parameters.voxel_size,
                (point.y / parameters.voxel_size).round() * parameters.voxel_size,
                (point.z / parameters.voxel_size).round() * parameters.voxel_size,
                0.0,
            ));
        }
    }

    PointCloud::from_full_cloud(voxel_center)
}

// Parallel version of voxel_downsample_center without using hashsets.
pub fn voxel_downsample_center_par(
    input: &PointCloud,
    parameters: VoxelDownsampleParameters,
) -> PointCloud {
    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = input.buffer.par_iter();
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = input.buffer.iter();

    let mut hashes = it
        .enumerate()
        .map(|(_, point)| {
            return (hash_func_3d_points(point, parameters.voxel_size), *point);
        })
        .collect::<Vec<(usize, Point)>>();

    #[cfg(all(feature = "rayon", feature = "cpu"))]
    hashes.par_sort_unstable_by(|(hash1, _), (hash2, _)| hash1.cmp(&hash2));
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    hashes.sort_unstable_by(|(hash1, _), (hash2, _)| hash1.cmp(&hash2));

    // TODO parallelize this
    hashes.dedup_by(|(hash1, _), (hash2, _)| hash1 == hash2);

    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = hashes.par_iter();
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = hashes.iter();

    let centers = it
        .map(|(_, p)| {
            let x = (p.x / parameters.voxel_size).round() * parameters.voxel_size;
            let y = (p.y / parameters.voxel_size).round() * parameters.voxel_size;
            let z = (p.z / parameters.voxel_size).round() * parameters.voxel_size;
            Point::new(x, y, z, 0.0)
        })
        .collect();

    PointCloud::from_full_cloud(centers)
}

// hash point into grid cell and return the average of the points inside that cell as the new point
fn voxel_downsample_average(
    input: &PointCloud,
    parameters: VoxelDownsampleParameters,
) -> PointCloud {
    let hashes = hash_points(&input.buffer, parameters.voxel_size);

    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = hashes.par_iter();
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = hashes.iter();

    let out = it
        .map(|bucket_ref| {
            #[cfg(all(feature = "cpu", not(feature = "rayon")))]
            let bucket = bucket_ref.1;
            #[cfg(all(feature = "rayon", feature = "cpu"))]
            let bucket = bucket_ref;
            let voxel_average = bucket
                .iter()
                .fold(Point::new(0.0, 0.0, 0.0, 0.0), |acc, point| {
                    Point::new(
                        acc.x + point.x,
                        acc.y + point.y,
                        acc.z + point.z,
                        acc.i + point.i,
                    )
                });
            let x = voxel_average.x / bucket.len() as f32;
            let y = voxel_average.y / bucket.len() as f32;
            let z = voxel_average.z / bucket.len() as f32;
            let i = voxel_average.i / bucket.len() as f32;
            Point::new(x, y, z, i)
        })
        .collect::<Vec<Point>>();

    PointCloud::from_full_cloud(out)
}

#[cfg(all(feature = "cpu", not(feature = "rayon")))]
#[inline]
fn hash_points(input: &Vec<Point>, voxel_size: f32) -> rustc_hash::FxHashMap<usize, Vec<Point>> {
    let mut hashes = rustc_hash::FxHashMap::default();

    for point in input.iter() {
        let hash = hash_func_3d_points(point, voxel_size);
        if !hashes.contains_key(&hash) {
            hashes.insert(hash, Vec::new());
        }
        hashes
            .get_mut(&hash)
            .expect("entry created the line before")
            .push(*point);
    }

    hashes
}

#[cfg(all(feature = "rayon", feature = "cpu"))]
#[inline]
fn hash_points(input: &Vec<Point>, voxel_size: f32) -> dashmap::DashMap<usize, Vec<Point>> {
    let hashes = dashmap::DashMap::new();

    input.par_iter().for_each(|point| {
        let hash = hash_func_3d_points(point, voxel_size);
        if !hashes.contains_key(&hash) {
            hashes.insert(hash, Vec::new());
        }
        hashes
            .get_mut(&hash)
            .expect("entry created the line before")
            .push(*point);
    });

    hashes
}

// hash point into grid cell and return the point that is closest to the average of the points inside that cell as the new point.
fn voxel_downsample_median(
    input: &PointCloud,
    parameters: VoxelDownsampleParameters,
) -> PointCloud {
    let hashes = hash_points(&input.buffer, parameters.voxel_size);

    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = hashes.par_iter();
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = hashes.iter();

    let out = it
        .map(|bucket_ref| {
            #[cfg(all(feature = "cpu", not(feature = "rayon")))]
            let bucket = bucket_ref.1;
            #[cfg(all(feature = "rayon", feature = "cpu"))]
            let bucket = bucket_ref;
            let voxel_average = bucket
                .iter()
                .fold(Point::new(0.0, 0.0, 0.0, 0.0), |acc, point| {
                    Point::new(
                        acc.x + point.x,
                        acc.y + point.y,
                        acc.z + point.z,
                        acc.i + point.i,
                    )
                });
            let x = voxel_average.x / bucket.len() as f32;
            let y = voxel_average.y / bucket.len() as f32;
            let z = voxel_average.z / bucket.len() as f32;

            let mut min_dist = f32::MAX;
            let mut closest_point = Point::new(0.0, 0.0, 0.0, 0.0);
            for point in bucket.iter() {
                let dist = (point.x - x).powi(2) + (point.y - y).powi(2) + (point.z - z).powi(2);
                if dist < min_dist {
                    min_dist = dist;
                    closest_point = *point;
                }
            }

            closest_point
        })
        .collect::<Vec<Point>>();

    PointCloud::from_full_cloud(out)
}

#[allow(unused)]
pub fn voxel_downsample(input: &PointCloud, parameters: VoxelDownsampleParameters) -> PointCloud {
    match parameters.strategy {
        VoxelDownsampleStrategy::Center => voxel_downsample_center(input, parameters),
        VoxelDownsampleStrategy::Average => voxel_downsample_average(input, parameters),
        VoxelDownsampleStrategy::Median => voxel_downsample_median(input, parameters),
    }
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

    let data: ArrayBase<OwnedRepr<f64>, _> = ArrayBase::from_shape_vec(
        (cloud.buffer.len(), 3),
        cloud
            .as_slice()
            .iter()
            .map(|point| vec![point.x as f64, point.y as f64, point.z as f64])
            .flatten()
            .collect(),
    )
    .unwrap();

    let clustering = Dbscan::new(
        parameters.tolerance,
        parameters.min_points_per_cluster,
        Euclidean::default(),
    )
    .fit(&data);

    clustering
        .0
        .iter()
        .map(|cluster| {
            let mut cluster_points = Vec::with_capacity(cluster.1.len());
            for idx in cluster.1 {
                cluster_points.push(cloud.buffer[*idx].clone());
            }
            PointCloud::from_full_cloud(cluster_points)
        })
        .collect()
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

fn within_intensity(point_intensity: f32, min_intensity: f32, max_intensity: f32) -> bool {
    point_intensity >= min_intensity && point_intensity <= max_intensity
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
    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = input.buffer.into_par_iter();

    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = input.buffer.into_iter();

    let res = it
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

            let mut is_within_intensity =
                within_intensity(point.i, params.min_intensity, params.max_intensity);
            if params.invert_intensity {
                is_within_intensity = !is_within_intensity;
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

            let is_inside_polygon = match &params.polygon {
                None => true,
                Some(polygon) => {
                    let mut is_inside_polygon =
                        point_inside_polygon_winding_number(p_t, polygon, polygon.len());
                    if params.invert_polygon {
                        is_inside_polygon = !is_inside_polygon;
                    }
                    polygon.len() != 0 && is_inside_polygon
                }
            };

            is_inside_range
                && is_inside_box
                && is_within_intensity
                && (!params.enable_horizontal_fov || is_inside_fov)
                && is_inside_polygon
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

        let clusters = euclidean_cluster(&pointcloud, &parameters);
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
        let filtered = passthrough_filter(pointcloud, &params);
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
        let polygon = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        params.polygon = Some(polygon);
        let filtered = passthrough_filter(pointcloud.clone(), &params);
        assert_eq!(filtered.buffer.len(), 3);
    }

    #[test]
    fn fov_passthrough() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            Point::new(1.0, 0.0, 0.0, 0.0),
            Point::new(-1.0, 0.0, 0.0, 0.0),
        ]);
        let mut params = PassthroughFilterParameters::default();
        params.fov_left = 90.0;
        params.fov_right = 180.0;
        params.enable_horizontal_fov = true;
        params.forward = (1.0, 0.0);

        let filtered = passthrough_filter(pointcloud, &params);
        assert_eq!(filtered.buffer.len(), 1);
    }

    #[test]
    fn voxel_downsample_average_test() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.1, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.1, 0.0),
            Point::new(0.0, 0.0, 0.0, 0.0),
        ]);

        let params = VoxelDownsampleParameters {
            voxel_size: 0.21,
            strategy: VoxelDownsampleStrategy::Average,
        };

        let filtered = voxel_downsample_average(&pointcloud, params);
        assert_eq!(filtered.buffer.len(), 1);
    }

    #[test]
    fn voxel_downsample_median_test() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.1, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.1, 0.0),
            Point::new(0.0, 0.0, 0.0, 0.0),
        ]);

        let params = VoxelDownsampleParameters {
            voxel_size: 0.21,
            strategy: VoxelDownsampleStrategy::Median,
        };

        let filtered = voxel_downsample_median(&pointcloud, params);
        assert_eq!(filtered.buffer.len(), 1);
    }

    #[test]
    fn voxel_downsample_center_test() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.1, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.1, 0.0),
            Point::new(0.0, 0.0, 0.0, 0.0),
        ]);
        let params = VoxelDownsampleParameters {
            voxel_size: 0.1,
            strategy: VoxelDownsampleStrategy::Center,
        };

        let filtered = voxel_downsample_center(&pointcloud, params);
        assert_eq!(filtered.buffer.len(), 4);

        // compress to one point
        let pointcloud_2 = PointCloud::from_full_cloud(vec![
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.1, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.1, 0.0),
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.1, 0.1, 0.0),
        ]);

        let params_2 = VoxelDownsampleParameters {
            voxel_size: 0.21,
            strategy: VoxelDownsampleStrategy::Center,
        };

        let filtered_2 = voxel_downsample_center(&pointcloud_2, params_2);
        assert_eq!(filtered_2.buffer.len(), 1);
    }

    #[test]
    fn voxel_downsample_center_par_test() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.1, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.1, 0.0),
            Point::new(0.0, 0.0, 0.0, 0.0),
        ]);
        let params = VoxelDownsampleParameters {
            voxel_size: 0.1,
            strategy: VoxelDownsampleStrategy::Center,
        };

        let filtered = voxel_downsample_center_par(&pointcloud, params);
        assert_eq!(filtered.buffer.len(), 4);

        // compress to one point
        let pointcloud_2 = PointCloud::from_full_cloud(vec![
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.1, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.1, 0.0),
            Point::new(0.0, 0.0, 0.0, 0.0),
            Point::new(0.1, 0.1, 0.1, 0.0),
        ]);

        let params_2 = VoxelDownsampleParameters {
            voxel_size: 0.21,
            strategy: VoxelDownsampleStrategy::Center,
        };

        let filtered_2 = voxel_downsample_center_par(&pointcloud_2, params_2);
        assert_eq!(filtered_2.buffer.len(), 1);
    }

    #[test]
    fn intensity_passthrough() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            Point::new(1.0, 0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0, 1.0),
            Point::new(1.0, 0.0, 0.0, 2.0),
            Point::new(1.0, 0.0, 0.0, 3.0),
            Point::new(1.0, 0.0, 0.0, 4.0),
        ]);
        let mut params = PassthroughFilterParameters::default();
        params.min_intensity = 1.0;
        params.max_intensity = 3.0;

        let filtered = passthrough_filter(pointcloud, &params);
        assert_eq!(filtered.buffer.len(), 3);
        assert_eq!(filtered.buffer[0].i, 1.0);
        assert_eq!(filtered.buffer[1].i, 2.0);
        assert_eq!(filtered.buffer[2].i, 3.0);
    }
}
