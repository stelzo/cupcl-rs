use crate::*;
use petal_clustering::Fit;
use num_traits::Float;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub struct VoxelDownsampleParameters {
    pub voxel_size: f64,
    pub strategy: VoxelDownsampleStrategy,
}

impl Default for VoxelDownsampleParameters {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            strategy: VoxelDownsampleStrategy::default(),
        }
    }
}

fn hash_func_3d_points<T: Point<U>, U: Float + Into<f64>>(point: &T, voxel_size: f64) -> usize {
    let inv_res: f64 = 1.0 / voxel_size;
    let x: i32 = (point.get_x().into() * inv_res).round() as i32;
    let y: i32 = (point.get_y().into() * inv_res).round() as i32;
    let z: i32 = (point.get_z().into() * inv_res).round() as i32;
    let key = (x * 73856093) ^ (y * 471943) ^ (z * 83492791) % 100000;
    key as usize
}

fn voxel_downsample_center<T: Point<U>, U: Float + Into<f64>>(
    input: &PointCloud<T, U>,
    parameters: VoxelDownsampleParameters,
) -> PointCloud<T, U> {
    let mut hashes = rustc_hash::FxHashSet::default();
    let mut voxel_center = Vec::new();

    for point in input.buffer.iter() {
        let hash = hash_func_3d_points(point, parameters.voxel_size);
        if !hashes.contains(&hash) {
            hashes.insert(hash);
            voxel_center.push(T::with_xyzif64(
                ((point.get_x().into() / parameters.voxel_size).round() * parameters.voxel_size)
                    .into(),
                ((point.get_y().into() / parameters.voxel_size).round() * parameters.voxel_size)
                    .into(),
                ((point.get_z().into() / parameters.voxel_size).round() * parameters.voxel_size)
                    .into(),
                U::zero().into(),
            ));
        }
    }

    PointCloud::from_full_cloud(voxel_center)
}

pub fn voxel_downsample_center_par<T: Point<U>, U: Float + Into<f64>>(
    input: &PointCloud<T, U>,
    parameters: VoxelDownsampleParameters,
) -> PointCloud<T, U> {
    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = input.buffer.par_iter();
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = input.buffer.iter();

    let mut hashes = it
        .enumerate()
        .map(|(_, point)| {
            return (hash_func_3d_points(point, parameters.voxel_size), *point);
        })
        .collect::<Vec<(usize, T)>>();

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
            let x: f64 = (p.get_x().into() / parameters.voxel_size).round() * parameters.voxel_size;
            let y: f64 = (p.get_y().into() / parameters.voxel_size).round() * parameters.voxel_size;
            let z: f64 = (p.get_z().into() / parameters.voxel_size).round() * parameters.voxel_size;
            T::with_xyzif64(x, y, z, U::zero().into())
        })
        .collect();

    PointCloud::from_full_cloud(centers)
}

fn voxel_downsample_average<T: Point<U>, U: Float + Into<f64>>(
    input: &PointCloud<T, U>,
    parameters: VoxelDownsampleParameters,
) -> PointCloud<T, U> {
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
            let voxel_average = bucket.iter().fold(T::zero(), |acc: T, point| {
                T::with_xyzi(
                    acc.get_x() + point.get_x(),
                    acc.get_y() + point.get_y(),
                    acc.get_z() + point.get_z(),
                    acc.get_i() + point.get_i(),
                )
            });

            let x: f64 = voxel_average.get_x().into() / bucket.len() as f64;
            let y: f64 = voxel_average.get_y().into() / bucket.len() as f64;
            let z: f64 = voxel_average.get_z().into() / bucket.len() as f64;
            let i: f64 = voxel_average.get_i().into() / bucket.len() as f64;
            T::with_xyzif64(x, y, z, i)
        })
        .collect::<Vec<T>>();

    PointCloud::from_full_cloud(out)
}

#[cfg(all(feature = "cpu", not(feature = "rayon")))]
#[inline(always)]
fn hash_points<T: Point<U>, U: Float + Into<f64>>(
    input: &Vec<T>,
    voxel_size: f64,
) -> rustc_hash::FxHashMap<usize, Vec<T>> {
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
fn hash_points<T: Point<U>, U: Float + Into<f64>>(
    input: &Vec<T>,
    voxel_size: f64,
) -> dashmap::DashMap<usize, Vec<T>> {
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

fn voxel_downsample_median<T: Point<U>, U: Float + Into<f64>>(
    input: &PointCloud<T, U>,
    parameters: VoxelDownsampleParameters,
) -> PointCloud<T, U> {
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
            let voxel_average = bucket.iter().fold(T::zero(), |acc, point| {
                T::with_xyzi(
                    acc.get_x() + point.get_x(),
                    acc.get_y() + point.get_y(),
                    acc.get_z() + point.get_z(),
                    acc.get_i() + point.get_i(),
                )
            });

            let xyzi = T::with_xyzif64(
                voxel_average.get_x().into() / bucket.len() as f64,
                voxel_average.get_y().into() / bucket.len() as f64,
                voxel_average.get_z().into() / bucket.len() as f64,
                voxel_average.get_i().into() / bucket.len() as f64,
            );

            let mut min_dist = f64::MAX;
            let mut closest_point = T::zero();
            for point in bucket.iter() {
                let dist = (point.get_x().into() - xyzi.get_x().into()).powi(2)
                    + (point.get_y().into() - xyzi.get_y().into()).powi(2)
                    + (point.get_z().into() - xyzi.get_z().into()).powi(2);
                if dist < min_dist {
                    min_dist = dist;
                    closest_point = *point;
                }
            }

            closest_point
        })
        .collect::<Vec<T>>();

    PointCloud::from_full_cloud(out)
}

pub fn voxel_downsample<T: Point<U>, U: Float + Into<f64>>(
    input: &PointCloud<T, U>,
    parameters: VoxelDownsampleParameters,
) -> PointCloud<T, U> {
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

pub fn euclidean_cluster<T: Point<U>, U: Float + Into<f64>>(
    cloud: &PointCloud<T, U>,
    parameters: &EuclideanClusterParameters,
) -> Vec<PointCloud<T, U>> {
    if cloud.buffer.len() < parameters.min_points_per_cluster {
        return vec![cloud.clone()];
    }

    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = cloud.as_slice().par_iter();
    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = cloud.as_slice().iter();

    let data: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, _> = ndarray::ArrayBase::from_shape_vec(
        (cloud.buffer.len(), 3),
        it.map(|point| {
            vec![
                point.get_x().into(),
                point.get_y().into(),
                point.get_z().into(),
            ]
        })
        .flatten()
        .collect(),
    )
    .unwrap();

    let clustering = petal_clustering::Dbscan::new(
        parameters.tolerance,
        parameters.min_points_per_cluster,
        petal_neighbors::distance::Euclidean::default(),
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

#[inline(always)]
fn rotate_by_quaternion(p: (f64, f64, f64), q: (f64, f64, f64, f64)) -> (f64, f64, f64) {
    let u = (q.0, q.1, q.2);
    let s = q.3;

    let x = (
        (u.0 + u.0) * dot_float3(u, p),
        (u.1 + u.1) * dot_float3(u, p),
        (u.2 + u.2) * dot_float3(u, p),
    );
    let y = (
        p.0 * (s * s - dot_float3(u, u)),
        p.1 * (s * s - dot_float3(u, u)),
        p.2 * (s * s - dot_float3(u, u)),
    );
    let z = (
        cross_float3(u, p).0 * (s + s),
        cross_float3(u, p).1 * (s + s),
        cross_float3(u, p).2 * (s + s),
    );

    (x.0 + y.0 + z.0, x.1 + y.1 + z.1, x.2 + y.2 + z.2)
}

#[inline(always)]
fn dot_float3(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

#[inline(always)]
fn cross_float3(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

#[inline(always)]
fn transform_point(
    p: (f64, f64, f64),
    rotation: (f64, f64, f64, f64),
    translation: (f64, f64, f64),
) -> (f64, f64, f64) {
    let rotated = rotate_by_quaternion(p, rotation);
    (
        rotated.0 + translation.0,
        rotated.1 + translation.1,
        rotated.2 + translation.2,
    )
}

#[inline(always)]
fn safe_angle(angle: f64) -> f64 {
    let factor = (angle / (2.0 * std::f64::consts::PI)).floor() as i32;
    angle - factor as f64 * 2.0 * std::f64::consts::PI
}

#[inline(always)]
fn angle2d<U: Float>(v1: (U, U), v2: (U, U)) -> U {
    v1.1.atan2(v1.0) - v2.1.atan2(v2.0)
}

#[inline(always)]
fn inside_range(p: (f64, f64, f64), min_dist: f64, max_dist: f64) -> bool {
    let dist = p.0 * p.0 + p.1 * p.1 + p.2 * p.2;
    dist >= min_dist * min_dist && dist <= max_dist * max_dist
}

#[inline(always)]
fn inside_box(p: (f64, f64, f64), min: (f64, f64, f64), max: (f64, f64, f64)) -> bool {
    p.0 >= min.0 && p.0 <= max.0 && p.1 >= min.1 && p.1 <= max.1 && p.2 >= min.2 && p.2 <= max.2
}

#[inline(always)]
fn within_intensity<U: Float>(point_intensity: U, min_intensity: U, max_intensity: U) -> bool {
    point_intensity >= min_intensity && point_intensity <= max_intensity
}

#[inline(always)]
fn inside_horizontal_fov<U: Float + Into<f64>>(
    p: (U, U, U),
    fov_right: U,
    fov_left: U,
    forward: (U, U),
) -> bool {
    let angle = angle2d(forward, (p.0, p.1));
    let fov_angle = safe_angle((fov_right - fov_left).into());
    let local_angle = safe_angle((angle - fov_left).into());
    local_angle <= fov_angle
}

#[inline(always)]
fn point_inside_polygon_winding_number<U: Float>(
    p: (U, U, U),
    polygon: &[(U, U)],
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

pub fn passthrough_filter<T: Point<U>, U: Float + Into<f64>>(
    input: PointCloud<T, U>,
    params: &PassthroughFilterParameters,
) -> PointCloud<T, U> {
    #[cfg(all(feature = "rayon", feature = "cpu"))]
    let it = input.buffer.into_par_iter();

    #[cfg(all(feature = "cpu", not(feature = "rayon")))]
    let it = input.buffer.into_iter();

    let res = it
        .filter(|point| {
            let p_t = transform_point(
                (
                    point.get_x().into(),
                    point.get_y().into(),
                    point.get_z().into(),
                ),
                params.rotation,
                params.translation,
            );

            let mut is_inside_range = inside_range(p_t, params.min_dist, params.max_dist);
            if params.invert_distance {
                is_inside_range = !is_inside_range;
            }

            let mut is_within_intensity = within_intensity(
                point.get_i().into(),
                params.min_intensity,
                params.max_intensity,
            );
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
        .collect::<Vec<T>>();

    PointCloud::from_full_cloud(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::*;

    #[derive(pcl_derive::PointXYZI, Clone, Debug, PartialEq, Copy, Default)]
    struct PointXYZI {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    }

    impl PointXYZI {
        fn new(x: f32, y: f32, z: f32, i: f32) -> Self {
            Self {
                x,
                y,
                z,
                intensity: i,
            }
        }
    }

    #[test]
    fn cluster() {
        let cloud: Vec<PointXYZI> = read_pcl("test_files/cluster_sample.pcd".to_string());
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
        let mut cloud: Vec<PointXYZI> = generate_random_pointcloud(1000, 0.1, 1.0);

        let cloud_size = cloud.len();
        let pointcloud = PointCloud::from_full_cloud(cloud.clone());
        let params = PassthroughFilterParameters::default();
        let filtered = passthrough_filter(pointcloud, &params);
        let out_size = filtered.buffer.len();
        let back_to_vec = filtered.as_slice();
        assert_eq!(out_size, cloud_size);
        equal_list_with_linear_search(cloud.as_slice(), back_to_vec);

        cloud.push(PointXYZI::new(0.0, 0.0, 0.0, 0.0));
        cloud.push(PointXYZI::new(0.0, 0.0, 0.0, 0.0));

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
            PointXYZI::new(0.5, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.6, 0.0, 0.0),
            PointXYZI::new(0.0, 0.7, 6.0, 0.0),
            PointXYZI::new(0.0, 2.0, 0.0, 0.0),
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
            PointXYZI::new(1.0, 0.0, 0.0, 0.0),
            PointXYZI::new(-1.0, 0.0, 0.0, 0.0),
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
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.1, 0.0, 0.0),
            PointXYZI::new(0.0, 0.0, 0.1, 0.0),
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
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
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.1, 0.0, 0.0),
            PointXYZI::new(0.0, 0.0, 0.1, 0.0),
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
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
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.1, 0.0, 0.0),
            PointXYZI::new(0.0, 0.0, 0.1, 0.0),
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
        ]);
        let params = VoxelDownsampleParameters {
            voxel_size: 0.1,
            strategy: VoxelDownsampleStrategy::Center,
        };

        let filtered = voxel_downsample_center(&pointcloud, params);
        assert_eq!(filtered.buffer.len(), 4);

        // compress to one point
        let pointcloud_2 = PointCloud::from_full_cloud(vec![
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.1, 0.0, 0.0),
            PointXYZI::new(0.0, 0.0, 0.1, 0.0),
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.1, 0.1, 0.0),
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
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.1, 0.0, 0.0),
            PointXYZI::new(0.0, 0.0, 0.1, 0.0),
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
        ]);
        let params = VoxelDownsampleParameters {
            voxel_size: 0.1,
            strategy: VoxelDownsampleStrategy::Center,
        };

        let filtered = voxel_downsample_center_par(&pointcloud, params);
        assert_eq!(filtered.buffer.len(), 4);

        // compress to one point
        let pointcloud_2 = PointCloud::from_full_cloud(vec![
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.0, 0.0, 0.0),
            PointXYZI::new(0.0, 0.1, 0.0, 0.0),
            PointXYZI::new(0.0, 0.0, 0.1, 0.0),
            PointXYZI::new(0.0, 0.0, 0.0, 0.0),
            PointXYZI::new(0.1, 0.1, 0.1, 0.0),
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
            PointXYZI::new(1.0, 0.0, 0.0, 0.0),
            PointXYZI::new(1.0, 0.0, 0.0, 1.0),
            PointXYZI::new(1.0, 0.0, 0.0, 2.0),
            PointXYZI::new(1.0, 0.0, 0.0, 3.0),
            PointXYZI::new(1.0, 0.0, 0.0, 4.0),
        ]);
        let mut params = PassthroughFilterParameters::default();
        params.min_intensity = 1.0;
        params.max_intensity = 3.0;

        let filtered = passthrough_filter(pointcloud, &params);
        assert_eq!(filtered.buffer.len(), 3);
        assert_eq!(filtered.buffer[0].intensity, 1.0);
        assert_eq!(filtered.buffer[1].intensity, 2.0);
        assert_eq!(filtered.buffer[2].intensity, 3.0);
    }
}
