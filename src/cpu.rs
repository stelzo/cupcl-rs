use crate::*;
use num_traits::Float;
use petal_clustering::Fit;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
use ros_pointcloud2::{points::PointXYZ, PointCloud2Msg};

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

fn hash_func_3d_points<T: Point>(point: &T, voxel_size: f64) -> usize {
    let inv_res: f64 = 1.0 / voxel_size;
    let x: i32 = (point.get_x() * inv_res as f32).round() as i32;
    let y: i32 = (point.get_y() * inv_res as f32).round() as i32;
    let z: i32 = (point.get_z() * inv_res as f32).round() as i32;
    let key = (x * 73856093) ^ (y * 471943) ^ (z * 83492791) % 100000;
    key as usize
}

/*fn voxel_downsample_center<T: Point<U>, U: Float + Into<f64>>(
    input: PointCloud<T, U>,
    parameters: VoxelDownsampleParameters,
) -> PointCloud<T, U> {
    let mut hashes = rustc_hash::FxHashSet::default();
    let mut voxel_center = Vec::new();

    for point in input.it {
        let hash: usize = hash_func_3d_points(&point, parameters.voxel_size);
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
}*/

pub fn voxel_downsample_center_par(
    input: PointCloud,
    parameters: VoxelDownsampleParameters,
) -> PointCloud {
    #[cfg(all(not(feature = "ros"), feature = "rayon", feature = "cpu"))]
    let it = input.buffer.par_iter();
    #[cfg(all(not(feature = "ros"), feature = "cpu", not(feature = "rayon")))]
    let it = input.it.unwrap(); // TODO sure only be made with it in this case?

    #[cfg(all(feature = "ros", feature = "cpu", not(feature = "rayon")))]
    let it = input.ros_cloud.unwrap().try_into_vec().unwrap().into_iter(); // TODO handle error
    #[cfg(all(feature = "ros", feature = "cpu", feature = "rayon"))]
    let it = input.ros_cloud.unwrap().try_into_par_iter().unwrap();

    let mut hashes = it
        .map(|point: PointXYZ| {
            return (hash_func_3d_points(&point, parameters.voxel_size), point);
        })
        .collect::<Vec<(usize, PointXYZ)>>();

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

    let centers: Vec<PointXYZ> = it
        .map(|(_, p)| {
            let x: f64 = (p.get_x() as f64 / parameters.voxel_size).round() * parameters.voxel_size;
            let y: f64 = (p.get_y() as f64 / parameters.voxel_size).round() * parameters.voxel_size;
            let z: f64 = (p.get_z() as f64 / parameters.voxel_size).round() * parameters.voxel_size;
            PointXYZ::new(x as f32, y as f32, z as f32)
        })
        .collect();

    PointCloud::from_ros_cloud(PointCloud2Msg::try_from_iter(centers).unwrap())
}

fn voxel_downsample_average(
    input: PointCloud,
    parameters: VoxelDownsampleParameters,
) -> PointCloud {
    #[cfg(all(feature = "ros", feature = "cpu", not(feature = "rayon")))]
    let it = input.ros_cloud.unwrap().try_into_vec().unwrap().into_iter(); // TODO handle error
    #[cfg(all(feature = "ros", feature = "cpu", feature = "rayon"))]
    let it = input.ros_cloud.unwrap().try_into_par_iter().unwrap();

    #[cfg(feature = "rayon")]
    let hashes = dashmap::DashMap::new();
    #[cfg(not(feature = "rayon"))]
    let mut hashes = rustc_hash::FxHashMap::default();
    it.for_each(|point: PointXYZ| {
        let hash = hash_func_3d_points(&point, parameters.voxel_size);
        if !hashes.contains_key(&hash) {
            hashes.insert(hash, Vec::new());
        }
        hashes
            .get_mut(&hash)
            .expect("entry created the line before")
            .push(point);
    });

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
            let voxel_average =
                bucket
                    .iter()
                    .fold(PointXYZ::default(), |acc: PointXYZ, point| {
                        PointXYZ::new(
                            acc.get_x() + point.get_x(),
                            acc.get_y() + point.get_y(),
                            acc.get_z() + point.get_z(),
                        )
                    });

            let x: f64 = voxel_average.get_x() as f64 / bucket.len() as f64;
            let y: f64 = voxel_average.get_y() as f64 / bucket.len() as f64;
            let z: f64 = voxel_average.get_z() as f64 / bucket.len() as f64;
            let i: f64 = voxel_average.get_i() as f64 / bucket.len() as f64;
            PointXYZ::new(x as f32, y as f32, z as f32)
        })
        .collect::<Vec<PointXYZ>>();

    PointCloud::from_full_cloud(out)
}

#[cfg(all(feature = "cpu", not(feature = "rayon")))]
#[inline(always)]
fn hash_points<T: Point, U: Float + Into<f64>>(
    input: Box<dyn Iterator<Item = T>>,
    voxel_size: f64,
) -> rustc_hash::FxHashMap<usize, Vec<T>> {
    let mut hashes = rustc_hash::FxHashMap::default();

    for point in input {
        let hash = hash_func_3d_points(&point, voxel_size);
        if !hashes.contains_key(&hash) {
            hashes.insert(hash, Vec::new());
        }
        hashes
            .get_mut(&hash)
            .expect("entry created the line before")
            .push(point);
    }

    hashes
}

#[cfg(all(feature = "rayon", feature = "cpu"))]
#[inline]
fn hash_points<T: Point>(input: &Vec<T>, voxel_size: f64) -> dashmap::DashMap<usize, Vec<T>> {
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

fn voxel_downsample_median(input: PointCloud, parameters: VoxelDownsampleParameters) -> PointCloud {
    #[cfg(all(feature = "ros", feature = "cpu", not(feature = "rayon")))]
    let it = input.ros_cloud.unwrap().try_into_vec().unwrap().into_iter(); // TODO handle error
    #[cfg(all(feature = "ros", feature = "cpu", feature = "rayon"))]
    let it = input.ros_cloud.unwrap().try_into_par_iter().unwrap();

    #[cfg(feature = "rayon")]
    let hashes = dashmap::DashMap::new();
    #[cfg(not(feature = "rayon"))]
    let mut hashes = rustc_hash::FxHashMap::default();
    it.for_each(|point: PointXYZ| {
        let hash = hash_func_3d_points(&point, parameters.voxel_size);
        if !hashes.contains_key(&hash) {
            hashes.insert(hash, Vec::new());
        }
        hashes
            .get_mut(&hash)
            .expect("entry created the line before")
            .push(point);
    });

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
            let voxel_average = bucket.iter().fold(PointXYZ::default(), |acc, point| {
                PointXYZ::new(
                    acc.get_x() + point.get_x(),
                    acc.get_y() + point.get_y(),
                    acc.get_z() + point.get_z(),
                )
            });

            let xyzi = PointXYZ::new(
                voxel_average.get_x() as f32 / bucket.len() as f32,
                voxel_average.get_y() as f32 / bucket.len() as f32,
                voxel_average.get_z() as f32 / bucket.len() as f32,
            );

            let mut min_dist = f64::MAX;
            let mut closest_point = PointXYZ::default();
            for point in bucket.iter() {
                let dist = (point.get_x() as f64 - xyzi.get_x() as f64).powi(2)
                    + (point.get_y() as f64 - xyzi.get_y() as f64).powi(2)
                    + (point.get_z() as f64 - xyzi.get_z() as f64).powi(2);
                if dist < min_dist {
                    min_dist = dist;
                    closest_point = *point;
                }
            }

            closest_point
        })
        .collect::<Vec<PointXYZ>>();

    PointCloud::from_full_cloud(out)
}

pub fn voxel_downsample(input: PointCloud, parameters: VoxelDownsampleParameters) -> PointCloud {
    match parameters.strategy {
        VoxelDownsampleStrategy::Center => todo!(), //voxel_downsample_center(input, parameters),
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
    cloud: PointCloud,
    parameters: &EuclideanClusterParameters,
) -> Vec<PointCloud> {
    match cloud.buffer {
        Some(buff) => {
            if buff.len() < parameters.min_points_per_cluster {
                return Vec::new(); // no clusters, TODO maybe return the input cloud?
            }
        }
        None => {}
    };

    #[cfg(all(feature = "ros", feature = "cpu"))]
    let points: Vec<PointXYZ> = cloud.ros_cloud.unwrap().try_into_vec().unwrap();

    #[cfg(all(feature = "ros", feature = "cpu", not(feature = "rayon")))]
    let it = points.iter();
    #[cfg(all(feature = "ros", feature = "cpu", feature = "rayon"))]
    let it = points.par_iter();

    #[cfg(all(not(feature = "ros"), feature = "cpu", feature = "rayon"))]
    let it = cloud.as_slice().par_iter();

    #[cfg(all(not(feature = "ros"), feature = "cpu", not(feature = "rayon")))]
    let it = cloud.it;

    #[cfg(all(not(feature = "ros"), feature = "rayon", feature = "cpu"))]
    let it = cloud.as_slice().par_iter();
    #[cfg(all(not(feature = "ros"), feature = "cpu", not(feature = "rayon")))]
    let it = cloud.it;

    let data: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, _> = ndarray::ArrayBase::from_shape_vec(
        (points.len(), 3),
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
                let p = points[*idx].clone();
                cluster_points.push(p);
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
        cross_float3(u, p).0 * s * 2.0,
        cross_float3(u, p).1 * s * 2.0,
        cross_float3(u, p).2 * s * 2.0,
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

pub fn passthrough_filter(
    input: PointCloud2Msg,
    params: PassthroughFilterParameters,
) -> PointCloud2Msg {
    #[cfg(all(feature = "ros", feature = "cpu", not(feature = "rayon")))]
    let it = input.try_into_iter().unwrap(); // TODO handle error
    #[cfg(all(feature = "ros", feature = "cpu", feature = "rayon"))]
    let it = input.try_into_par_iter().unwrap();

    #[cfg(all(not(feature = "ros"), feature = "rayon", feature = "cpu"))]
    let it = input.buffer.into_par_iter();

    #[cfg(all(not(feature = "ros"), feature = "cpu", not(feature = "rayon")))]
    let it = input.it;

    let res = it.filter(move |point: &PointXYZ| {
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
    });

    #[cfg(feature = "rayon")]
    {
        ros_pointcloud2::PointCloud2Msg::try_from_par_iter(res).unwrap()
    }
    #[cfg(not(feature = "rayon"))]
    {
        ros_pointcloud2::PointCloud2Msg::try_from_iter(res).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::*;

    #[test]
    fn cluster() {
        let cloud: Vec<PointXYZI> = read_pcl("test_files/cluster_sample.pcd".to_string());
        let pointcloud = PointCloud::from_full_cloud(cloud);

        let parameters = EuclideanClusterParameters {
            tolerance: 0.02,
            min_points_per_cluster: 50,
        };

        let clusters = euclidean_cluster(pointcloud, &parameters);
        assert!(!clusters.is_empty());

        for cluster in clusters {
            let pcl_out: Vec<PointXYZI> = cluster.it.unwrap().collect();
            assert!(pcl_out.len() > parameters.min_points_per_cluster as usize);
        }
    }

    #[test]
    fn passthrough() {
        let mut cloud: Vec<PointXYZI> = generate_random_pointcloud(1000, 0.1, 1.0);

        let cloud_size = cloud.len();
        let pointcloud = PointCloud::from_full_cloud(cloud.clone());
        let params = PassthroughFilterParameters::default();
        let filtered = passthrough_filter(pointcloud, params);
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), cloud_size);
        equal_list_with_linear_search(cloud.as_slice(), out_pcl.as_slice());

        cloud.push(PointXYZI::new(0.0, 0.0, 0.0, 0.0));
        cloud.push(PointXYZI::new(0.0, 0.0, 0.0, 0.0));

        let cloud_size = cloud.len();

        let pointcloud = PointCloud::from_full_cloud(cloud.clone());
        let mut params = PassthroughFilterParameters::default();

        params.min = (-1.0, -1.0, -1.0);
        params.max = (1.0, 1.0, 1.0);
        params.min_dist = 0.05;
        let filtered = passthrough_filter(pointcloud, params.clone());
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), cloud_size - 2);

        params.invert_distance = true;
        let filtered =
            passthrough_filter(PointCloud::from_full_cloud(cloud.clone()), params.clone());
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 2);

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
        let filtered = passthrough_filter(pointcloud, params.clone());
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 3);
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

        let filtered = passthrough_filter(pointcloud, params.clone());
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 1);
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

        let filtered = voxel_downsample_average(pointcloud, params);
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 1);
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

        let filtered = voxel_downsample_median(pointcloud, params);
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 1);
    }

    #[test]
    /*fn voxel_downsample_center_test() {
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

        let filtered = voxel_downsample_center(pointcloud, params);
        let out_pcl: Vec<PointXYZI> = filtered.it.collect();
        assert_eq!(out_pcl.len(), 4);

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

        let filtered_2 = voxel_downsample_center(pointcloud_2, params_2);
        let out_pcl: Vec<PointXYZI> = filtered_2.it.collect();
        assert_eq!(out_pcl.len(), 1);
    }*/
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

        let filtered = voxel_downsample_center_par(pointcloud, params);
        let out_pcl: Vec<PointXYZI> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 4);

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

        let filtered_2 = voxel_downsample_center_par(pointcloud_2, params_2);
        let out_pcl: Vec<PointXYZ> = filtered_2.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 1);
    }

    #[test]
    fn intensity_passthrough() {
        let pointcloud = PointCloud::from_full_cloud(vec![
            PointXYZ::new(1.0, 0.0, 0.0),
            PointXYZ::new(1.0, 0.0, 0.0),
            PointXYZ::new(1.0, 0.0, 0.0),
            PointXYZ::new(1.0, 0.0, 0.0),
            PointXYZ::new(1.0, 0.0, 0.0),
        ]);
        let mut params = PassthroughFilterParameters::default();
        params.min_intensity = 1.0;
        params.max_intensity = 3.0;

        let filtered = passthrough_filter(pointcloud, params);
        let out_pcl: Vec<PointXYZ> = filtered.it.unwrap().collect();
        assert_eq!(out_pcl.len(), 3);
        assert_eq!(out_pcl[0].intensity, 1.0);
        assert_eq!(out_pcl[1].intensity, 2.0);
        assert_eq!(out_pcl[2].intensity, 3.0);
    }
}
