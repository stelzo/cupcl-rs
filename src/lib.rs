#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)] // did only happen in auto generated bindgen tests

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub i: f32, // intensity or padding byte
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < 0.0001
            && (self.y - other.y).abs() < 0.0001
            && (self.z - other.z).abs() < 0.0001
            && (self.i - other.i).abs() < 0.0001
    }
}

impl Point {
    pub fn new(x: f32, y: f32, z: f32, i: f32) -> Self {
        Self { x, y, z, i }
    }
}

pub struct CudaPointCloud {
    pub buffer: CudaBuffer,
}

impl CudaPointCloud {
    pub fn new(stream: &CudaStream, n: usize) -> Self {
        Self {
            buffer: CudaBuffer::new(stream, std::mem::size_of::<Point>(), n),
        }
    }

    pub fn from_full_cloud(stream: &CudaStream, pointcloud: Vec<Point>) -> Self {
        Self {
            buffer: CudaBuffer::from_vec(stream, pointcloud, std::mem::size_of::<Point>()),
        }
    }

    /**
     * Assuming the cloud is float4 which is default output of all API functions.
     */
    pub fn as_slice(&self) -> &[Point] {
        unsafe { std::slice::from_raw_parts(self.buffer.gpu_ptr as *const Point, self.buffer.n) }
    }
}

pub struct CudaStream {
    pub stream: *mut ::std::os::raw::c_void,
}

impl CudaStream {
    pub fn new() -> Self {
        let stream = unsafe { cupcl_create_stream() };
        Self { stream }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cupcl_destroy_stream(self.stream);
        };
    }
}

pub fn voxel_downsample(
    stream: &CudaStream,
    input: &CudaPointCloud,
    voxel_size: (f32, f32, f32),
) -> CudaPointCloud {
    let filter =
        unsafe { cupcl_init_voxel_filter(stream.stream, voxel_size.0, voxel_size.1, voxel_size.2) };
    let mut output = CudaPointCloud::new(stream, input.buffer.n);
    output.buffer.n = unsafe {
        cupcl_voxel_filter(
            filter,
            stream.stream,
            input.buffer.as_ptr(),
            input.buffer.n as u32,
            output.buffer.as_ptr(),
        )
    } as usize;
    unsafe {
        cupcl_free_voxel_filter(filter);
    }
    output
}

pub fn euclidean_cluster(
    stream: &CudaStream,
    cloud: &CudaPointCloud,
    voxel_size: f32,
    min_points: u32,
    max_points: u32,
    count_threshold: i32,
) -> Vec<Vec<Point>> {
    let cluster_instance = unsafe {
        cupcl_init_extract_cluster(
            stream.stream,
            min_points,
            max_points,
            voxel_size,
            voxel_size,
            voxel_size,
            count_threshold,
        )
    };
    let output = CudaPointCloud::from_full_cloud(stream, cloud.as_slice().to_vec());
    let index_obj = CudaPointCloud::new(stream, cloud.buffer.n as usize);
    let index = index_obj.buffer.as_ptr();
    let ret = unsafe {
        cupcl_extract_cluster(
            cluster_instance,
            stream.stream,
            cloud.buffer.as_ptr(),
            cloud.buffer.n as i32,
            output.buffer.as_ptr(),
            index,
        )
    };
    unsafe {
        cupcl_free_extract_cluster(cluster_instance);
    }

    if ret != 0 {
        println!(
            "Unknown Error in euclidean_cluster. We actually do not know what errors can occur."
        );
        return vec![];
    }

    let n_clusters: u32 = unsafe { *index };
    let mut clusters = Vec::with_capacity(n_clusters as usize);
    let buff_float_ptr = output.buffer.as_ptr::<f32>();

    for i in 1..n_clusters + 1 {
        let cluster_size = unsafe { *index.offset(i as isize) };
        let mut cluster = Vec::with_capacity(cluster_size as usize);
        let mut outoff = 0;
        for w in 1..i {
            if i > 1 {
                outoff += unsafe { *index.offset(w as isize) };
            }
        }
        for k in 0..cluster_size {
            let local_point_offset = ((outoff + k) * 4) as isize;
            let point_start_ptr = unsafe { buff_float_ptr.offset(local_point_offset) };
            let point_x = unsafe { *point_start_ptr };
            let point_y = unsafe { *point_start_ptr.offset(1) };
            let point_z = unsafe { *point_start_ptr.offset(2) };
            let point_i = unsafe { *point_start_ptr.offset(3) };
            cluster.push(Point::new(point_x, point_y, point_z, point_i)); // TODO check if intensity is also saved
        }
        clusters.push(cluster);
    }
    clusters
}

pub struct CudaBuffer {
    pub gpu_ptr: *mut ::std::os::raw::c_void,
    pub element_size: usize,
    pub n: usize,
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cupcl_free_buffer(self.gpu_ptr);
        }
    }
}

impl CudaBuffer {
    pub fn new(stream: &CudaStream, element_size: usize, n: usize) -> Self {
        let gpu_ptr =
            unsafe { cupcl_create_empty_buffer(stream.stream, element_size as u32, n as u32) };
        Self {
            gpu_ptr,
            element_size,
            n,
        }
    }

    pub fn upload(&mut self, stream: &CudaStream, buffer: &[u8]) {
        //assert_eq!(buffer.len(), self.element_size * self.n);
        unsafe {
            cupcl_upload_buffer(
                stream.stream,
                self.gpu_ptr,
                buffer.as_ptr() as *mut ::std::os::raw::c_void,
                self.element_size as u32,
                self.n as u32,
            )
        }
    }

    pub fn from_vec<T>(stream: &CudaStream, vec: Vec<T>, element_size: usize) -> Self {
        let gpu_ptr = unsafe {
            cupcl_create_buffer(
                stream.stream,
                vec.as_ptr() as *mut ::std::os::raw::c_void,
                element_size as u32,
                vec.len() as u32,
            )
        };
        Self {
            gpu_ptr,
            element_size,
            n: vec.len(),
        }
    }

    pub fn read_value<T: Copy>(&self) -> &T {
        debug_assert_eq!(self.element_size * self.n, std::mem::size_of::<T>());
        unsafe { &*(self.gpu_ptr as *mut T) }
    }

    pub fn as_ptr<T>(&self) -> *mut T {
        self.gpu_ptr as *mut T
    }
}

pub struct PassthroughFilterParameters {
    pub min: (f32, f32, f32),
    pub max: (f32, f32, f32),
    pub invert_bounding_box: bool,
    pub min_dist: f32,
    pub max_dist: f32,
    pub invert_distance: bool,
    pub rotation: (f32, f32, f32, f32),
    pub translation: (f32, f32, f32),
    pub fov_right: f32,
    pub fov_left: f32,
    pub forward: (f32, f32),
    pub invert_vertical_fov: bool,
    pub polygon: Option<CudaBuffer>,
    pub invert_polygon: bool,
}

impl Default for PassthroughFilterParameters {
    fn default() -> Self {
        Self {
            min: (f32::MIN, f32::MIN, f32::MIN),
            max: (f32::MAX, f32::MAX, f32::MAX),
            invert_bounding_box: false,
            min_dist: 0.0,
            max_dist: f32::MAX / 2.5,
            invert_distance: false,
            rotation: (0.0, 0.0, 0.0, 1.0),
            translation: (0.0, 0.0, 0.0),
            fov_right: 180.0,
            fov_left: -179.9999999,
            forward: (1.0, 0.0),
            invert_vertical_fov: false,
            polygon: None,
            invert_polygon: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

pub fn passthrough_filter(
    stream: &CudaStream,
    input: &CudaBuffer,
    params: &PassthroughFilterParameters,
) -> CudaPointCloud {
    let mut output = CudaPointCloud::new(stream, input.n);
    let out_size = CudaBuffer::new(stream, std::mem::size_of::<u32>(), 1);
    assert_ne!(input.gpu_ptr, std::ptr::null_mut());
    assert_ne!(output.buffer.gpu_ptr, std::ptr::null_mut());
    unsafe {
        cupcl_passthrough_filter(
            stream.stream,
            input.as_ptr(),
            input.n as u32,
            input.element_size as u32,
            params.min_dist,
            params.max_dist,
            params.min.0,
            params.min.1,
            params.min.2,
            params.max.0,
            params.max.1,
            params.max.2,
            params.invert_bounding_box,
            params.invert_distance,
            output.buffer.as_ptr(),
            out_size.as_ptr(),
            params.rotation.0,
            params.rotation.1,
            params.rotation.2,
            params.rotation.3,
            params.translation.0,
            params.translation.1,
            params.translation.2,
            params.fov_right.to_radians(),
            params.fov_left.to_radians(),
            params.forward.0,
            params.forward.1,
            params.invert_vertical_fov,
            params
                .polygon
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.as_ptr()),
            params.polygon.as_ref().map_or(0, |p| p.n as i32),
            params.invert_polygon,
        );
    }
    output.buffer.n = out_size.read_value::<u32>().clone() as usize;
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;

    pub struct PCDPoint {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    /*#[test]
    fn voxel() {
        let input = read_pcl("test_files/filter_sample.pcd".to_string());
        let input_n = input.len();
        assert_eq!(input_n, 119978);
        let stream = CudaStream::new();
        let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, input);
        let voxel_size = (1.0, 1.0, 1.0);
        let filtered = voxel_downsample(&stream, &cuda_pointcloud, voxel_size);
        assert_eq!(filtered.buffer.n, 3440);
    }*/

    fn equal_list_with_linear_search(lhs: &[Point], rhs: &[Point]) {
        assert_eq!(lhs.len(), rhs.len());
        for p in lhs {
            assert!(rhs.contains(p));
        }
    }

    #[test]
    fn passthrough() {
        let stream = CudaStream::new();
        let mut cloud = generate_random_pointcloud(1000, 0.1, 1.0);

        let cloud_size = cloud.len();
        let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, cloud.clone());
        let params = PassthroughFilterParameters::default();
        let filtered = passthrough_filter(&stream, &cuda_pointcloud.buffer, &params);
        let out_size = filtered.buffer.n;
        let back_to_vec = filtered.as_slice();
        assert_eq!(out_size, cloud_size);
        equal_list_with_linear_search(cloud.as_slice(), back_to_vec);

        cloud.push(Point::new(0.0, 0.0, 0.0, 0.0));
        cloud.push(Point::new(0.0, 0.0, 0.0, 0.0));

        let cloud_size = cloud.len();

        let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, cloud.clone());
        let mut params = PassthroughFilterParameters::default();

        params.min = (-1.0, -1.0, -1.0);
        params.max = (1.0, 1.0, 1.0);
        params.min_dist = 0.05;
        let filtered = passthrough_filter(&stream, &cuda_pointcloud.buffer, &params);
        assert_eq!(filtered.buffer.n, cloud_size - 2);

        params.invert_distance = true;
        let filtered = passthrough_filter(&stream, &cuda_pointcloud.buffer, &params);
        assert_eq!(filtered.buffer.n, 2);

        let cloud = vec![
            Point::new(0.5, 0.0, 0.0, 0.0),
            Point::new(0.0, 0.6, 0.0, 0.0),
            Point::new(0.0, 0.7, 6.0, 0.0),
            Point::new(0.0, 2.0, 0.0, 0.0),
        ];
        let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, cloud.clone());
        let mut params = PassthroughFilterParameters::default();
        let polygon = vec![
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(-1.0, 0.0),
            Point2::new(0.0, -1.0),
        ];
        let cuda_polygon =
            CudaBuffer::from_vec(&stream, polygon.clone(), std::mem::size_of::<Point2>());
        params.polygon = Some(cuda_polygon);
        let filtered = passthrough_filter(&stream, &cuda_pointcloud.buffer, &params);
        assert_eq!(filtered.buffer.n, 3);

        /*let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, vec![
            Point::new(1.0, 0.0, 0.0, 0.0),
            Point::new(1.0, 0.5, 0.0, 0.0),
        ]);
        let mut params = PassthroughFilterParameters::default();
        params.fov_left = 10.0;
        params.fov_right = -10.0;
        params.forward = (1.0, 0.0);

        let filtered = passthrough_filter(
            &stream,
            &cuda_pointcloud.buffer,
            &params,
        );
        assert_eq!(filtered.buffer.n, 1);*/
    }

    fn read_pcl_file(path: String) -> Vec<PCDPoint> {
        let mut data_start = false;
        let mut points = Vec::new();
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap();
            if data_start {
                let mut split = line.split_whitespace();
                let x = split.next().unwrap().parse::<f32>().unwrap();
                let y = split.next().unwrap().parse::<f32>().unwrap();
                let z = split.next().unwrap().parse::<f32>().unwrap();
                points.push(PCDPoint { x, y, z });
            } else if line == "DATA ascii" {
                data_start = true;
            }
        }
        points
    }

    fn read_pcl(path: String) -> Vec<Point> {
        let points = read_pcl_file(path);
        let mut pointcloud = Vec::with_capacity(points.len());
        for point in points {
            pointcloud.push(Point::new(point.x, point.y, point.z, 0.0));
        }
        pointcloud
    }

    fn generate_random_pointcloud(num_points: usize, min: f32, max: f32) -> Vec<Point> {
        let mut rng = rand::thread_rng();
        let mut pointcloud = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let point = Point::new(
                rng.gen_range(min..max),
                rng.gen_range(min..max),
                rng.gen_range(min..max),
                0.0,
            );
            pointcloud.push(point);
        }
        pointcloud
    }

    #[test]
    fn alloc_stream() {
        let stream = CudaStream::new();
        assert_ne!(stream.stream, std::ptr::null_mut());
    }

    #[test]
    fn read_pcl_test() {
        let points = read_pcl("test_files/cluster_sample.pcd".to_string());
        assert!(points.len() > 170000);
    }

    #[test]
    fn alloc_empty_buffer() {
        let stream = CudaStream::new();
        let pointcloud = CudaPointCloud::new(&stream, 1000);
        assert_ne!(pointcloud.buffer.gpu_ptr, std::ptr::null_mut());

        let buff_float_ptr = pointcloud.buffer.as_ptr::<f32>();

        let first_point_x = unsafe { *buff_float_ptr.offset(0) };
        let first_point_y = unsafe { *buff_float_ptr.offset(1) };
        let first_point_z = unsafe { *buff_float_ptr.offset(2) };
        let first_point_i = unsafe { *buff_float_ptr.offset(3) };

        assert_eq!(first_point_x, 0.0);
        assert_eq!(first_point_y, 0.0);
        assert_eq!(first_point_z, 0.0);
        assert_eq!(first_point_i, 0.0);

        let pointcloud = CudaPointCloud::new(&stream, 1000);
        assert_ne!(pointcloud.buffer.gpu_ptr, std::ptr::null_mut());
    }

    #[test]
    fn alloc_full_cloud() {
        let stream = CudaStream::new();
        let cloud = generate_random_pointcloud(1000, -1.0, 1.0);
        let pointcloud = CudaPointCloud::from_full_cloud(&stream, cloud.clone());
        assert_ne!(pointcloud.buffer.gpu_ptr, std::ptr::null_mut());

        let buff_float_ptr = pointcloud.buffer.as_ptr::<f32>();

        let first_point_x = unsafe { *buff_float_ptr.offset(0) };
        let first_point_y = unsafe { *buff_float_ptr.offset(1) };
        let first_point_z = unsafe { *buff_float_ptr.offset(2) };
        let first_point_i = unsafe { *buff_float_ptr.offset(3) };
        let second_point_x = unsafe { *buff_float_ptr.offset(4) };

        assert_eq!(first_point_x, cloud[0].x);
        assert_eq!(first_point_y, cloud[0].y);
        assert_eq!(first_point_z, cloud[0].z);
        assert_eq!(first_point_i, cloud[0].i);
        assert_eq!(second_point_x, cloud[1].x);

        let conv_back = pointcloud.as_slice().to_vec();
        assert_eq!(conv_back.len(), cloud.len());
        assert_eq!(conv_back, cloud);
    }

    #[test]
    fn cluster() {
        let stream = CudaStream::new();
        let cloud = read_pcl("test_files/cluster_sample.pcd".to_string());
        let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, cloud);

        let voxel_size = 0.05;
        let min_points = 100;
        let max_points = 2500000;
        let count_threshold = 20;
        let clusters = euclidean_cluster(
            &stream,
            &cuda_pointcloud,
            voxel_size,
            min_points,
            max_points,
            count_threshold,
        );
        assert_eq!(clusters.len(), 4);

        for cluster in clusters {
            assert!(cluster.len() > min_points as usize);
        }
    }
}
