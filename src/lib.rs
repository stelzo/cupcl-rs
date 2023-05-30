#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)] // did only happen in auto generated bindgen tests

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point {
    x: f32,
    y: f32,
    z: f32,
    i: f32, // intensity or padding byte
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
            buffer: CudaBuffer::new(stream, std::mem::size_of::<Point>(), n)
        }
    }

    pub fn from_full_cloud(stream: &CudaStream, pointcloud: Vec<Point>) -> Self {
        Self {
            buffer: CudaBuffer::from_vec(stream, pointcloud, std::mem::size_of::<Point>())
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
        Self {
            stream,
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
       unsafe { cupcl_destroy_stream(self.stream); };
    }
}

pub fn voxel_downsample(stream: &CudaStream, input: &CudaPointCloud, voxel_size: (f32, f32, f32)) -> CudaPointCloud {
    let filter = unsafe { cupcl_init_voxel_filter(stream.stream, voxel_size.0, voxel_size.1, voxel_size.2) };
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

pub fn euclidean_cluster(stream: &CudaStream, cloud: &CudaPointCloud, voxel_size: f32, min_points: u32, max_points: u32, count_threshold: i32) -> Vec<Vec<Point>> {
    let cluster_instance = unsafe { cupcl_init_extract_cluster(stream.stream, min_points, max_points, voxel_size, voxel_size, voxel_size, count_threshold) };
    let output = CudaPointCloud::from_full_cloud(stream, cloud.as_slice().to_vec());
    let index_obj = CudaPointCloud::new(stream, cloud.buffer.n as usize);
    let index = index_obj.buffer.as_ptr();
    let ret = unsafe {
        cupcl_extract_cluster(cluster_instance, stream.stream, cloud.buffer.as_ptr(), cloud.buffer.n as i32, output.buffer.as_ptr(), index)
    };
    assert_eq!(ret, 0);

    unsafe {
        cupcl_free_extract_cluster(cluster_instance);
    }

    let n_clusters: u32 = unsafe { *index.offset(0) };
    let mut clusters = Vec::with_capacity(n_clusters as usize);
    let buff_float_ptr = output.buffer.as_ptr::<f32>();

    for i in 0..n_clusters {
        let cluster_size = unsafe { *index.offset(i as isize + 1) };
        let mut cluster = Vec::with_capacity(cluster_size as usize);
        let mut outoff = 0;
        let end_outoff = unsafe { *index.offset(i as isize) };
        for j in 1..end_outoff {
            outoff += unsafe { *index.offset(j as isize) };
        }
        for j in 0..cluster_size {
            let point_x = unsafe { *buff_float_ptr.offset((outoff + j) as isize) };
            let point_y = unsafe { *buff_float_ptr.offset((outoff + j) as isize + 1) };
            let point_z = unsafe { *buff_float_ptr.offset((outoff + j) as isize + 2) };
            cluster.push(Point::new(point_x, point_y, point_z, 0.0)); // TODO check if intensity is also saved
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
        let gpu_ptr = unsafe { cupcl_create_empty_buffer(stream.stream, element_size as u32, n as u32) };
        Self {
            gpu_ptr,
            element_size,
            n,
        }
    }

    pub fn from_vec<T>(stream: &CudaStream, vec: Vec<T>, element_size: usize) -> Self {
        let gpu_ptr = unsafe { cupcl_create_buffer(stream.stream, vec.as_ptr() as *mut ::std::os::raw::c_void, element_size as u32, vec.len() as u32) };
        Self {
            gpu_ptr,
            element_size,
            n: vec.len(),
        }
    }

    pub fn read_value<T: Copy>(&self) -> &T {
        assert!(self.element_size * self.n == std::mem::size_of::<T>());
        unsafe { &*(self.gpu_ptr as *mut T) }
    }

    pub fn as_ptr<T>(&self) -> *mut T {
        self.gpu_ptr as *mut T
    }
}

pub fn passthrough_filter(
    stream: &CudaStream,
    input: &CudaBuffer,
    min: (f32, f32, f32),
    max: (f32, f32, f32),
    invert_bounding_box: bool,
    min_dist: f32,
    max_dist: f32,
    invert_distance: bool,
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
            min_dist,
            max_dist,
            min.0,
            min.1,
            min.2,
            max.0,
            max.1,
            max.2,
            invert_bounding_box,
            invert_distance,
            output.buffer.as_ptr(),
            out_size.as_ptr(),
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
    use std::io::BufReader;
    use std::io::BufRead;

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

    #[test]
    fn passthrough() {
        let mut cloud = generate_random_pointcloud(1000, 0.1, 1.0);
        cloud.push(Point::new(0.0, 0.0, 0.0, 0.0));
        cloud.push(Point::new(0.0, 0.0, 0.0, 0.0));

        let cloud_size = cloud.len();

        let stream = CudaStream::new();
        let cuda_pointcloud = CudaPointCloud::from_full_cloud(&stream, cloud.clone());
        let min = (-1.0, -1.0, -1.0);
        let max = (1.0, 1.0, 1.0);
        let filtered = passthrough_filter(&stream, &cuda_pointcloud.buffer, min, max, false, 0.05, f32::MAX, false);
        assert_eq!(filtered.buffer.n, cloud_size - 2);

        let filtered = passthrough_filter(&stream, &cuda_pointcloud.buffer, min, max, false, 0.05, f32::MAX, true);
        assert_eq!(filtered.buffer.n, 2);
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
            let point = Point::new(rng.gen_range(min..max), rng.gen_range(min..max), rng.gen_range(min..max), 0.0);
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
        let clusters = euclidean_cluster(&stream, &cuda_pointcloud, voxel_size, min_points, max_points, count_threshold);
        assert_eq!(clusters.len(), 4);

        for cluster in clusters {
            assert!(cluster.len() > min_points as usize);
        }
    }
}

