#[cfg(test)]
mod io;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use crate::cuda::{
    euclidean_cluster, passthrough_filter, voxel_downsample, CudaBuffer, CudaStream,
};

#[cfg(feature = "cpu")]
mod cpu;

#[cfg(feature = "cpu")]
pub use crate::cpu::{euclidean_cluster, passthrough_filter, voxel_downsample};

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

#[derive(Debug, Clone)]
pub struct PointCloud {
    #[cfg(feature = "cpu")]
    pub buffer: Vec<Point>,

    #[cfg(feature = "cuda")]
    pub buffer: CudaBuffer,
}

impl PointCloud {
    #[cfg(feature = "cpu")]
    pub fn new(n: usize) -> Self {
        Self {
            buffer: vec![Point::new(0.0, 0.0, 0.0, 0.0); n],
        }
    }

    #[cfg(feature = "cpu")]
    pub fn from_full_cloud(pointcloud: Vec<Point>) -> Self {
        Self { buffer: pointcloud }
    }

    #[cfg(feature = "cpu")]
    pub fn as_slice(&self) -> &[Point] {
        &self.buffer
    }

    #[cfg(feature = "cuda")]
    pub fn new(stream: &CudaStream, n: usize) -> Self {
        Self {
            buffer: CudaBuffer::new(stream, std::mem::size_of::<Point>(), n),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_full_cloud(stream: &CudaStream, pointcloud: Vec<Point>) -> Self {
        Self {
            buffer: CudaBuffer::from_vec(stream, pointcloud, std::mem::size_of::<Point>()),
        }
    }

    /**
     * Assuming the cloud is float4 which is default output of all API functions.
     */
    #[cfg(feature = "cuda")]
    pub fn as_slice(&self) -> &[Point] {
        unsafe { std::slice::from_raw_parts(self.buffer.gpu_ptr as *const Point, self.buffer.n) }
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
    pub enable_horizontal_fov: bool,

    #[cfg(feature = "cuda")]
    pub polygon: Option<CudaBuffer>,

    #[cfg(feature = "cpu")]
    pub polygon: Option<Vec<(f32, f32)>>,

    pub invert_polygon: bool,
    pub invert_fov: bool,
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
            fov_right: 0.0,
            fov_left: 0.0,
            forward: (1.0, 0.0),
            enable_horizontal_fov: false,
            polygon: None,
            invert_polygon: false,
            invert_fov: false,
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
