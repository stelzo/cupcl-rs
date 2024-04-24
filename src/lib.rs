#[cfg(test)]
mod io;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use crate::cuda::*;

#[cfg(feature = "cpu")]
mod cpu;

#[cfg(feature = "cpu")]
pub use crate::cpu::*;

use num_traits::{Float, Zero};

pub trait Point<T>: Zero + Clone + Copy + 'static + Sync + Send + PartialEq {
    fn get_x(&self) -> T;
    fn get_y(&self) -> T;
    fn get_z(&self) -> T;
    fn get_i(&self) -> T;

    fn set_x(&mut self, x: T);
    fn set_y(&mut self, y: T);
    fn set_z(&mut self, z: T);
    fn set_i(&mut self, i: T);

    fn with_xyzi(x: T, y: T, z: T, i: T) -> Self;
    fn with_xyzif64(x: f64, y: f64, z: f64, i: f64) -> Self;
}

pub enum VoxelDownsampleStrategy {
    Center,
    Average,
    Median,
}

impl Default for VoxelDownsampleStrategy {
    fn default() -> Self {
        Self::Median
    }
}

#[derive(Debug, Clone)]
pub struct PointCloud<T, U>
where
    U: Float + Into<f64>,
    T: Point<U>,
{
    #[cfg(feature = "cpu")]
    pub buffer: Vec<T>,

    #[cfg(feature = "cuda")]
    pub buffer: CudaBuffer<T, U>,

    _phantom_t: std::marker::PhantomData<U>,
}

impl<T, U> PointCloud<T, U>
where
    U: Float + Into<f64>,
    T: Point<U>,
{
    #[cfg(feature = "cpu")]
    pub fn new(n: usize) -> Self {
        Self {
            buffer: vec![T::zero(); n],
            _phantom_t: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "cpu")]
    pub fn from_full_cloud(pointcloud: Vec<T>) -> Self {
        Self {
            buffer: pointcloud,
            _phantom_t: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "cpu")]
    pub fn as_slice(&self) -> &[T] {
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

#[derive(Debug)]

pub struct PassthroughFilterParameters {
    pub min: (f64, f64, f64),
    pub max: (f64, f64, f64),
    pub invert_bounding_box: bool,
    pub min_dist: f64,
    pub max_dist: f64,
    pub invert_distance: bool,
    pub rotation: (f64, f64, f64, f64),
    pub translation: (f64, f64, f64),
    pub fov_right: f64,
    pub fov_left: f64,
    pub forward: (f64, f64),
    pub enable_horizontal_fov: bool,

    #[cfg(feature = "cuda")]
    pub polygon: Option<CudaBuffer>,

    #[cfg(feature = "cpu")]
    pub polygon: Option<Vec<(f64, f64)>>,

    pub invert_polygon: bool,
    pub invert_fov: bool,
    pub min_intensity: f64,
    pub max_intensity: f64,
    pub invert_intensity: bool,
}

impl Default for PassthroughFilterParameters {
    fn default() -> Self {
        Self {
            min: (f64::MIN, f64::MIN, f64::MIN),
            max: (f64::MAX, f64::MAX, f64::MAX),
            invert_bounding_box: false,
            min_dist: 0.0,
            max_dist: f64::MAX / 2.5,
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
            min_intensity: f64::MIN,
            max_intensity: f64::MAX,
            invert_intensity: false,
        }
    }
}
