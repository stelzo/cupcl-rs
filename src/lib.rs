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
use ros_pointcloud2::points::PointXYZ;

pub trait Point: Default + Clone + Copy + Sync + Send + PartialEq + 'static {
    fn get_x(&self) -> f32;
    fn get_y(&self) -> f32;
    fn get_z(&self) -> f32;
    fn get_i(&self) -> f32;

    fn set_x(&mut self, x: f32);
    fn set_y(&mut self, y: f32);
    fn set_z(&mut self, z: f32);
    fn set_i(&mut self, i: f32);

    fn with_xyzi(x: f32, y: f32, z: f32, i: f32) -> Self;
    fn with_xyzif64(x: f64, y: f64, z: f64, i: f64) -> Self;
}

impl Point for PointXYZ {
    fn get_x(&self) -> f32 {
        self.x
    }

    fn get_y(&self) -> f32 {
        self.y
    }

    fn get_z(&self) -> f32 {
        self.z
    }

    fn get_i(&self) -> f32 {
        0.0
    }

    fn set_x(&mut self, x: f32) {
        self.x = x;
    }

    fn set_y(&mut self, y: f32) {
        self.y = y;
    }

    fn set_z(&mut self, z: f32) {
        self.z = z;
    }

    fn set_i(&mut self, i: f32) {
    }

    fn with_xyzi(x: f32, y: f32, z: f32, i: f32) -> Self {
        Self {
            x,
            y,
            z,
        }
    }

    fn with_xyzif64(x: f64, y: f64, z: f64, i: f64) -> Self {
        Self {
            x: x as f32,
            y: y as f32,
            z: z as f32,
        }
    }
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

pub struct PointCloud {
    #[cfg(feature = "cpu")]
    pub buffer: Option<Vec<PointXYZ>>,

    pub it: Option<Box<dyn Iterator<Item = PointXYZ>>>,

    #[cfg(feature = "ros")]
    pub ros_cloud: Option<ros_pointcloud2::PointCloud2Msg>,

    #[cfg(feature = "cuda")]
    pub buffer: CudaBuffer<T, U>,
}

impl PointCloud {
    #[cfg(feature = "cpu")]
    pub fn from_full_cloud(pointcloud: Vec<PointXYZ>) -> Self {
        Self {
            buffer: None,
            it: Some(Box::new(pointcloud.into_iter())),
            ros_cloud: None,
        }
    }

    pub fn from_iterable<I: IntoIterator<Item = PointXYZ> + 'static>(iter: I) -> Self {
        Self {
            buffer: None,
            ros_cloud: None,
            it: Some(Box::new(iter.into_iter())),
        }
    }

    #[cfg(feature = "ros")]
    #[inline]
    pub fn from_ros_cloud(cloud: ros_pointcloud2::PointCloud2Msg) -> Self {
        Self {
            buffer: None,
            it: None,
            ros_cloud: Some(cloud),
        }
    }

    #[cfg(feature = "cpu")]
    pub fn as_slice(&self) -> Option<&[PointXYZ]> {
        self.buffer.as_ref().map(|v| v.as_slice()) // TODO not working when ROS is enabled or build from iterator
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ros")]
    #[test]
    fn test_ros() {
        use ros_pointcloud2::prelude::*;

        let cloud = vec![
            PointXYZ {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            PointXYZ {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
        ];

        // somewhere else
        let internal_msg = PointCloud2Msg::try_from_vec(cloud).unwrap();

        // describe transformation to internal format
        let mut pointcloud = PointCloud::from_ros_cloud(internal_msg);
    }
}

#[derive(Debug, Clone)]

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
