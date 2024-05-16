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

pub trait Point<T>: Zero + Clone + Copy + Sync + Send + PartialEq + 'static {
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

pub struct PointCloud<T, U>
where
    U: Float + Into<f64>,
    T: Point<U>,
{
    #[cfg(feature = "cpu")]
    pub buffer: Option<Vec<T>>,

    pub it: Box<dyn Iterator<Item = T>>,

    #[cfg(feature = "cuda")]
    pub buffer: CudaBuffer<T, U>,

    _phantom_t: std::marker::PhantomData<U>,
}

impl<T, U> PointCloud<T, U>
where
    U: Float + Into<f64>,
    T: Point<U> + 'static,
{
    #[cfg(feature = "cpu")]
    pub fn from_full_cloud(pointcloud: Vec<T>) -> Self {
        Self {
            buffer: None,
            it: Box::new(pointcloud.into_iter()),
            _phantom_t: std::marker::PhantomData,
        }
    }

    pub fn from_iterable<I: IntoIterator<Item = T> + 'static>(iter: I) -> Self {
        Self {
            buffer: None,
            it: Box::new(iter.into_iter()),
            _phantom_t: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "ros")]
    #[inline]
    pub fn from_ros_cloud<
        const N: usize,
        C: ros_pointcloud2::PointConvertible<N>,
        F: Fn(C) -> T + 'static,
    >(
        iter: impl Iterator<Item = C> + 'static,
        conversion: F,
    ) -> Self {
        Self {
            buffer: None,
            it: Box::new(iter.map(conversion)),
            _phantom_t: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "cpu")]
    pub fn as_slice(&self) -> Option<&[T]> {
        self.buffer.as_ref().map(|v| v.as_slice())
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
    use ros_pointcloud2::points::PointXYZ;

    #[derive(pcl_derive::PointXYZ, Default, Debug, Clone, Copy, PartialEq)]
    struct MyPointXYZ {
        x: f32,
        y: f32,
        z: f32,
    }

    // TODO macro
    impl From<PointXYZ> for MyPointXYZ {
        fn from(p: PointXYZ) -> Self {
            Self {
                x: p.x,
                y: p.y,
                z: p.z,
            }
        }
    }

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
        let mut pointcloud = PointCloud::<MyPointXYZ, f32>::from_ros_cloud(
            internal_msg.try_into_iter().unwrap(),
            |p: PointXYZ| MyPointXYZ::from(p),
        );

        // work with pointcloud
        let first_iter = pointcloud.it.next().unwrap();
        assert_eq!(
            first_iter,
            MyPointXYZ {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );

        let second_iter = pointcloud.it.next().unwrap();
        assert_eq!(
            second_iter,
            MyPointXYZ {
                x: 4.0,
                y: 5.0,
                z: 6.0
            }
        );
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
