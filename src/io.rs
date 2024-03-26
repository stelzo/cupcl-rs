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
    let cuda_pointcloud = PointCloud::from_full_cloud(&stream, input);
    let voxel_size = (1.0, 1.0, 1.0);
    let filtered = voxel_downsample(&stream, &cuda_pointcloud, voxel_size);
    assert_eq!(filtered.buffer.n, 3440);
}*/

pub fn equal_list_with_linear_search(lhs: &[Point], rhs: &[Point]) {
    assert_eq!(lhs.len(), rhs.len());
    for p in lhs {
        assert!(rhs.contains(p));
    }
}

pub fn read_pcl_file(path: String) -> Vec<PCDPoint> {
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

pub fn read_pcl(path: String) -> Vec<Point> {
    let points = read_pcl_file(path);
    let mut pointcloud = Vec::with_capacity(points.len());
    for point in points {
        pointcloud.push(Point::new(point.x, point.y, point.z, 0.0));
    }
    pointcloud
}

pub fn generate_random_pointcloud(num_points: usize, min: f32, max: f32) -> Vec<Point> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_pcl_test() {
        let points = read_pcl("test_files/cluster_sample.pcd".to_string());
        assert!(points.len() > 170000);
    }

    #[test]
    fn generate_random_pointcloud_test() {
        let input = generate_random_pointcloud(100, 0.0, 1.0);
        let input_n = input.len();
        assert_eq!(input_n, 100);
    }
}
