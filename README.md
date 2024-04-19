# cupcl

Rust bindings for CUDA point cloud processing with NVIDIA cuPCL.

** This is a work in progress. **

- [ ] Voxel Downsample (libcudafilter.so) - See [#39](https://github.com/NVIDIA-AI-IOT/cuPCL/issues/39)
- [x] Passthrough (own implementation)
- [x] Clustering (libcudacluster.so)
- [ ] ICP
- [ ] NDT
- [ ] Octree
- [ ] Segmentation

# Setup

Install cuPCL from NVIDIA for your architecture.
```shell
git clone -b x86_64_lib|jp4.x|jp5.x https://github.com/NVIDIA-AI-IOT/cuPCL
sudo find cuPCL -name '*.so' -exec cp {} /usr/local/lib \;
sudo ldconfig
```

Check if the everything works before using it in your project.
```shell
cargo test # cpu
cargo test --no-default-features --features cuda
```

Use the `cuda` feature for acceleration:
```shell
[dependencies]
cupcl = { git = "https://github.com/stelzo/cupcl-rs", branch = "main", default-features = false, features = "cuda" }
```