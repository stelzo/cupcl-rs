# cupcl-rs

Rust bindings for CUDA point cloud processing with NVIDIA cuPCL.

** This is a work in progress. **

[ ] Voxel Downsample (libcudafilter.so) - See [#39](https://github.com/NVIDIA-AI-IOT/cuPCL/issues/39)
[x] Passthrough (own implementation)
[x] Clustering (libcudacluster.so)
[ ] ICP
[ ] NDT
[ ] Octree
[ ] Segmentation

# Setup

Install cuPCL from NVIDIA for your architecture.
```bash
git clone -b x86_64_lib|jp4.x|jp5.x https://github.com/NVIDIA-AI-IOT/cuPCL
sudo find cuPCL -name '*.so' -exec cp {} /usr/local/lib \;
sudo ldconfig
```

Check if the everything works before using it in your project.
```bash
cargo test
```


