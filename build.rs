#[cfg(feature = "cuda")]
extern crate bindgen;

#[cfg(feature = "cuda")]
use std::env;

#[cfg(feature = "cuda")]
use std::path::PathBuf;

#[cfg(feature = "cuda")]
pub fn link_cuda() {
    println!("cargo:rerun-if-changed=cluster.hpp");
    println!("cargo:rerun-if-changed=cluster.cpp");
    println!("cargo:rerun-if-changed=common.hpp");
    println!("cargo:rerun-if-changed=common.cpp");
    println!("cargo:rerun-if-changed=passthrough.cu");
    println!("cargo:rerun-if-changed=passthrough.hpp");
    println!("cargo:rerun-if-changed=wrapper.hpp");

    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib("stdc++")
        .files(["cluster.cpp", "common.cpp", "passthrough.cu"])
        .include("/usr/local/cuda/targets/aarch64-linux/include")
        .compile("cupcl");

    println!("cargo:rustc-link-lib=cupcl");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=cudacluster");
    println!("cargo:rustc-link-lib=cudafilter");
}

fn main() {
    let cpu_feature_enabled = cfg!(feature = "cpu");
    let cuda_feature_enabled = cfg!(feature = "cuda");

    if cpu_feature_enabled && cuda_feature_enabled {
        panic!("The features 'cpu' and 'cuda' cannot be enabled at the same time.");
    }
    
    #[cfg(feature = "cuda")]
    link_cuda()
}
