use std::env::consts::OS;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    let profile = std::env::var("PROFILE").unwrap_or("Release".parse().unwrap());
    let profile = if profile.eq_ignore_ascii_case("debug") {
        "Debug"
    } else {
        "Release"
    };

    let cuda_arch = std::env::var("CUDA_ARCH").unwrap_or("all".parse().unwrap());
    let mut dst = Config::new("..")
        .define("CMAKE_BUILD_TYPE", profile)
        .define("WITH_TEST", "OFF")
        .define("CMAKE_CUDA_ARCHITECTURES", cuda_arch)
        .cxxflag("-O3")
        .build();
    dst.push("lib");
    println!("cargo:rustc-link-search=native={}", dst.display());

    //Default cuda path on linux.
    let cuda_path = if std::env::var("CUDA_PATH").is_ok() {
        PathBuf::from(std::env::var("CUDA_PATH").unwrap())
    } else {
        //Assume that the default installation path of CUDA for non-Windows operating systems is the same as that for Linux.
        PathBuf::from("/usr/local/cuda")
    };
    let mut cuda_lib_path = cuda_path.clone();
    if OS == "windows" {
        cuda_lib_path.push("lib");
        cuda_lib_path.push("x64");
    } else {
        cuda_lib_path.push("lib64");
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=static=spacemesh-cuda");
}
