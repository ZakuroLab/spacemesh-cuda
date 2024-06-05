use cmake::Config;

fn main() {
    let profile = std::env::var("PROFILE").unwrap();
    let profile = match profile.as_str() {
        "debug" => "Debug",
        "release" => "Release",
        _ => "Release",
    };
    let mut dst = Config::new("..")
        .define("CMAKE_BUILD_TYPE", profile)
        .define("WITH_TEST", "OFF")
        .cxxflag("-O3")
        .build();
    dst.push("lib");
    println!("cargo:rustc-link-search=native={}", dst.display());

    let default_cuda_lib_path = "/usr/local/cuda/targets/x86_64-linux/lib/";
    let default_boost_path = "/usr/local/lib/";
    println!("cargo:rustc-link-search=native={}", default_cuda_lib_path);
    println!("cargo:rustc-link-search=native={}", default_boost_path);
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=static=spacemesh-cuda");
}
