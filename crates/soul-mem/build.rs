//! 构建脚本：用 prost-build 把 `proto/soul_mem.proto` 生成 Rust 消息类型到 OUT_DIR。
//!
//! 仅生成消息（不含 gRPC service）；传输为 zenoh pub/sub，载荷为 protobuf 二进制。
//! 使用 `protoc-bin-vendored` 自带编译器，避免本机安装 protoc。

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc_path = protoc_bin_vendored::protoc_bin_path().map_err(|e| e.to_string())?;
    // prost-build 通过 PROTOC 环境变量定位编译器。
    unsafe {
        std::env::set_var("PROTOC", &protoc_path);
    }

    let proto_dir = "proto";
    let proto_file = "proto/soul_mem.proto";
    println!("cargo:rerun-if-changed={proto_file}");
    println!("cargo:rerun-if-changed={proto_dir}");

    let mut config = prost_build::Config::new();
    config.compile_protos(&[proto_file], &[proto_dir])?;

    Ok(())
}
