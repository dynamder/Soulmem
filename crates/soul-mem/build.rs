//! 构建脚本：用 vendored protoc 编译 `proto/soulmem.proto`。
//!
//! 仓库约定不在系统里依赖 `protoc`（CI 与开发机都没有装），因此用
//! `protoc-bin-vendored` 提供二进制并通过 `PROTOC` 环境变量交给 prost/tonic。

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // SAFETY: 构建脚本单线程执行，设置 PROTOC 时不存在并发读取该环境变量的线程。
    unsafe { std::env::set_var("PROTOC", protoc) };

    // 本 crate 只做服务端，客户端 stub 由各调用方各自生成，故不生成 client。
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&["proto/soulmem.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/soulmem.proto");
    Ok(())
}
