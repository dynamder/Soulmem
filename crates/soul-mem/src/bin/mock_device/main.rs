//! 单机“外部设备”模拟器：仅用 zenoh 订阅/发布，扮演另一台设备。
//!
//! 用法（先启动服务，再运行本模拟器；本机回环即“两台逻辑设备”）：
//! ```text
//! cargo run -p soul-mem --bin mock-device -- --zenoh-prefix soulmem
//! ```

mod scenario;

fn main() {
    env_logger_builder();
    let opts = match parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    if let Err(e) = runtime.block_on(run(opts)) {
        log::error!("mock-device failed: {e}");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone)]
struct Opts {
    zenoh_prefix: String,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    let mut zenoh_prefix = "soulmem".to_string();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--zenoh-prefix" => zenoh_prefix = args.next().ok_or("--zenoh-prefix needs a value")?,
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Opts { zenoh_prefix })
}

async fn run(opts: Opts) -> anyhow::Result<()> {
    let client = soul_mem::zenoh::ZenohClient::open(
        opts.zenoh_prefix.clone(),
        format!("mock-device-{}", std::process::id()),
    )
    .await
    .map_err(anyhow::Error::from)?;
    scenario::run(&client, &opts).await
}

fn env_logger_builder() {
    let mut builder = env_logger::Builder::from_default_env();
    builder.filter_level(log::LevelFilter::Info);
    builder.format_timestamp_millis();
    let _ = builder.try_init();
}
