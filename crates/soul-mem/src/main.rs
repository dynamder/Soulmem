//! 服务进程入口（薄）：加载配置 → 启动 → 优雅退出。
//! 业务逻辑均在 lib 内，便于集成测试直接驱动。

use soul_mem::config::Config;
use soul_mem::server::RunningServer;

fn main() {
    env_logger_init();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to build tokio runtime: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = runtime.block_on(async_main()) {
        log::error!("soul-mem exited with error: {e}");
        std::process::exit(1);
    }
}

async fn async_main() -> anyhow::Result<()> {
    let config = Config::from_env().map_err(anyhow::Error::from)?;
    RunningServer::run(config)
        .await
        .map_err(anyhow::Error::from)
}

/// 简易 env_logger 初始化（若已由其它库初始化则忽略）。
fn env_logger_init() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = env_logger_builder().try_init();
    });
}

fn env_logger_builder() -> env_logger::Builder {
    let mut builder = env_logger::Builder::from_default_env();
    builder.filter_level(log::LevelFilter::Info);
    builder.format_timestamp_millis();
    builder
}
