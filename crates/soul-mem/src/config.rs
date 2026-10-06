//! 服务配置。
//!
//! - `db_path` 与 `key_prefix` 由环境变量提供：它们因部署而异，不适合写进随仓库分发的文件；
//! - **其余全部从 TOML 文件读取**，路径由 `SOULMEM_CONFIG` 指定（默认 `soulmem.toml`；
//!   文件不存在时整份取默认值）；
//! - 一个进程只服务一个角色/一个记忆库。
//! - zenoh keyexpr 前缀**含角色/库标识**（与 `proto/soulmem.proto` 一致）：未设置
//!   `SOULMEM_KEY_PREFIX` 时默认为 `soulmem/<character>`；因此「同名角色」的多个实例会
//!   共享前缀并重复响应，部署上须保证一个 `character` 只对应一个实例（或显式设置唯一的
//!   `SOULMEM_KEY_PREFIX`）。
//!
//! 配置在启动时构造一次，并以 `Arc<Config>` 注入各组件（见 `bootstrap` / `MemoryService`），
//! 运行期不再重建或克隆设置。
//!
//! 示例 `soulmem.toml`（字段可省略，省略即取默认值）：
//!
//! ```toml
//! character = "soulmem"
//! enable_zenoh = true
//! enable_grpc = true
//! grpc_addr = "127.0.0.1:50051"
//! window_capacity = 20
//! consolidate_interval_secs = 300
//! idle_quiet_secs = 120
//! heartbeat_interval_secs = 30
//! candidate_k = 32
//! top_k = 10
//! max_cluster_nodes = 2048
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{ServiceError, ServiceResult};

/// `SOULMEM_CONFIG` 未设置时的配置文件路径。
const DEFAULT_CONFIG_PATH: &str = "soulmem.toml";
/// `SOULMEM_DB_PATH` 未设置时的数据文件路径。
const DEFAULT_DB_PATH: &str = "soulmem.db";
/// zenoh keyexpr 根命名空间。未显式设置 `SOULMEM_KEY_PREFIX` 时，
/// 完整前缀为 `<DEFAULT_KEY_PREFIX>/<character>`（含角色/库标识）。
const DEFAULT_KEY_PREFIX: &str = "soulmem";
/// 默认角色名。
const DEFAULT_CHARACTER: &str = "soulmem";
/// 默认 gRPC 监听地址。
const DEFAULT_GRPC_ADDR: &str = "127.0.0.1:50051";
/// 默认滑动窗口容量。
const DEFAULT_WINDOW_CAPACITY: usize = 20;
/// 默认巩固触发间隔（秒）。
const DEFAULT_CONSOLIDATE_SECS: u64 = 300;
/// 默认"静默"判定阈值（秒）：超过它未收到增量/检索，维护任务才转入 Idle。
const DEFAULT_IDLE_QUIET_SECS: u64 = 120;
/// 默认心跳间隔（秒）。
const DEFAULT_HEARTBEAT_SECS: u64 = 30;
/// 默认数据库每槽位召回预算。
const DEFAULT_CANDIDATE_K: usize = 32;
/// 默认检索返回记忆条数上限。
const DEFAULT_TOP_K: usize = 10;
/// 默认工作记忆簇节点软上限（仅告警，不自动淘汰）。
const DEFAULT_MAX_CLUSTER_NODES: usize = 2048;

/// TOML 文件映射：未出现的字段一律取默认值。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    /// 角色名，同时作为 SurrealDB 的库名。
    character: String,
    /// 是否启动 zenoh 适配器。
    enable_zenoh: bool,
    /// 是否启动 gRPC 适配器。
    enable_grpc: bool,
    /// gRPC 监听地址。
    grpc_addr: SocketAddr,
    /// 滑动窗口容量。
    window_capacity: usize,
    /// Idle 时定时巩固的间隔（秒）。
    consolidate_interval_secs: u64,
    /// 静默多久后视为 Idle（秒）。
    idle_quiet_secs: u64,
    /// 心跳/事件推送间隔（秒）。
    heartbeat_interval_secs: u64,
    /// 数据库候选召回每槽位预算。
    candidate_k: usize,
    /// 检索返回记忆条数上限。
    top_k: usize,
    /// 工作记忆簇节点软上限。
    max_cluster_nodes: usize,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            character: DEFAULT_CHARACTER.to_string(),
            enable_zenoh: true,
            enable_grpc: true,
            //SAFEUNWRAP: 常量字面量，格式恒合法。
            grpc_addr: DEFAULT_GRPC_ADDR.parse().expect("默认 gRPC 地址恒合法"),
            window_capacity: DEFAULT_WINDOW_CAPACITY,
            consolidate_interval_secs: DEFAULT_CONSOLIDATE_SECS,
            idle_quiet_secs: DEFAULT_IDLE_QUIET_SECS,
            heartbeat_interval_secs: DEFAULT_HEARTBEAT_SECS,
            candidate_k: DEFAULT_CANDIDATE_K,
            top_k: DEFAULT_TOP_K,
            max_cluster_nodes: DEFAULT_MAX_CLUSTER_NODES,
        }
    }
}

/// 服务启动配置。
#[derive(Debug, Clone)]
pub struct Config {
    /// 角色名，同时作为 SurrealDB 的库名。
    pub character: String,
    /// SurrealDB 数据文件路径（环境变量 `SOULMEM_DB_PATH`）。
    pub db_path: PathBuf,
    /// zenoh keyexpr 前缀（`<prefix>/req` 等）。未设置 `SOULMEM_KEY_PREFIX` 时
    /// 默认为 `soulmem/<character>`（含角色/库标识，见模块文档）。
    pub key_prefix: String,
    /// 是否启动 zenoh 适配器。
    pub enable_zenoh: bool,
    /// 是否启动 gRPC 适配器。
    pub enable_grpc: bool,
    /// gRPC 监听地址。
    pub grpc_addr: SocketAddr,
    /// 滑动窗口容量。
    pub window_capacity: usize,
    /// Idle 时定时巩固的间隔（秒）。
    pub consolidate_interval_secs: u64,
    /// 静默多久后视为 Idle（秒），供定时维护使用。
    pub idle_quiet_secs: u64,
    /// 心跳/事件推送间隔（秒）。
    pub heartbeat_interval_secs: u64,
    /// 数据库候选召回每槽位预算。
    pub candidate_k: usize,
    /// 检索返回记忆条数上限。
    pub top_k: usize,
    /// 工作记忆簇节点软上限。
    ///
    /// 仅用于观测与告警：本服务**不自动淘汰**。膨胀的根因在 `soul-mem-algo` 的
    /// `prefetch_db`（它把数据库候选无条件并永久写入工作记忆簇），因此根治需要
    /// 在算法层引入暂存/淘汰策略，而非在服务层兜底。
    pub max_cluster_nodes: usize,
}

impl Config {
    /// 从 `SOULMEM_CONFIG`（默认 `soulmem.toml`）、`SOULMEM_DB_PATH`、`SOULMEM_KEY_PREFIX`
    /// 组装配置。TOML 文件不存在时整份取默认值。
    pub fn load() -> ServiceResult<Self> {
        let path = env_string("SOULMEM_CONFIG", DEFAULT_CONFIG_PATH);
        let file = load_file(Path::new(&path))?;
        let db_path = PathBuf::from(env_string("SOULMEM_DB_PATH", DEFAULT_DB_PATH));
        let key_prefix_override = env_opt("SOULMEM_KEY_PREFIX");
        Self::assemble(file, db_path, key_prefix_override)
    }

    /// 合并 TOML 内容与环境变量，并做取值校验。
    fn assemble(
        file: FileConfig,
        db_path: PathBuf,
        key_prefix_override: Option<String>,
    ) -> ServiceResult<Self> {
        if file.window_capacity == 0 {
            return Err(ServiceError::Config("window_capacity 必须大于 0".into()));
        }
        if file.top_k == 0 {
            return Err(ServiceError::Config("top_k 必须大于 0".into()));
        }
        if file.max_cluster_nodes == 0 {
            return Err(ServiceError::Config("max_cluster_nodes 必须大于 0".into()));
        }

        // 角色名会同时用于 SurrealDB 库名与 keyexpr 前缀，需保证是合法单段。
        let character = validate_character(&file.character)?;
        let key_prefix = match key_prefix_override {
            Some(raw) => validate_keyexpr(&raw)?,
            None => format!("{DEFAULT_KEY_PREFIX}/{character}"),
        };

        Ok(Self {
            character,
            db_path,
            key_prefix,
            enable_zenoh: file.enable_zenoh,
            enable_grpc: file.enable_grpc,
            grpc_addr: file.grpc_addr,
            window_capacity: file.window_capacity,
            consolidate_interval_secs: file.consolidate_interval_secs,
            idle_quiet_secs: file.idle_quiet_secs,
            heartbeat_interval_secs: file.heartbeat_interval_secs,
            candidate_k: file.candidate_k,
            top_k: file.top_k,
            max_cluster_nodes: file.max_cluster_nodes,
        })
    }
}

/// 读取并解析 TOML 配置；文件不存在按默认值处理。
fn load_file(path: &Path) -> ServiceResult<FileConfig> {
    match std::fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).map_err(|error| {
            ServiceError::Config(format!("解析配置文件 {} 失败: {error}", path.display()))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!(path = %path.display(), "配置文件不存在，使用默认值");
            Ok(FileConfig::default())
        }
        Err(error) => Err(ServiceError::Config(format!(
            "读取配置文件 {} 失败: {error}",
            path.display()
        ))),
    }
}

/// 读取字符串环境变量；未设置或空串取默认值。
fn env_string(key: &str, default: &str) -> String {
    env_opt(key).unwrap_or_else(|| default.to_string())
}

/// 读取可选环境变量；未设置或空串返回 `None`。
fn env_opt(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

/// 校验 keyexpr 前缀：非空、无首尾 `/`、无连续 `//`、无空白、不含 zenoh 通配符。
fn validate_keyexpr(raw: &str) -> ServiceResult<String> {
    if raw.is_empty() {
        return Err(ServiceError::Config("keyexpr 前缀不能为空".into()));
    }
    if raw.starts_with('/') || raw.ends_with('/') || raw.contains("//") {
        return Err(ServiceError::Config(format!("keyexpr 前缀含非法斜杠: {raw}")));
    }
    if raw.chars().any(char::is_whitespace) {
        return Err(ServiceError::Config(format!("keyexpr 前缀不能含空白: {raw}")));
    }
    if raw.chars().any(|c| matches!(c, '*' | '?' | '$' | '#')) {
        return Err(ServiceError::Config(format!("keyexpr 前缀不能含通配符: {raw}")));
    }
    Ok(raw.to_string())
}

/// 校验角色名：非空、无空白、不含 `/`（保证 `soulmem/<character>` 是干净两段）。
fn validate_character(raw: &str) -> ServiceResult<String> {
    if raw.is_empty() {
        return Err(ServiceError::Config("角色名不能为空".into()));
    }
    if raw.chars().any(char::is_whitespace) || raw.contains('/') {
        return Err(ServiceError::Config(format!(
            "角色名不能含空白或斜杠（将用于 keyexpr 前缀）: {raw}"
        )));
    }
    Ok(raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_defaults_when_fields_omitted() {
        let file: FileConfig = toml::from_str("").expect("空 TOML 应可解析");
        assert_eq!(file.character, DEFAULT_CHARACTER);
        assert!(file.enable_zenoh);
        assert!(file.enable_grpc);
        assert_eq!(file.window_capacity, DEFAULT_WINDOW_CAPACITY);
        assert_eq!(file.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn toml_overrides_are_applied() {
        let text = r#"
            character = "yuki"
            enable_zenoh = false
            window_capacity = 7
            top_k = 3
        "#;
        let file: FileConfig = toml::from_str(text).expect("应可解析");
        let config = Config::assemble(file, PathBuf::from("a.db"), Some("prefix".into()))
            .expect("应通过校验");
        assert_eq!(config.character, "yuki");
        assert!(!config.enable_zenoh);
        assert_eq!(config.window_capacity, 7);
        assert_eq!(config.top_k, 3);
        assert_eq!(config.db_path, PathBuf::from("a.db"));
        assert_eq!(config.key_prefix, "prefix");
    }

    #[test]
    fn zero_window_capacity_is_rejected() {
        let file: FileConfig = toml::from_str("window_capacity = 0").expect("应可解析");
        assert!(Config::assemble(file, PathBuf::from("a.db"), Some("p".into())).is_err());
    }

    #[test]
    fn zero_top_k_is_rejected() {
        let file: FileConfig = toml::from_str("top_k = 0").expect("应可解析");
        assert!(Config::assemble(file, PathBuf::from("a.db"), Some("p".into())).is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        assert!(toml::from_str::<FileConfig>("typo_field = 1").is_err());
    }

    #[test]
    fn default_key_prefix_includes_character() {
        let file: FileConfig = toml::from_str("character = \"yuki\"").expect("应可解析");
        let config =
            Config::assemble(file, PathBuf::from("a.db"), None).expect("应通过校验");
        assert_eq!(config.key_prefix, "soulmem/yuki");
    }

    #[test]
    fn explicit_key_prefix_overrides_character() {
        let file: FileConfig = toml::from_str("character = \"yuki\"").expect("应可解析");
        let config = Config::assemble(file, PathBuf::from("a.db"), Some("custom".into()))
            .expect("应通过校验");
        assert_eq!(config.key_prefix, "custom");
    }

    #[test]
    fn invalid_key_prefix_is_rejected() {
        let file: FileConfig = toml::from_str("").expect("应可解析");
        assert!(Config::assemble(file.clone(), PathBuf::from("a.db"), Some("a/*".into())).is_err());
        assert!(Config::assemble(file.clone(), PathBuf::from("a.db"), Some("/a".into())).is_err());
        assert!(Config::assemble(file.clone(), PathBuf::from("a.db"), Some("a b".into())).is_err());
    }

    #[test]
    fn invalid_character_is_rejected() {
        let file: FileConfig = toml::from_str("character = \"a/b\"").expect("应可解析");
        assert!(Config::assemble(file, PathBuf::from("a.db"), None).is_err());
    }
}
