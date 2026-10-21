//! 服务配置。
//!
//! - 全部字段按所属部分分组从 TOML 文件读取；文件路径由调用方传入，文件不存在时整份取默认值；
//! - [`Config::load`] 只接收参数、不读取环境变量：路径由入口（`main`）决定；
//! - 一个进程只服务一个角色/一个记忆库；
//! - zenoh keyexpr 前缀**含角色/库标识**（与 `proto/soulmem.proto` 一致）：`[zenoh].key_prefix`
//!   未设置时默认为 `soulmem/<character>`；因此「同名角色」的多个实例会共享前缀并重复响应，
//!   部署上须保证一个 `character` 只对应一个实例（或显式设置唯一的 `key_prefix`）。
//!
//! 配置在启动时构造一次，并以 `Arc<Config>` 注入各组件（见 `bootstrap` / `MemoryService`），
//! 运行期不再重建或克隆设置。
//!
//! 示例 `soulmem.toml`（字段可省略，省略即取默认值）：
//!
//! ```toml
//! character = "soulmem"
//! db_path = "soulmem.db"
//!
//! [zenoh]
//! enable = true
//! # key_prefix = "soulmem/soulmem"
//!
//! [grpc]
//! enable = true
//! addr = "127.0.0.1:50051"
//!
//! [working_memory]
//! window_capacity = 20
//! max_cluster_nodes = 2048
//!
//! [retrieval]
//! candidate_k = 32
//! top_k = 10
//! similarity_threshold = 0.35
//! abstract_source_priority = 2.0
//! threads = 0                 # 0 = 自动（核数-1）
//! include_short_summary = true
//! # short_clipping_length = 8
//!
//! [retrieval.association]
//! damping_factor = 0.65
//! residue_threshold = 0.00001
//! top_k = 8
//!
//! [consolidation]
//! interval_secs = 300
//! idle_quiet_secs = 120
//! link_top_k = 3
//! link_verb = "related"
//! link_confidence = 1.0
//!
//! [service]
//! heartbeat_interval_secs = 30
//! event_channel_capacity = 64
//! command_buffer = 32
//! shutdown_grace_secs = 5
//!
//! [llm]
//! # base_url = "https://api.example.com/v1"
//! # model = "your-model"
//! # api_key = "..."          # 敏感信息：soulmem.toml 不应提交到仓库
//! probe_timeout_secs = 5
//! ```
//!
//! 仓库根另有可直接复制的 `soulmem.example.toml`。

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use secrecy::SecretString;
use serde::Deserialize;

use soul_mem_algo::algo::retrieve::association::AssociationConfig;

use crate::error::{ServiceError, ServiceResult};

/// 入口未显式传入配置路径时的默认值。
pub const DEFAULT_CONFIG_PATH: &str = "soulmem.toml";
/// 配置未提供 `db_path` 时的数据文件路径。
const DEFAULT_DB_PATH: &str = "soulmem.db";
/// zenoh keyexpr 根命名空间。未设置 `[zenoh].key_prefix` 时，
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
/// 默认相似度兜底分。
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.35;
/// 默认抽象情境源在 Bayes 动作提取中的权重倍率。
const DEFAULT_ABSTRACT_SOURCE_PRIORITY: f64 = 2.0;
/// 默认巩固建链时保留的活跃节点数量上限。
const DEFAULT_CONSOLIDATION_LINK_TOP_K: usize = 3;
/// 默认巩固建链谓词。
const DEFAULT_CONSOLIDATION_LINK_VERB: &str = "related";
/// 默认巩固建链置信度。
const DEFAULT_CONSOLIDATION_LINK_CONFIDENCE: f32 = 1.0;
/// 默认事件广播通道容量。
const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 64;
/// 默认服务命令通道容量。
const DEFAULT_COMMAND_BUFFER: usize = 32;
/// 默认退出宽限（秒）。
const DEFAULT_SHUTDOWN_GRACE_SECS: u64 = 5;
/// 默认 LLM 探活超时（秒）。
const DEFAULT_LLM_PROBE_TIMEOUT_SECS: u64 = 5;

/// `[zenoh]` 段。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZenohConfig {
    /// 是否启动 zenoh 适配器。
    pub enable: bool,
    /// keyexpr 前缀；`None` 时由 `load` 解析为 `soulmem/<character>`。
    pub key_prefix: Option<String>,
}

impl Default for ZenohConfig {
    fn default() -> Self {
        Self {
            enable: true,
            key_prefix: None,
        }
    }
}

/// `[grpc]` 段。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GrpcConfig {
    /// 是否启动 gRPC 适配器。
    pub enable: bool,
    /// 监听地址。
    pub addr: SocketAddr,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            enable: true,
            //SAFEUNWRAP: 常量字面量，格式恒合法。
            addr: DEFAULT_GRPC_ADDR.parse().expect("默认 gRPC 地址恒合法"),
        }
    }
}

/// `[working_memory]` 段。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkingMemoryConfig {
    /// 滑动窗口容量。
    pub window_capacity: usize,
    /// 工作记忆簇节点软上限：仅用于观测与告警，本服务**不自动淘汰**。膨胀的根因在
    /// `soul-mem-algo` 的 `prefetch_db`（它把数据库候选无条件并永久写入工作记忆簇），
    /// 因此根治需要在算法层引入暂存/淘汰策略，而非在服务层兜底。
    pub max_cluster_nodes: usize,
}

impl Default for WorkingMemoryConfig {
    fn default() -> Self {
        Self {
            window_capacity: DEFAULT_WINDOW_CAPACITY,
            max_cluster_nodes: DEFAULT_MAX_CLUSTER_NODES,
        }
    }
}

/// `[retrieval]` 段。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    /// 数据库候选召回每槽位预算。
    pub candidate_k: usize,
    /// 检索返回记忆条数上限（同时用作管线内相似度段的 max_results）。
    pub top_k: usize,
    /// 相似度兜底分：低于它的节点直接过滤。
    pub similarity_threshold: f32,
    /// 抽象情境源在 Bayes 动作提取中的权重倍率。
    pub abstract_source_priority: f64,
    /// 检索并行池线程数；`0` 表示自动（核数 - 1，至少 1）。
    pub threads: usize,
    /// 短期记忆是否附带摘要。
    pub include_short_summary: bool,
    /// 短期记忆裁剪长度；`None` 表示不裁剪。
    pub short_clipping_length: Option<usize>,
    /// PPR 联想扩散参数。
    pub association: AssociationConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            candidate_k: DEFAULT_CANDIDATE_K,
            top_k: DEFAULT_TOP_K,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            abstract_source_priority: DEFAULT_ABSTRACT_SOURCE_PRIORITY,
            threads: 0,
            include_short_summary: true,
            short_clipping_length: None,
            association: AssociationConfig::default(),
        }
    }
}

/// `[consolidation]` 段。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsolidationConfig {
    /// Idle 时定时巩固的间隔（秒）。
    pub interval_secs: u64,
    /// 静默多久后视为 Idle（秒）。
    pub idle_quiet_secs: u64,
    /// 巩固建链时保留的活跃节点数量上限。
    pub link_top_k: usize,
    /// 巩固建链谓词。
    pub link_verb: String,
    /// 巩固建链置信度。
    pub link_confidence: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            interval_secs: DEFAULT_CONSOLIDATE_SECS,
            idle_quiet_secs: DEFAULT_IDLE_QUIET_SECS,
            link_top_k: DEFAULT_CONSOLIDATION_LINK_TOP_K,
            link_verb: DEFAULT_CONSOLIDATION_LINK_VERB.to_string(),
            link_confidence: DEFAULT_CONSOLIDATION_LINK_CONFIDENCE,
        }
    }
}

/// `[service]` 段：进程级工程参数。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceConfig {
    /// 心跳/事件推送间隔（秒）。
    pub heartbeat_interval_secs: u64,
    /// 事件广播通道容量。
    pub event_channel_capacity: usize,
    /// 适配器到服务循环的命令通道容量。
    pub command_buffer: usize,
    /// 退出时等待服务循环排空在途命令的宽限（秒）。
    pub shutdown_grace_secs: u64,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: DEFAULT_HEARTBEAT_SECS,
            event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
            command_buffer: DEFAULT_COMMAND_BUFFER,
            shutdown_grace_secs: DEFAULT_SHUTDOWN_GRACE_SECS,
        }
    }
}

/// `[llm]` 段。
///
/// 缺少 `base_url` / `model` 时服务照常启动，LLM 降级为不可用；`api_key` 属于敏感信息，
/// 只从配置文件读取，`Debug` 输出会脱敏。
#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    /// OpenAI-compatible API base。
    pub base_url: String,
    /// 模型名。
    pub model: String,
    /// 密钥；`None` 表示不发鉴权头。`SecretString` 会在 `Debug` 中脱敏、drop 时 zeroize。
    pub api_key: Option<SecretString>,
    /// 手动触发探活时的等待上限（秒）。
    pub probe_timeout_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            api_key: None,
            probe_timeout_secs: DEFAULT_LLM_PROBE_TIMEOUT_SECS,
        }
    }
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("has_api_key", &self.api_key.is_some())
            .field("probe_timeout_secs", &self.probe_timeout_secs)
            .finish()
    }
}

/// TOML 文件映射：未出现的字段一律取默认值。
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    /// 角色名，同时作为 SurrealDB 的库名。
    character: String,
    /// SurrealDB 数据文件路径。
    db_path: PathBuf,
    zenoh: ZenohConfig,
    grpc: GrpcConfig,
    working_memory: WorkingMemoryConfig,
    retrieval: RetrievalConfig,
    consolidation: ConsolidationConfig,
    service: ServiceConfig,
    llm: LlmConfig,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            character: DEFAULT_CHARACTER.to_string(),
            db_path: PathBuf::from(DEFAULT_DB_PATH),
            zenoh: ZenohConfig::default(),
            grpc: GrpcConfig::default(),
            working_memory: WorkingMemoryConfig::default(),
            retrieval: RetrievalConfig::default(),
            consolidation: ConsolidationConfig::default(),
            service: ServiceConfig::default(),
            llm: LlmConfig::default(),
        }
    }
}

/// 服务启动配置。
#[derive(Debug, Clone)]
pub struct Config {
    /// 角色名，同时作为 SurrealDB 的库名。
    pub character: String,
    /// SurrealDB 数据文件路径（TOML `db_path` 字段）。
    pub db_path: PathBuf,
    /// zenoh 适配器设置；`key_prefix` 已在 `load` 中解析为完整前缀。
    pub zenoh: ZenohConfig,
    /// gRPC 适配器设置。
    pub grpc: GrpcConfig,
    /// 工作记忆设置。
    pub working_memory: WorkingMemoryConfig,
    /// 检索设置。
    pub retrieval: RetrievalConfig,
    /// 巩固/维护设置。
    pub consolidation: ConsolidationConfig,
    /// 进程级工程设置。
    pub service: ServiceConfig,
    /// LLM 设置。
    pub llm: LlmConfig,
}

impl Config {
    /// 从 `path` 指定的 TOML 文件加载配置；文件不存在时整份取默认值。
    ///
    /// 本函数**不读取任何环境变量**，也不长期保留中间态；文件路径由入口决定。
    pub fn load(path: &Path) -> ServiceResult<Self> {
        let file = match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|error| {
                ServiceError::Config(format!("解析配置文件 {} 失败: {error}", path.display()))
            })?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!(path = %path.display(), "配置文件不存在，使用默认值");
                FileConfig::default()
            }
            Err(error) => {
                return Err(ServiceError::Config(format!(
                    "读取配置文件 {} 失败: {error}",
                    path.display()
                )));
            }
        };

        if file.working_memory.window_capacity == 0 {
            return Err(ServiceError::Config("window_capacity 必须大于 0".into()));
        }
        if file.working_memory.max_cluster_nodes == 0 {
            return Err(ServiceError::Config("max_cluster_nodes 必须大于 0".into()));
        }
        if file.retrieval.top_k == 0 {
            return Err(ServiceError::Config("top_k 必须大于 0".into()));
        }
        if file.retrieval.candidate_k == 0 {
            return Err(ServiceError::Config("candidate_k 必须大于 0".into()));
        }
        if file.service.event_channel_capacity == 0 {
            return Err(ServiceError::Config(
                "event_channel_capacity 必须大于 0".into(),
            ));
        }
        if file.service.command_buffer == 0 {
            return Err(ServiceError::Config("command_buffer 必须大于 0".into()));
        }

        // 角色名会同时用于 SurrealDB 库名与 keyexpr 前缀，需保证是合法单段。
        let character = validate_character(&file.character)?;
        let key_prefix = match file.zenoh.key_prefix {
            Some(raw) => validate_keyexpr(&raw)?,
            None => format!("{DEFAULT_KEY_PREFIX}/{character}"),
        };

        Ok(Self {
            character,
            db_path: file.db_path,
            zenoh: ZenohConfig {
                enable: file.zenoh.enable,
                key_prefix: Some(key_prefix),
            },
            grpc: file.grpc,
            working_memory: file.working_memory,
            retrieval: file.retrieval,
            consolidation: file.consolidation,
            service: file.service,
            llm: file.llm,
        })
    }

    /// 已解析的 zenoh keyexpr 前缀。
    ///
    /// `load` 总会把它填成 `Some`；保留 `Option` 只是为了让 `ZenohConfig` 可直接反序列化。
    pub fn key_prefix(&self) -> &str {
        self.zenoh
            .key_prefix
            .as_deref()
            .unwrap_or(DEFAULT_KEY_PREFIX)
    }
}

/// 校验 keyexpr 前缀：非空、无首尾 `/`、无连续 `//`、无空白、不含 zenoh 通配符。
fn validate_keyexpr(raw: &str) -> ServiceResult<String> {
    if raw.is_empty() {
        return Err(ServiceError::Config("keyexpr 前缀不能为空".into()));
    }
    if raw.starts_with('/') || raw.ends_with('/') || raw.contains("//") {
        return Err(ServiceError::Config(format!(
            "keyexpr 前缀含非法斜杠: {raw}"
        )));
    }
    if raw.chars().any(char::is_whitespace) {
        return Err(ServiceError::Config(format!(
            "keyexpr 前缀不能含空白: {raw}"
        )));
    }
    if raw.chars().any(|c| matches!(c, '*' | '?' | '$' | '#')) {
        return Err(ServiceError::Config(format!(
            "keyexpr 前缀不能含通配符: {raw}"
        )));
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

    /// 写一份临时 TOML 再走 `Config::load`，避免为测试单独保留组装入口。
    fn load_from_toml(text: &str) -> ServiceResult<Config> {
        let path =
            std::env::temp_dir().join(format!("soulmem-config-test-{}.toml", uuid::Uuid::new_v4()));
        std::fs::write(&path, text).expect("写入临时配置文件");
        let result = Config::load(&path);
        let _ = std::fs::remove_file(&path);
        result
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../soulmem.example.toml");
        let file: FileConfig = toml::from_str(text).expect("示例配置应能解析");
        assert_eq!(file.character, DEFAULT_CHARACTER);
        assert_eq!(file.retrieval.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn defaults_when_fields_omitted() {
        let config = load_from_toml("").expect("空配置应通过");
        assert_eq!(config.character, DEFAULT_CHARACTER);
        assert_eq!(config.db_path, PathBuf::from(DEFAULT_DB_PATH));
        assert!(config.zenoh.enable);
        assert_eq!(
            config.key_prefix(),
            format!("{DEFAULT_KEY_PREFIX}/{DEFAULT_CHARACTER}")
        );
        assert!(config.grpc.enable);
        assert_eq!(
            config.working_memory.window_capacity,
            DEFAULT_WINDOW_CAPACITY
        );
        assert_eq!(config.retrieval.top_k, DEFAULT_TOP_K);
        assert_eq!(
            config.retrieval.similarity_threshold,
            DEFAULT_SIMILARITY_THRESHOLD
        );
        assert_eq!(
            config.consolidation.link_verb,
            DEFAULT_CONSOLIDATION_LINK_VERB
        );
        assert_eq!(config.service.command_buffer, DEFAULT_COMMAND_BUFFER);
        assert!(config.llm.base_url.is_empty());
        assert!(config.llm.api_key.is_none());
    }

    #[test]
    fn grouped_overrides_are_applied() {
        let text = r#"
            character = "yuki"
            db_path = "custom.db"

            [zenoh]
            enable = false
            key_prefix = "custom/prefix"

            [grpc]
            enable = false
            addr = "127.0.0.1:60000"

            [working_memory]
            window_capacity = 7
            max_cluster_nodes = 99

            [retrieval]
            candidate_k = 5
            top_k = 3
            similarity_threshold = 0.5
            abstract_source_priority = 1.5
            threads = 2
            include_short_summary = false
            short_clipping_length = 4

            [retrieval.association]
            damping_factor = 0.8
            top_k = 5

            [consolidation]
            interval_secs = 10
            idle_quiet_secs = 11
            link_top_k = 2
            link_verb = "linked"
            link_confidence = 0.5

            [service]
            heartbeat_interval_secs = 12
            event_channel_capacity = 13
            command_buffer = 14
            shutdown_grace_secs = 15

            [llm]
            base_url = "http://127.0.0.1:8080/v1"
            model = "m"
            api_key = "secret"
            probe_timeout_secs = 3
        "#;
        let config = load_from_toml(text).expect("应通过校验");
        assert_eq!(config.character, "yuki");
        assert_eq!(config.db_path, PathBuf::from("custom.db"));
        assert!(!config.zenoh.enable);
        assert_eq!(config.key_prefix(), "custom/prefix");
        assert!(!config.grpc.enable);
        assert_eq!(
            config.grpc.addr,
            "127.0.0.1:60000".parse::<SocketAddr>().unwrap()
        );
        assert_eq!(config.working_memory.window_capacity, 7);
        assert_eq!(config.working_memory.max_cluster_nodes, 99);
        assert_eq!(config.retrieval.candidate_k, 5);
        assert_eq!(config.retrieval.top_k, 3);
        assert_eq!(config.retrieval.similarity_threshold, 0.5);
        assert_eq!(config.retrieval.abstract_source_priority, 1.5);
        assert_eq!(config.retrieval.threads, 2);
        assert!(!config.retrieval.include_short_summary);
        assert_eq!(config.retrieval.short_clipping_length, Some(4));
        assert_eq!(config.retrieval.association.damping_factor, 0.8);
        assert_eq!(config.retrieval.association.top_k, 5);
        assert_eq!(config.consolidation.interval_secs, 10);
        assert_eq!(config.consolidation.idle_quiet_secs, 11);
        assert_eq!(config.consolidation.link_top_k, 2);
        assert_eq!(config.consolidation.link_verb, "linked");
        assert_eq!(config.consolidation.link_confidence, 0.5);
        assert_eq!(config.service.heartbeat_interval_secs, 12);
        assert_eq!(config.service.event_channel_capacity, 13);
        assert_eq!(config.service.command_buffer, 14);
        assert_eq!(config.service.shutdown_grace_secs, 15);
        assert_eq!(config.llm.base_url, "http://127.0.0.1:8080/v1");
        assert_eq!(config.llm.model, "m");
        assert_eq!(config.llm.probe_timeout_secs, 3);
    }

    #[test]
    fn missing_file_uses_defaults() {
        let path =
            std::env::temp_dir().join(format!("soulmem-missing-{}.toml", uuid::Uuid::new_v4()));
        let config = Config::load(&path).expect("缺失文件应取默认值");
        assert_eq!(config.character, DEFAULT_CHARACTER);
        assert_eq!(config.db_path, PathBuf::from(DEFAULT_DB_PATH));
        assert_eq!(
            config.key_prefix(),
            format!("{DEFAULT_KEY_PREFIX}/{DEFAULT_CHARACTER}")
        );
    }

    #[test]
    fn zero_window_capacity_is_rejected() {
        assert!(load_from_toml("[working_memory]\nwindow_capacity = 0").is_err());
    }

    #[test]
    fn zero_top_k_is_rejected() {
        assert!(load_from_toml("[retrieval]\ntop_k = 0").is_err());
    }

    #[test]
    fn zero_candidate_k_is_rejected() {
        assert!(load_from_toml("[retrieval]\ncandidate_k = 0").is_err());
    }

    #[test]
    fn zero_command_buffer_is_rejected() {
        assert!(load_from_toml("[service]\ncommand_buffer = 0").is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        assert!(load_from_toml("typo_field = 1").is_err());
        assert!(load_from_toml("[retrieval]\ntypo_field = 1").is_err());
    }

    #[test]
    fn default_key_prefix_includes_character() {
        let config = load_from_toml("character = \"yuki\"").expect("应通过校验");
        assert_eq!(config.key_prefix(), "soulmem/yuki");
    }

    #[test]
    fn explicit_key_prefix_overrides_character() {
        let config = load_from_toml(
            r#"
            character = "yuki"
            [zenoh]
            key_prefix = "custom"
            "#,
        )
        .expect("应通过校验");
        assert_eq!(config.key_prefix(), "custom");
    }

    #[test]
    fn invalid_key_prefix_is_rejected() {
        ["a/*", "/a", "a b"].into_iter().for_each(|bad| {
            let text = format!("[zenoh]\nkey_prefix = \"{bad}\"");
            assert!(load_from_toml(&text).is_err(), "应拒绝 {bad}");
        });
    }

    #[test]
    fn invalid_character_is_rejected() {
        assert!(load_from_toml("character = \"a/b\"").is_err());
    }

    #[test]
    fn llm_debug_redacts_api_key() {
        let config = load_from_toml("[llm]\napi_key = \"top-secret\"").expect("应通过校验");
        let debug = format!("{:?}", config.llm);
        assert!(!debug.contains("top-secret"));
        assert!(debug.contains("has_api_key: true"));
    }
}
