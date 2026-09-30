//! 服务配置（环境变量驱动）。
//!
//! 一个进程只服务一个角色/一个记忆库，因此角色名与库路径在启动时固定。
//! 所有变量以 `SOULMEM_` 为前缀，未设置时使用下方默认值。

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::error::{ServiceError, ServiceResult};

/// 默认 keyexpr 前缀。
const DEFAULT_KEY_PREFIX: &str = "soulmem";
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

/// 服务启动配置。
#[derive(Debug, Clone)]
pub struct Config {
    /// 角色名，同时作为 SurrealDB 的库名。
    pub character: String,
    /// SurrealDB 数据文件路径。
    pub db_path: PathBuf,
    /// zenoh keyexpr 前缀（`<prefix>/req` 等）。
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
    /// 从环境变量读取配置，未设置的项取默认值。
    pub fn from_env() -> ServiceResult<Self> {
        let character = opt("SOULMEM_CHARACTER").unwrap_or_else(|| "soulmem".to_string());

        let db_path = opt("SOULMEM_DB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("soulmem.db"));

        let key_prefix =
            opt("SOULMEM_KEY_PREFIX").unwrap_or_else(|| DEFAULT_KEY_PREFIX.to_string());

        let enable_zenoh = parse_bool("SOULMEM_ZENOH", true)?;
        let enable_grpc = parse_bool("SOULMEM_GRPC", true)?;

        let grpc_addr = match opt("SOULMEM_GRPC_ADDR") {
            Some(raw) => raw
                .parse::<SocketAddr>()
                .map_err(|e| ServiceError::Config(format!("SOULMEM_GRPC_ADDR 非法: {e}")))?,
            None => DEFAULT_GRPC_ADDR
                .parse::<SocketAddr>()
                .map_err(|e| ServiceError::Config(format!("默认 gRPC 地址非法: {e}")))?,
        };

        let window_capacity = parse_usize("SOULMEM_WINDOW_CAPACITY", DEFAULT_WINDOW_CAPACITY)?;
        if window_capacity == 0 {
            return Err(ServiceError::Config(
                "SOULMEM_WINDOW_CAPACITY 必须大于 0".into(),
            ));
        }
        let top_k = parse_usize("SOULMEM_TOP_K", DEFAULT_TOP_K)?;
        if top_k == 0 {
            return Err(ServiceError::Config("SOULMEM_TOP_K 必须大于 0".into()));
        }

        Ok(Self {
            character,
            db_path,
            key_prefix,
            enable_zenoh,
            enable_grpc,
            grpc_addr,
            window_capacity,
            consolidate_interval_secs: parse_u64(
                "SOULMEM_CONSOLIDATE_SECS",
                DEFAULT_CONSOLIDATE_SECS,
            )?,
            idle_quiet_secs: parse_u64("SOULMEM_IDLE_QUIET_SECS", DEFAULT_IDLE_QUIET_SECS)?,
            heartbeat_interval_secs: parse_u64("SOULMEM_HEARTBEAT_SECS", DEFAULT_HEARTBEAT_SECS)?,
            candidate_k: parse_usize("SOULMEM_CANDIDATE_K", DEFAULT_CANDIDATE_K)?,
            top_k,
            max_cluster_nodes: parse_usize("SOULMEM_MAX_CLUSTER_NODES", DEFAULT_MAX_CLUSTER_NODES)?,
        })
    }
}

/// 读取可选环境变量；空字符串视作未设置。
fn opt(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// 解析布尔环境变量（`1/true/yes/on` 为真，`0/false/no/off` 为假）。
fn parse_bool(key: &str, default: bool) -> ServiceResult<bool> {
    match opt(key) {
        None => Ok(default),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(ServiceError::Config(format!("{key} 非法布尔值: {other}"))),
        },
    }
}

/// 解析无符号整数环境变量。
fn parse_u64(key: &str, default: u64) -> ServiceResult<u64> {
    match opt(key) {
        None => Ok(default),
        Some(raw) => raw
            .parse::<u64>()
            .map_err(|e| ServiceError::Config(format!("{key} 非法整数: {e}"))),
    }
}

/// 解析 usize 环境变量。
fn parse_usize(key: &str, default: usize) -> ServiceResult<usize> {
    match opt(key) {
        None => Ok(default),
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|e| ServiceError::Config(format!("{key} 非法 usize: {e}"))),
    }
}
