//! 配置模型与加载。
//!
//! 集中、类型化的配置，避免魔法数散落；非法值在启动即失败而非运行中才暴露。
//! 读取顺序：默认值 ← 环境变量（`SOUL_MEM_*`）。

use crate::error::{Error, Result};
use std::path::PathBuf;

/// embedding 模型选择。Demo 默认使用确定性哈希模型，便于离线、可复现的单机验证。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingMode {
    /// 确定性哈希向量（离线可用，用于单机演示与测试）。
    Hash,
    /// 预留：真实 BGE/Qwen3 模型接入点（本期未实现）。
    Bge,
}

impl EmbeddingMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hash" => Ok(EmbeddingMode::Hash),
            "bge" => Ok(EmbeddingMode::Bge),
            other => Err(Error::InvalidArgument(format!(
                "unknown embedding mode: {other:?} (expected hash|bge)"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub device_id: String,
    pub zenoh_key_prefix: String,
    pub window_capacity: usize,
    pub similarity_threshold: f32,
    pub similarity_max_results: usize,
    pub llm_base_url: String,
    pub llm_api_key: String,
    pub llm_model: String,
    /// 各后台任务的触发间隔（秒）。0 表示停用该定时任务（仍可被控制信号强制触发）。
    pub persist_interval_secs: u64,
    pub consolidate_interval_secs: u64,
    pub forget_interval_secs: u64,
    pub embedding_mode: EmbeddingMode,
    /// 持久化快照路径；None 表示不做落盘（无持久化）。
    pub store_path: Option<PathBuf>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Config::from_env_with("")
    }

    /// 环境变量名统一为 `SOUL_MEM_<NAME>`。
    fn from_env_with(_prefix: &str) -> Result<Self> {
        let env = |name: &str| std::env::var(format!("SOUL_MEM_{name}")).ok();

        let device_id = env("DEVICE_ID").unwrap_or_else(default_device_id);
        let zenoh_key_prefix = env("ZENOH_KEY_PREFIX").unwrap_or_else(|| "soulmem".to_string());
        let window_capacity =
            parse_u64_env("WINDOW_CAPACITY", env("WINDOW_CAPACITY"), 20)? as usize;
        let similarity_threshold =
            parse_f32_env("SIMILARITY_THRESHOLD", env("SIMILARITY_THRESHOLD"), 0.05)?;
        let similarity_max_results =
            parse_u64_env("SIMILARITY_MAX_RESULTS", env("SIMILARITY_MAX_RESULTS"), 16)? as usize;
        let llm_base_url =
            env("LLM_BASE_URL").unwrap_or_else(|| "http://127.0.0.1:11434/v1".to_string());
        let llm_api_key = env("LLM_API_KEY").unwrap_or_else(|| "soulmem-demo".to_string());
        let llm_model = env("LLM_MODEL").unwrap_or_else(|| "demo-summarizer".to_string());
        let persist_interval_secs =
            parse_u64_env("PERSIST_INTERVAL_SECS", env("PERSIST_INTERVAL_SECS"), 60)?;
        let consolidate_interval_secs = parse_u64_env(
            "CONSOLIDATE_INTERVAL_SECS",
            env("CONSOLIDATE_INTERVAL_SECS"),
            0,
        )?;
        let forget_interval_secs =
            parse_u64_env("FORGET_INTERVAL_SECS", env("FORGET_INTERVAL_SECS"), 0)?;
        let embedding_mode =
            EmbeddingMode::parse(env("EMBEDDING_MODE").as_deref().unwrap_or("hash"))?;
        let store_path = env("STORE_PATH").map(PathBuf::from);

        Ok(Config {
            device_id,
            zenoh_key_prefix,
            window_capacity,
            similarity_threshold,
            similarity_max_results,
            llm_base_url,
            llm_api_key,
            llm_model,
            persist_interval_secs,
            consolidate_interval_secs,
            forget_interval_secs,
            embedding_mode,
            store_path,
        })
    }

    /// 校验并返回规范化的配置（demo 期主要做数值范围检查）。
    pub fn validate(&self) -> Result<()> {
        if self.window_capacity == 0 {
            return Err(Error::InvalidArgument("window_capacity must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(Error::InvalidArgument(
                "similarity_threshold must be within [0, 1]".into(),
            ));
        }
        if self.similarity_max_results == 0 {
            return Err(Error::InvalidArgument(
                "similarity_max_results must be > 0".into(),
            ));
        }
        if self.device_id.trim().is_empty() {
            return Err(Error::InvalidArgument("device_id must not be empty".into()));
        }
        Ok(())
    }
}

fn default_device_id() -> String {
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "soulmem".to_string());
    format!("{host}-{:08x}", uuid::Uuid::new_v4().as_u128() as u32)
}

/// 解析 `SOUL_MEM_<name>` 的整数值。
///
/// - 未设置（`None`）或空/空白字符串 → 默认值；
/// - 显式数值按原值返回（`0` 合法，用于停用定时任务）；
/// - 非法整数 → `InvalidArgument`。
fn parse_u64_env(name: &str, value: Option<String>, default: u64) -> Result<u64> {
    match value {
        None => Ok(default),
        Some(v) if v.trim().is_empty() => Ok(default),
        Some(v) => v
            .trim()
            .parse::<u64>()
            .map_err(|_| Error::InvalidArgument(format!("SOUL_MEM_{name} must be an integer"))),
    }
}

/// 解析 `SOUL_MEM_<name>` 的浮点值（未设置/空白 → 默认值）。
fn parse_f32_env(name: &str, value: Option<String>, default: f32) -> Result<f32> {
    match value {
        None => Ok(default),
        Some(v) if v.trim().is_empty() => Ok(default),
        Some(v) => v
            .trim()
            .parse::<f32>()
            .map_err(|_| Error::InvalidArgument(format!("SOUL_MEM_{name} must be a float"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_u64_env_unset_uses_default() {
        assert_eq!(parse_u64_env("WINDOW_CAPACITY", None, 20).unwrap(), 20);
        assert_eq!(
            parse_u64_env("CONSOLIDATE_INTERVAL_SECS", None, 0).unwrap(),
            0
        );
    }

    #[test]
    fn parse_u64_env_empty_uses_default() {
        assert_eq!(
            parse_u64_env("WINDOW_CAPACITY", Some(String::new()), 20).unwrap(),
            20
        );
        assert_eq!(
            parse_u64_env("WINDOW_CAPACITY", Some("   ".to_string()), 20).unwrap(),
            20
        );
    }

    #[test]
    fn parse_u64_env_explicit_values() {
        assert_eq!(
            parse_u64_env("WINDOW_CAPACITY", Some("7".to_string()), 20).unwrap(),
            7
        );
        // 显式 0 合法：表示停用该定时任务。
        assert_eq!(
            parse_u64_env("PERSIST_INTERVAL_SECS", Some("0".to_string()), 60).unwrap(),
            0
        );
    }

    #[test]
    fn parse_u64_env_invalid_is_error() {
        let err = parse_u64_env("WINDOW_CAPACITY", Some("abc".to_string()), 20).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn parse_f32_env_defaults_and_values() {
        assert_eq!(
            parse_f32_env("SIMILARITY_THRESHOLD", None, 0.05).unwrap(),
            0.05
        );
        assert_eq!(
            parse_f32_env("SIMILARITY_THRESHOLD", Some(String::new()), 0.05).unwrap(),
            0.05
        );
        assert_eq!(
            parse_f32_env("SIMILARITY_THRESHOLD", Some("0.3".to_string()), 0.05).unwrap(),
            0.3
        );
        assert!(matches!(
            parse_f32_env("SIMILARITY_THRESHOLD", Some("x".to_string()), 0.05),
            Err(Error::InvalidArgument(_))
        ));
    }
}
