//! 本地 llama-server 的进程生命周期与调用入口。
//!
//! 分工：
//! - **进程管理留在本文件**：探活、拉起子进程、优雅关闭、启动超时与异常退出诊断。
//! - **HTTP 调用全部交给 `soul-mem-llm`**：请求体拼装、响应解析、`reasoning_content`
//!   兜底、错误分类、重试与超时、可选 trace，都由那一条栈统一负责。
//!   本文件不再自己拼 JSON、不再自己解释状态码——那正是"四套 LLM 实现"的来源之一。
//!
//! `LlmBackend` 现在是**同步外壳**：soul-tune 的测试套件（以及 FRB 入口）是同步的，
//! 因此用一个全局运行时把异步引擎桥接过来。调用方必须在**阻塞上下文**中调用 `chat()`——
//! 这一点未变（旧实现用的是 `reqwest::blocking`，同样不能在异步上下文里调用）。
//!
//! # 从中导航
//!
//! - 想改"请求长什么样"（字段名、扩展体）：`local_config` 下面的
//!   [`soul_mem_llm::OaiCompatConfig`]，wire 形状在 `soul-mem-llm/src/oai_comp/wire.rs`。
//! - 想改"怎么重试/超时"：`soul-mem-llm/src/oai_comp/transport.rs` 与 `src/engine.rs`。
//! - 想看调用 trace：设 `SOULMEM_LLM_TRACE=<path>`，事件产生在 `soul-mem-llm/src/observer.rs`。
//! - 完整链路：`docs/architecture/llm-layer.md`。

use std::process::{Child, Command};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;
use soul_mem_llm::{Hints, OaiCompatBackend, OaiCompatConfig, Sampling, Task};

use crate::engine::llm::backend::LlmBackend;

/// 整次调用的总时限。
///
/// 与旧实现（`reqwest` 120s）保持一致：本地推理若超过它，重发整次生成只会再等一遍，
/// 不会更快得到结果。
const TOTAL_TIMEOUT: Duration = Duration::from_secs(120);

/// 沿用旧实现的固定温度。
///
/// 刻意不"顺手改成更合理"的值：本次是重构，不是调参；改了会让既有的 playtest
/// 观测结果失去可比性。
const DEFAULT_TEMPERATURE: f32 = 0.7;

/// 同步桥接用的全局运行时。
///
/// 刻意做成全局且**永不 drop**：`tokio::runtime::Runtime` 在异步上下文里被 drop 会 panic，
/// 而 `LlamaServer` 可能在任何地方被释放。本 crate 的 PAW 桥接已是同样做法。
fn bridge_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("创建 LLM 同步桥接运行时失败"))
}

/// 后端身份标签：能从模型路径取到文件名就取，否则用通用名。
fn model_label(model_path: Option<&str>) -> String {
    model_path
        .and_then(|path| std::path::Path::new(path).file_stem())
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "local-llama-server".to_string())
}

/// `SOULMEM_LLM_TRACE=<path>` → 一行一条 JSON 的调用 trace。
fn engine_with_trace(cfg: OaiCompatConfig) -> Result<soul_mem_llm::LlmEngine> {
    let backend = OaiCompatBackend::new(cfg)?;
    // 本地单进程服务：连接类失败重试有意义（服务可能正在启动），
    // 但"整调用重发"对长生成不划算，关掉。
    let engine = soul_mem_llm::LlmEngine::new(Arc::new(backend)).with_whole_call_retries(0);

    match std::env::var("SOULMEM_LLM_TRACE") {
        Ok(path) if !path.trim().is_empty() => {
            let observer = soul_mem_llm::JsonlObserver::create(path.trim())
                .context("创建 LLM trace 文件失败")?;
            Ok(engine.with_observer(observer))
        }
        _ => Ok(engine),
    }
}

pub struct LlamaServer {
    process: Option<Child>,
    engine: soul_mem_llm::LlmEngine,
}

impl LlmBackend for LlamaServer {
    fn chat(&mut self, system: &str, user_msg: &str, max_tokens: u32) -> Result<String> {
        let task = Task::system_user(system, user_msg)
            .with_sampling(
                Sampling::default()
                    .with_temperature(DEFAULT_TEMPERATURE)
                    .with_max_output_tokens(max_tokens),
            )
            // 旧实现恒定发送 `chat_template_kwargs.enable_thinking=false`；
            // 现在由语义 hint + provider 配置表达，效果一致。
            .with_hints(Hints::default().with_suppress_reasoning(true));

        let completion = bridge_runtime()
            .block_on(self.engine.complete(task))
            .context("LLM 调用失败（请检查 llama-server 是否仍在运行）")?;
        Ok(completion.text)
    }
}

impl LlamaServer {
    /// 直连一个**已运行**的 llama-server（不探测、不拉起；调用方负责确认健康）。
    pub fn connect(url: &str) -> Result<Self> {
        let api_url = url.trim_end_matches('/').to_string();
        Ok(Self {
            process: None,
            engine: engine_with_trace(local_config(&api_url, &model_label(None)))?,
        })
    }

    /// 加载 LLM 后端。来源决策（显式 URL / 显式模型 / 探活复用 / 目录扫描）由
    /// [`super::resolver`] 统一处理；本函数只负责：显式 URL 直连（不可达则回退拉起）、
    /// 拉起指定 model_path 的子进程。**不**做默认端口探活复用——调用方已决定"要这个模型"，
    /// 避免复用到端口上其他模型导致行为漂移（如 CANDLE_MODEL_PATH 指定 Qwen3.5 却复用到别的服务）。
    pub fn load(model_path: &str) -> Result<Self> {
        let port = std::env::var("SOUL_TUNE_LLAMA_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(8081);

        let api_url = format!("http://127.0.0.1:{}", port);

        if let Ok(url) = std::env::var("SOUL_TUNE_LLAMA_URL")
            && super::resolver::probe_health(&url)
        {
            return Self::connect(&url);
        }
        // URL 已配置但不可达：回退到下方拉起本地模型

        let server_path = std::env::var("SOUL_TUNE_LLAMA_SERVER_PATH")
            .unwrap_or_else(|_| "llama-server".to_string());

        let mut process = Command::new(&server_path)
            .args([
                "-m",
                model_path,
                "--port",
                &port.to_string(),
                "-c",
                "32768",
                "--no-webui",
                "-ngl",
                "99",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "启动 llama-server 失败\n  路径: {}\n  模型: {}\n  请确保 llama-server 已安装并在 PATH 中（或设置 SOUL_TUNE_LLAMA_SERVER_PATH）",
                    server_path, model_path
                )
            })?;

        let health_client = reqwest::blocking::ClientBuilder::new()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .context("创建 HTTP client 失败")?;

        let health_url = format!("{}/health", api_url);
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(300);
        loop {
            match process.try_wait() {
                Ok(Some(status)) => {
                    anyhow::bail!(
                        "llama-server 启动后异常退出 (exit: {})\n  路径: {}\n  模型: {}\n  请手动运行检查错误信息",
                        status,
                        server_path,
                        model_path
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    anyhow::bail!("检查 llama-server 进程状态失败: {}", e);
                }
            }

            if start.elapsed() > timeout {
                Self::kill_process(&process);
                anyhow::bail!(
                    "llama-server 启动超时 ({}s)\n  路径: {}\n  模型: {}\n  请尝试增大超时时间或手动启动",
                    timeout.as_secs(),
                    server_path,
                    model_path
                );
            }

            match health_client.get(&health_url).send() {
                Ok(resp) if resp.status().is_success() => break,
                _ => {}
            }

            std::thread::sleep(Duration::from_millis(500));
        }

        Ok(Self {
            process: Some(process),
            engine: engine_with_trace(local_config(&api_url, &model_label(Some(model_path))))?,
        })
    }

    /// 供算法层直接使用的语义引擎。
    ///
    /// 拿到 `&LlmEngine` 后即可调用 `lazy_forget` 等算法入口——
    /// 不再需要"闭包工厂"把 transport 细节搬运过去。
    pub fn engine(&self) -> &soul_mem_llm::LlmEngine {
        &self.engine
    }

    #[cfg(windows)]
    fn kill_process(process: &Child) {
        let _ = Command::new("taskkill")
            .args(["/PID", &process.id().to_string(), "/F"])
            .output();
    }

    #[cfg(unix)]
    fn kill_process(process: &Child) {
        let _ = Command::new("kill")
            .args(["-9", &process.id().to_string()])
            .output();
    }
}

/// 本地服务的 provider 配置。
///
/// 这里是"本地 provider 与远程 provider 的差异"的唯一落点：
/// 无鉴权、环回地址绕过代理、把抑制 thinking 映射到 llama.cpp 的 `chat_template_kwargs`。
fn local_config(api_url: &str, model: &str) -> OaiCompatConfig {
    OaiCompatConfig::new("llama-server", format!("{api_url}/v1"), model)
        .with_auth_header(None, None)
        .with_reasoning_suppression("chat_template_kwargs", json!({"enable_thinking": false}))
        .with_total_timeout(Some(TOTAL_TIMEOUT))
        .with_first_byte_timeout(Some(TOTAL_TIMEOUT))
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Some(ref process) = self.process {
            Self::kill_process(process);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_label_prefers_file_stem() {
        assert_eq!(
            model_label(Some("D:/models/Qwen3-4B-Instruct-Q4_K_M.gguf")),
            "Qwen3-4B-Instruct-Q4_K_M"
        );
        assert_eq!(model_label(None), "local-llama-server");
    }

    #[test]
    fn local_config_matches_the_old_hand_rolled_request() {
        let cfg = local_config("http://127.0.0.1:8081", "m");
        assert_eq!(cfg.base_url, "http://127.0.0.1:8081/v1");
        assert!(
            cfg.resolved_no_proxy(),
            "本地服务必须绕过系统代理，否则会被 HTTP_PROXY 劫持"
        );
        assert!(cfg.auth_header.is_none(), "本地服务不应发送鉴权头");
        assert_eq!(
            cfg.reasoning_suppression_body["chat_template_kwargs"],
            json!({"enable_thinking": false}),
            "旧实现恒定发送该字段，重构后必须等价"
        );
        assert_eq!(cfg.total_timeout, Some(TOTAL_TIMEOUT));
    }

    #[test]
    fn engine_builds_without_a_running_server() {
        // 构造引擎不应触网（探活由 resolver 负责）
        let cfg = local_config("http://127.0.0.1:9", "m");
        assert!(engine_with_trace(cfg).is_ok());
    }
}
