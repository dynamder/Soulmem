//! LLM 统一来源解析：**所有需要模型的地方**共用同一套决策——
//!
//! 1. 探测**已在运行**的 llama-server（`SOUL_TUNE_LLAMA_URL` 或默认端口 `SOUL_TUNE_LLAMA_PORT`）
//!    → 直接复用，不重复拉起；
//! 2. 无运行服务 → 自动拉起一个 llama-server 并加载**本地缓存的模型**
//!    （`SOUL_TUNE_CANDLE_MODEL_PATH` 显式指定，或扫描 `SOUL_TUNE_MODEL_DIR` /
//!    `models/` / `~/.soul-tune/models` 下的 .gguf/.ggml/.bin）；
//! 3. 都没有 → 返回不可用原因，由调用方决定报错或降级（如遗忘测试降级为遮罩路径）。

use std::path::{Path, PathBuf};

use anyhow::Result;

use super::llama_server::LlamaServer;

/// 模型来源状态（供配置页状态横幅 / 运行前提示展示，只探测、不启动）。
#[derive(Clone, Debug)]
pub struct LlmStatus {
    pub available: bool,
    /// `running`（复用运行中的 llama-server）| `spawned`（将自动拉起本地模型）| `unavailable`
    pub source: String,
    pub url: Option<String>,
    pub model_path: Option<String>,
    pub reason: Option<String>,
}

/// 解析结果：来源说明 + 可用时的 LlamaServer 实例（已探活/已拉起）。
pub struct LlmResolution {
    pub status: LlmStatus,
    pub server: Option<LlamaServer>,
}

/// 探测一个 llama-server HTTP 端点是否健康（GET /health，2s 超时，无代理）。
pub fn probe_health(url: &str) -> bool {
    let client = match reqwest::blocking::ClientBuilder::new()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let health = format!("{}/health", url.trim_end_matches('/'));
    match client.get(&health).send() {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// 默认的 llama-server 地址：`SOUL_TUNE_LLAMA_URL` 优先，否则 `127.0.0.1:SOUL_TUNE_LLAMA_PORT`。
pub fn default_url() -> String {
    if let Ok(url) = std::env::var("SOUL_TUNE_LLAMA_URL") {
        let url = url.trim().to_string();
        if !url.is_empty() {
            return url;
        }
    }
    let port = std::env::var("SOUL_TUNE_LLAMA_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8081);
    format!("http://127.0.0.1:{port}")
}

/// 查找本地缓存的模型文件（.gguf/.ggml/.bin，取体积最大者）：
/// 1. `SOUL_TUNE_CANDLE_MODEL_PATH` 显式指定（存在即用）；
/// 2. `SOUL_TUNE_MODEL_DIR` 目录扫描；
/// 3. 默认目录：`models/`（当前工作目录）与 `~/.soul-tune/models`。
pub fn find_cached_model() -> Option<String> {
    if let Ok(p) = std::env::var("SOUL_TUNE_CANDLE_MODEL_PATH") {
        let p = p.trim().to_string();
        if !p.is_empty() && Path::new(&p).is_file() {
            return Some(p);
        }
    }

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("SOUL_TUNE_MODEL_DIR") {
        let d = d.trim().to_string();
        if !d.is_empty() {
            dirs.push(PathBuf::from(d));
        }
    }
    dirs.push(PathBuf::from("models"));
    if let Some(home) = home_dir() {
        dirs.push(home.join(".soul-tune").join("models"));
    }

    let mut best: Option<(u64, PathBuf)> = None;
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let is_model = p
                .extension()
                .map(|e| {
                    let e = e.to_string_lossy().to_lowercase();
                    e == "gguf" || e == "ggml" || e == "bin"
                })
                .unwrap_or(false);
            if !is_model {
                continue;
            }
            let size = p.metadata().map(|m| m.len()).unwrap_or(0);
            if best.as_ref().map(|(s, _)| size > *s).unwrap_or(true) {
                best = Some((size, p));
            }
        }
    }
    best.map(|(_, p)| p.to_string_lossy().to_string())
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// 只探测、不启动：返回"如果现在请求模型会发生什么"（配置页状态横幅用）。
/// 优先级与 [`resolve_llm`] 一致：显式 URL / 显式模型优先于探活复用。
pub fn probe_status() -> LlmStatus {
    // 显式 URL
    if let Ok(u) = std::env::var("SOUL_TUNE_LLAMA_URL") {
        let u = u.trim().to_string();
        if !u.is_empty() {
            let healthy = probe_health(&u);
            return LlmStatus {
                available: true,
                source: if healthy { "running" } else { "unavailable" }.into(),
                url: Some(u),
                model_path: None,
                reason: (!healthy).then(|| "SOUL_TUNE_LLAMA_URL 配置但探活失败".into()),
            };
        }
    }

    // 显式模型路径
    if let Ok(p) = std::env::var("SOUL_TUNE_CANDLE_MODEL_PATH") {
        let p = p.trim().to_string();
        if !p.is_empty() {
            let exists = Path::new(&p).is_file();
            return LlmStatus {
                available: exists,
                source: if exists { "spawned" } else { "unavailable" }.into(),
                url: None,
                model_path: Some(p),
                reason: (!exists)
                    .then(|| "SOUL_TUNE_CANDLE_MODEL_PATH 指向的文件不存在".into()),
            };
        }
    }

    let url = default_url();
    if probe_health(&url) {
        return LlmStatus {
            available: true,
            source: "running".into(),
            url: Some(url),
            model_path: None,
            reason: None,
        };
    }
    if let Some(model) = find_cached_model() {
        return LlmStatus {
            available: true,
            source: "spawned".into(),
            url: None,
            model_path: Some(model),
            reason: None,
        };
    }
    LlmStatus {
        available: false,
        source: "unavailable".into(),
        url: None,
        model_path: None,
        reason: Some(
            "未发现运行中的 llama-server，也未找到本地缓存模型\
             （可设置 SOUL_TUNE_CANDLE_MODEL_PATH 或 SOUL_TUNE_MODEL_DIR）"
                .into(),
        ),
    }
}

/// 解析 LLM。优先级（**显式配置优先**，避免模型漂移）：
/// 1. `SOUL_TUNE_LLAMA_URL` 显式指定 → 探活复用该服务；
/// 2. `SOUL_TUNE_CANDLE_MODEL_PATH` 显式指定 → **拉起它**（即使默认端口有残留服务也不复用，
///    保证与配置的模型一致——否则端口上其他模型会导致生成行为漂移）；
/// 3. 无显式配置 → 探活默认端口上运行中的 llama-server → 复用；
/// 4. 无 → 扫描本地模型目录 → 拉起；
/// 5. 都没有 → 降级。
pub fn resolve_llm() -> LlmResolution {
    let url = default_url();

    // 1. 显式 URL：用户明确指定连接谁
    if let Ok(u) = std::env::var("SOUL_TUNE_LLAMA_URL") {
        let u = u.trim().to_string();
        if !u.is_empty() && probe_health(&u) {
            let server = LlamaServer::connect(&u);
            return LlmResolution {
                status: LlmStatus {
                    available: true,
                    source: "running".into(),
                    url: Some(u),
                    model_path: None,
                    reason: None,
                },
                server: server.ok(),
            };
        }
    }

    // 2. 显式模型路径：优先拉起用户指定的模型（不探活复用端口残留服务）
    if let Ok(p) = std::env::var("SOUL_TUNE_CANDLE_MODEL_PATH") {
        let p = p.trim().to_string();
        if !p.is_empty() && Path::new(&p).is_file() {
            return match LlamaServer::load(&p) {
                Ok(server) => LlmResolution {
                    status: LlmStatus {
                        available: true,
                        source: "spawned".into(),
                        url: None,
                        model_path: Some(p),
                        reason: None,
                    },
                    server: Some(server),
                },
                Err(e) => LlmResolution {
                    status: LlmStatus {
                        available: false,
                        source: "unavailable".into(),
                        url: None,
                        model_path: Some(p),
                        reason: Some(format!("启动 llama-server 失败: {e:#}")),
                    },
                    server: None,
                },
            };
        }
    }

    // 3. 探活默认端口：无显式配置时复用运行中的服务
    if probe_health(&url) {
        let server = LlamaServer::connect(&url);
        return LlmResolution {
            status: LlmStatus {
                available: true,
                source: "running".into(),
                url: Some(url),
                model_path: None,
                reason: None,
            },
            server: server.ok(),
        };
    }

    // 4. 目录扫描本地缓存模型
    if let Some(model) = find_cached_model() {
        return match LlamaServer::load(&model) {
            Ok(server) => LlmResolution {
                status: LlmStatus {
                    available: true,
                    source: "spawned".into(),
                    url: None,
                    model_path: Some(model),
                    reason: None,
                },
                server: Some(server),
            },
            Err(e) => LlmResolution {
                status: LlmStatus {
                    available: false,
                    source: "unavailable".into(),
                    url: None,
                    model_path: Some(model),
                    reason: Some(format!("启动 llama-server 失败: {e:#}")),
                },
                server: None,
            },
        };
    }

    LlmResolution {
        status: LlmStatus {
            available: false,
            source: "unavailable".into(),
            url: None,
            model_path: None,
            reason: Some(
                "未发现运行中的 llama-server，也未找到本地缓存模型\
                 （可设置 SOUL_TUNE_CANDLE_MODEL_PATH 或 SOUL_TUNE_MODEL_DIR）"
                    .into(),
            ),
        },
        server: None,
    }
}

/// 便捷：拿一个已就绪的 LlamaServer（供闭包/套件初始化），不可用时返回 None。
pub fn try_resolve_server() -> Option<LlamaServer> {
    resolve_llm().server
}

/// 供 `LlamaServer` 内部使用：带 URL 直连（不探测、不拉起）。
pub fn connect_server(url: &str) -> Result<LlamaServer> {
    LlamaServer::connect(url)
}
