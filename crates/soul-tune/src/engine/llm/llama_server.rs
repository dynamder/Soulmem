use std::process::{Child, Command};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::engine::llm::backend::LlmBackend;

pub struct LlamaServer {
    process: Option<Child>,
    client: reqwest::blocking::Client,
    api_url: String,
}

impl LlmBackend for LlamaServer {
    fn generate_queries(&mut self, system: &str, user_message: &str) -> Result<String> {
        let user_content = format!(
            "用户说: \"{}\"\n\n\
            从角色记忆中检索回应这句话所需的信息。输出一个 JSON 数组，格式如下:\n\
            [\n  {{\"tag\": [\"概念\", \"规则\"], \"variant\": {{\"Semantic\": [{{\"concept_identifier\": \"弹幕规则\"}}]}}, \"priority\": 7}},\n  {{\"tag\": [\"价值观\"], \"variant\": {{\"Semantic\": [{{\"concept_identifier\": \"欢愉至上\"}}]}}, \"priority\": 5}}\n]\n\n\
            concept_identifier 必须是具体的名词短语（如 \"弹幕规则\"、\"欢愉至上\"、\"Rust\"），\
            不能是模糊类别（如 \"爱好\"、\"技能\"）——这类词无法命中记忆中的具体概念。\
            \n\n只输出 JSON 数组，不要其他内容。",
            user_message
        );

        let messages = vec![
            serde_json::json!({"role": "system", "content": system}),
            serde_json::json!({"role": "user", "content": user_content}),
        ];

        self.chat_completion(&messages, 2048)
    }

    fn generate_response(
        &mut self,
        system: &str,
        context: &str,
        user_message: &str,
    ) -> Result<String> {
        let system_prompt = if context.is_empty() {
            system.to_string()
        } else {
            format!("{}\n\n相关记忆:\n{}", system, context)
        };

        let messages = vec![
            serde_json::json!({"role": "system", "content": system_prompt}),
            serde_json::json!({"role": "user", "content": user_message}),
        ];

        self.chat_completion(&messages, 512)
    }
}

impl LlamaServer {
    pub fn load(model_path: &str) -> Result<Self> {
        let port = std::env::var("SOUL_TUNE_LLAMA_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(8081);

        let api_url = format!("http://127.0.0.1:{}", port);

        if let Ok(url) = std::env::var("SOUL_TUNE_LLAMA_URL") {
            let client = reqwest::blocking::ClientBuilder::new()
                .no_proxy()
                .timeout(Duration::from_secs(120))
                .build()
                .context("创建 HTTP client 失败")?;
            return Ok(Self {
                process: None,
                client,
                api_url: url,
            });
        }

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

        let client = reqwest::blocking::ClientBuilder::new()
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
                        status, server_path, model_path
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

            match client.get(&health_url).send() {
                Ok(resp) if resp.status().is_success() => break,
                _ => {}
            }

            std::thread::sleep(Duration::from_millis(500));
        }

        Ok(Self {
            process: Some(process),
            client: reqwest::blocking::ClientBuilder::new()
                .no_proxy()
                .timeout(Duration::from_secs(120))
                .build()
                .context("创建 HTTP client 失败")?,
            api_url,
        })
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

    fn chat_completion(&self, messages: &[serde_json::Value], max_tokens: u32) -> Result<String> {
        let body = serde_json::json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": false,
            "chat_template_kwargs": {"enable_thinking": false}
        });

        let url = format!("{}/v1/chat/completions", self.api_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .context("LLM 请求失败（请检查 llama-server 是否仍在运行）")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("LLM API 返回 {}: {}", status, text);
        }

        let data: serde_json::Value = resp.json().context("解析 LLM 响应 JSON 失败")?;
        let text = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(text)
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Some(ref process) = self.process {
            Self::kill_process(process);
        }
    }
}
