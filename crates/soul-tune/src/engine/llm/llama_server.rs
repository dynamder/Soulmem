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
        let prompt = format!(
            "{}\n\n用户说: \"{}\"\n\n作为角色，你需要从记忆中检索相关信息才能自然地回应这句话。\
             \n请输出一个 JSON 数组，格式严格如下，不要包含任何其他内容:\n\n\
             [\n  {{\n    \"tag\": [\"标签1\", \"标签2\"],\n    \"variant\": {{\n      \"Semantic\": [\n        {{\"concept_identifier\": \"具体概念关键词\"}}\n      ]\n    }},\n    \"priority\": 1\n  }}\n]\n\n\
             要求:\n\
             - 数组中的每项必须包含 \"tag\"、\"variant\"、\"priority\" 三个字段\n\
             - \"tag\" 是字符串数组，描述这个查询的类别\n\
             - \"variant\" 是对象，key 必须是 \"Semantic\"，value 是数组\n\
             - \"Semantic\" 数组中的每项必须包含 \"concept_identifier\" 字段\n\
             - \"priority\" 是 1~10 的整数\n\
             - 只输出 JSON 数组，不要输出其他任何文本、解释、或标记",
            system, user_message
        );
        self.raw_completion(&prompt)
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
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, user_message
        );
        self.raw_completion(&prompt)
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
                "4096",
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

    fn raw_completion(&self, prompt: &str) -> Result<String> {
        let body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": false
        });

        let url = format!("{}/v1/completions", self.api_url);
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
        let text = data["choices"][0]["text"]
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
