use anyhow::Result;

/// soul-tune 各测试套件用的**同步**对话接口。
///
/// 它现在只是一层同步外壳，不是"另一套 LLM 实现"：
/// - 网络后端（`LlamaServer`）的请求拼装、响应解析、重试、超时、错误分类与 trace
///   全部由 `soul-mem-llm` 负责；
/// - 进程内后端（candle）保持自渲染提示词，它本来就不是 HTTP 调用。
///
/// 之所以保留同步签名：soul-tune 的测试套件与 FRB 入口都是同步的（这正是旧实现
/// 用 `reqwest::blocking` 的原因）。**提示词构建（查询生成、实体提取、回复生成）
/// 仍由调用方完成**，后端只负责把 system + user 两条消息变成 assistant 文本。
pub trait LlmBackend {
    fn chat(&mut self, system: &str, user_msg: &str, max_tokens: u32) -> Result<String>;
}
