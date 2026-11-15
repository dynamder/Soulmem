use anyhow::Result;

/// soul-tune 测试套件用的同步对话接口。
///
/// 网络后端的请求/解析/重试/超时/错误分类由 `soul-mem-llm` 负责；进程内 candle 后端自渲染。
/// 同步是因为测试套件与 FRB 入口同步；提示词构建仍由调用方完成，后端只返回 assistant 文本。
pub trait LlmBackend {
    fn chat(&mut self, system: &str, user_msg: &str, max_tokens: u32) -> Result<String>;
}
