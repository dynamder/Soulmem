use anyhow::Result;

/// 统一的 LLM 对话接口。
/// 所有提示词构建（查询生成、实体提取、回复生成）由调用方完成，
/// 后端只负责把 system + user 两条消息发送/渲染并返回 assistant 文本。
pub trait LlmBackend {
    fn chat(&mut self, system: &str, user_msg: &str, max_tokens: u32) -> Result<String>;
}
