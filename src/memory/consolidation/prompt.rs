use std::mem::take;

// ChatCompletionRequestMessage: 消息枚举，统一 system/user/assistant 等不同角色的消息类型，
// 使其可以存入同一个 Vec 中发送给 LLM API。
// ChatCompletionRequestSystemMessage: 系统角色消息，用于设定 LLM 的行为指令和角色。
// ChatCompletionRequestUserMessage: 用户角色消息，用于向 LLM 提供具体的输入数据。
use async_openai::types::chat::{
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
};

use crate::memory::working_memory::llm::prompt::PromptBuilder;

/// 记忆整合提示词构建器。
///
/// 将对话摘要（summary_text）和热点记忆（hot_memories）组装成发送给 LLM 的消息序列，
/// LLM 会按照指令将摘要拆解为结构化的记忆节点（nodes）和边（edges）。
pub struct ConsolidationPrompt {
    /// 对话摘要文本，即需要被整合的原始对话内容
    summary_text: String,
    /// 热点记忆列表，提供当前活跃的记忆作为上下文参考
    hot_memories: Vec<String>,
    /// 组装好的消息序列，包含 system 指令和 user 输入，发送给 LLM
    content: Vec<ChatCompletionRequestMessage>,
}

impl ConsolidationPrompt {
    /// 创建新的整合提示词。
    ///
    /// # 参数
    /// - `summary_text`: 对话摘要文本
    /// - `hot_memories`: 当前热点记忆列表，可为空
    pub fn new(summary_text: impl Into<String>, hot_memories: impl Into<Vec<String>>) -> Self {
        let mut prompt = Self {
            summary_text: summary_text.into(),
            hot_memories: hot_memories.into(),
            content: Vec::new(),
        };
        prompt.rebuild_messages();
        prompt
    }

    /// 重新构建消息序列。
    ///
    /// 生成两条消息：
    /// 1. **system 消息** — 告知 LLM 它是记忆整合引擎，定义输出格式和约束规则
    /// 2. **user 消息** — 提供对话摘要、热点记忆和 JSON Schema，作为 LLM 的处理输入
    fn rebuild_messages(&mut self) {
        self.content.clear();

        // system 消息：定义 LLM 的角色和行为约束
        let system_prompt = r#"You are a memory consolidation engine.
Task: split conversation summary into memory nodes and edges.
Output must be strict JSON only. Do not output markdown, explanation, or extra text.
All fields must conform to the provided JSON schema.
If information is uncertain, lower confidence instead of inventing details.
Every edge from/to must reference an existing node_id."#;

        // 将热点记忆格式化为 JSON 数组字符串，空列表时输出 "[]"
        let hot_memories_text = if self.hot_memories.is_empty() {
            "[]".to_string()
        } else {
            format!(
                "[\n{}\n]",
                self.hot_memories
                    .iter()
                    .map(|x| format!("  {:?}", x))
                    .collect::<Vec<_>>()
                    .join(",\n")
            )
        };

        // user 消息：拼接摘要文本、热点记忆和 JSON Schema，供 LLM 处理
        let user_prompt = format!(
            "summary_text:\n{}\n\nhot_memories:\n{}\n\njson_schema:\n{}",
            self.summary_text,
            hot_memories_text,
            include_str!("consolidation.schema.json")
        );

        // 将 system 和 user 消息通过 .into() 转为 ChatCompletionRequestMessage 枚举，
        // 存入 content 向量，最终作为 API 请求的消息列表
        self.content
            .push(ChatCompletionRequestSystemMessage::from(system_prompt).into());
        self.content
            .push(ChatCompletionRequestUserMessage::from(user_prompt).into());
    }
}

impl PromptBuilder for ConsolidationPrompt {
    /// 构建并返回消息序列，供 LLM 客户端发送。
    ///
    /// 如果消息已被消费（content 为空），会重新构建。
    /// 使用 `take` 取出内容后清空自身，避免重复发送。
    fn build_prompt(&mut self) -> Vec<ChatCompletionRequestMessage> {
        if self.content.is_empty() {
            self.rebuild_messages();
        }
        take(&mut self.content)
    }
}
