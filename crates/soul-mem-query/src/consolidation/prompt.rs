use std::mem::take;

// ChatCompletionRequestMessage: 消息枚举，统一 system/user/assistant 等不同角色的消息类型，
// 使其可以存入同一个 Vec 中发送给 LLM API。
// ChatCompletionRequestSystemMessage: 系统角色消息，用于设定 LLM 的行为指令和角色。
// ChatCompletionRequestUserMessage: 用户角色消息，用于向 LLM 提供具体的输入数据。
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
};

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
        let system_prompt = r#"You are SoulMem's "Memory Consolidation Decomposer."
Task: Split the given summary_text into memory-graph nodes and edges, and use hot_memories as context to build more reasonable relations.

Hard requirements:
1) Output exactly one JSON object. Do not output any explanation, Markdown, or comments.
2) The output must strictly conform to the `consolidation.schema.json` shown below. Treat this schema as the single source of truth.
3) The top level must contain only: nodes, edges.
4) node_id must use temporary IDs (n1, n2, n3, ...), and each must be unique.
5) edges.from / edges.to must reference existing node_id values in nodes.
6) memory_type must be one of: semantic | situation | procedure.
7) Be conservative when uncertain: do not invent facts; if a relation is uncertain, you may omit that edge.
8) intensity and confidence must be within [0, 1].
9) Avoid semantically duplicated nodes within this output (e.g., merge synonyms like "like" and "really like" when appropriate).
10) edges may be an empty array, but nodes must contain at least one item.
11) Every required string field must contain meaningful, non-whitespace content. Never use placeholders such as "unknown", "N/A", "null", or "不详".
12) Each payload must use the exact structure selected by memory_type. Semantic payloads require content, aliases, description, and concept_type. Procedure payloads require content and action_type. Situation payloads must use one of the kind values and structures defined by the schema.
13) If a required fact is not supported by the summary_text or hot_memories, omit that node instead of inventing a value or using a placeholder.
14) Write all human-readable text values in Simplified Chinese, including content, aliases, descriptions, names, roles, actions, and edge relations. Keep JSON field names, enum values, and situation payload kind values exactly as required by the schema.
15) aliases, participants, emotions, sensory_data, and event may be empty arrays when the source contains no corresponding facts. location may be null. Do not omit required fields and never use placeholder values.
16) Split meaningful atomic concepts into separate nodes when the text explicitly supports them. In particular, separate a named entity from its category, type, or important property, then connect them with an edge.
17) Do not create nodes for ordinary function words, generic grammar fragments, or concepts that add no reusable meaning. Semantic content must be a concise concept, while description must state the supported meaning or context and should not merely repeat content when more context is available.
18) Do not add fields that are not defined by `consolidation.schema.json`. Before returning, check the required fields, allowed enum values, payload shape, numeric ranges, node ID references, and that the JSON can be parsed as one object.
    19) Use specific_situation only for a concrete occurrence whose time and required context are supported. time_span must be an RFC 3339 timestamp such as 2026-08-16T10:00:00Z. Use abstract_location, abstract_participant, abstract_environment, or abstract_event for reusable situation elements.
    20) An abstract-to-specific situation edge must point from an abstract situation node to a specific_situation node.
    21) For a named entity, use the most complete canonical name explicitly supported by summary_text or hot_memories as semantic payload.content. Put supported short names, nicknames, and forms of address in aliases. If hot_memories already contains the same entity, reuse its exact content spelling instead of creating a shortened variant.

    Again: return only the JSON object itself."#;

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
            "Input:\nsummary_text:\n{}\n\nhot_memories:\n{}\n\njson_schema:\n{}",
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

    /// 构建并返回消息序列，供 LLM 客户端发送。
    pub fn into_messages(mut self) -> Vec<ChatCompletionRequestMessage> {
        if self.content.is_empty() {
            self.rebuild_messages();
        }
        take(&mut self.content)
    }
}
