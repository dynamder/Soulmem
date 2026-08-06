use crate::engine::llm::LlmBackend;

struct MockLlm {
    query_response: String,
    response_text: String,
}

impl LlmBackend for MockLlm {
    fn chat(&mut self, _system: &str, user_msg: &str, _max_tokens: u32) -> anyhow::Result<String> {
        if user_msg.contains("JSON 数组") {
            Ok(self.query_response.clone())
        } else {
            Ok(self.response_text.clone())
        }
    }
}

#[test]
fn test_mock_llm_chat_returns_query_json() {
    let json = r#"[
        {"tag": ["角色"], "variant": {"Semantic": [{"concept_identifier": "测试"}]}, "priority": 1}
    ]"#;
    let mut mock = MockLlm {
        query_response: json.into(),
        response_text: "mock response".into(),
    };

    let result = mock.chat("system", "请输出一个 JSON 数组", 2048).unwrap();
    assert_eq!(result, json);
}

#[test]
fn test_mock_llm_chat_returns_response() {
    let mut mock = MockLlm {
        query_response: String::new(),
        response_text: "你好，我是角色".into(),
    };

    let result = mock.chat("system prompt", "hello", 512).unwrap();
    assert_eq!(result, "你好，我是角色");
}

#[test]
fn test_playtest_repair_extract_think() {
    use crate::engine::playtest::repair::extract_think_content;
    let input = "<think>reasoning</think>body";
    assert_eq!(extract_think_content(input), Some("reasoning".into()));
}

#[test]
fn test_playtest_repair_strip_think() {
    use crate::engine::playtest::repair::strip_think_block;
    let input = "a<think>remove</think>b";
    assert_eq!(strip_think_block(input), "ab");
}

#[test]
fn test_playtest_repair_json_extract() {
    use crate::engine::playtest::repair::extract_json_array;
    let input = r#"prefix [{"k":"v"}] suffix"#;
    assert_eq!(extract_json_array(input), Some(r#"[{"k":"v"}]"#));
}
