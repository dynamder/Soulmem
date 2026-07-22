use crate::engine::llm::LlmBackend;

struct MockLlm {
    query_response: String,
    response_text: String,
}

impl LlmBackend for MockLlm {
    fn generate_queries(&mut self, _system: &str, _user_message: &str) -> anyhow::Result<String> {
        Ok(self.query_response.clone())
    }

    fn generate_response(
        &mut self,
        _system: &str,
        _context: &str,
        _user_message: &str,
    ) -> anyhow::Result<String> {
        Ok(self.response_text.clone())
    }
}

#[test]
fn test_mock_llm_generate_queries() {
    let json = r#"[
        {"tag": ["角色"], "variant": {"Semantic": [{"concept_identifier": "测试"}]}, "priority": 1}
    ]"#;
    let mut mock = MockLlm {
        query_response: json.into(),
        response_text: "mock response".into(),
    };

    let result = mock.generate_queries("system", "user message").unwrap();
    assert_eq!(result, json);
}

#[test]
fn test_mock_llm_generate_response() {
    let mut mock = MockLlm {
        query_response: String::new(),
        response_text: "你好，我是角色".into(),
    };

    let result = mock
        .generate_response("system prompt", "context", "hello")
        .unwrap();
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
