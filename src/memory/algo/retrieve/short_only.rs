use serde::Deserialize;

use crate::memory::{
    algo::retrieve::RetrRequest,
    memory_note::MemoryId,
    working_memory::{WorkingMemory, sliding_window::Information},
};
use std::sync::Arc;

use super::RetrStrategy;

#[derive(Debug, Clone, Deserialize)]
pub struct ShortOnlyConfig {
    #[serde(default)]
    pub clipping_length: Option<usize>,
    #[serde(default = "default_include_summary")]
    pub include_summary: bool,
}

fn default_include_summary() -> bool {
    false
}

impl ShortOnlyConfig {
    pub fn into_request(self, working_mem: Arc<WorkingMemory>) -> ShortOnlyRequest {
        ShortOnlyRequest {
            working_mem,
            clipping_length: self.clipping_length,
            include_summary: self.include_summary,
        }
    }
}

pub struct RetrShortOnly;

pub struct ShortOnlyRequest {
    working_mem: Arc<WorkingMemory>,
    //倒序计数
    pub clipping_length: Option<usize>,
    pub include_summary: bool,
}

impl RetrRequest for ShortOnlyRequest {}

impl RetrStrategy for RetrShortOnly {
    type Request = ShortOnlyRequest;
    type Return<'a>
        = (Arc<[Information]>, Arc<str>)
    where
        Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let window = {
            let mut window = request.working_mem.sliding_window().get_windows();
            if let Some(clipping_len) = request.clipping_length {
                window = window
                    .iter()
                    .rev()
                    .take(clipping_len)
                    .rev()
                    .cloned()
                    .collect::<Arc<_>>()
            }
            window
        };
        let summary = request.working_mem.sliding_window().get_summary();
        (window, summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::working_memory::sliding_window::{AssistantInformation, UserInformation};

    fn create_mock_working_memory_with_window() -> WorkingMemory {
        let mut wm = WorkingMemory::new(10);
        let sw = wm.sliding_window_mut();
        let window = sw.window();
        let mut guard = window.write();
        guard.push_back(Information::User(UserInformation::new("Hello")));
        guard.push_back(Information::Assistant(AssistantInformation::new(
            "Hi there!",
        )));
        guard.push_back(Information::User(UserInformation::new("How are you?")));
        drop(guard);
        let summary_lock = sw.summary();
        summary_lock.write().update("Previous summary");
        wm
    }

    #[test]
    fn test_retr_short_only_retrieve_all() {
        let wm = create_mock_working_memory_with_window();
        let config = ShortOnlyConfig {
            clipping_length: None,
            include_summary: true,
        };
        let request = config.into_request(Arc::new(wm));
        let result = RetrShortOnly {}.retrieve(request);

        assert_eq!(result.0.len(), 3);
        assert_eq!(result.0[0].get_str(), "Hello");
        assert_eq!(result.0[1].get_str(), "Hi there!");
        assert_eq!(result.0[2].get_str(), "How are you?");
        assert_eq!(result.1.as_ref(), "Previous summary");
    }

    #[test]
    fn test_retr_short_only_with_clipping() {
        let wm = create_mock_working_memory_with_window();
        let config = ShortOnlyConfig {
            clipping_length: Some(2),
            include_summary: false,
        };
        let request = config.into_request(Arc::new(wm));
        let result = RetrShortOnly {}.retrieve(request);

        assert_eq!(result.0.len(), 2);
        assert_eq!(result.0[0].get_str(), "Hi there!");
        assert_eq!(result.0[1].get_str(), "How are you?");
    }

    #[test]
    fn test_retr_short_only_empty_window() {
        let wm = WorkingMemory::new(10);
        let config = ShortOnlyConfig {
            clipping_length: None,
            include_summary: false,
        };
        let request = config.into_request(Arc::new(wm));
        let result = RetrShortOnly {}.retrieve(request);

        assert_eq!(result.0.len(), 0);
        assert_eq!(result.1.as_ref(), "");
    }
}
