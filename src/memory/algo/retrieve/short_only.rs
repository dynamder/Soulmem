//仅提取短期记忆策略，即仅提取滑动窗口
use super::RetrStrategy;
pub struct RetrShortOnly {
    clipping_length: Option<usize>,
    include_summary: bool,
}
impl RetrStrategy for RetrShortOnly {
    type RetrRequest = ();
    fn retrieve(&self, _request: Self::RetrRequest) -> Vec<String> {
        todo!()
    }
}
