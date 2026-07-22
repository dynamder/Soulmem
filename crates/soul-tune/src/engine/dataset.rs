#[derive(Debug, Clone)]
pub struct TestCaseConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub test_k_values: Vec<usize>,
}
