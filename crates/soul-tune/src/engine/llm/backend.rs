use anyhow::Result;

pub trait LlmBackend {
    fn generate_queries(&mut self, system: &str, user_message: &str) -> Result<String>;
    fn generate_response(
        &mut self,
        system: &str,
        context: &str,
        user_message: &str,
    ) -> Result<String>;
}
