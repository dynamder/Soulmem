use anyhow::{Context, Result};
use dotenvy::{from_filename, var};
use soul_mem_query::storage::SurrealConnectionConfig;

use crate::working_memory::llm::config::LLMConfig;

pub struct AppSettings {
    pub llm: LLMConfig,
    pub database: SurrealConnectionConfig,
}

impl AppSettings {
    pub fn load() -> Result<Self> {
        from_filename("soulmem.env").ok();

        let llm = LLMConfig::new(
            &required("API_KEY")?,
            &required("API_BASE")?,
            &required("MODEL")?,
        );
        let database = SurrealConnectionConfig::new(
            required("SURREAL_ENDPOINT")?,
            required("SURREAL_NAMESPACE")?,
            required("SURREAL_DATABASE")?,
        )
        .with_auth(
            required("SURREAL_USERNAME")?,
            required("SURREAL_PASSWORD")?,
        );

        Ok(Self { llm, database })
    }
}

fn required(name: &str) -> Result<String> {
    let value = var(name)
        .with_context(|| format!("missing required setting `{name}` in soulmem.env"))?;
    if value.trim().is_empty() {
        anyhow::bail!("setting `{name}` in soulmem.env cannot be empty");
    }
    Ok(value)
}
