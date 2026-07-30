use std::fs::File;
use std::io::Seek;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama as llama;
use candle_transformers::models::quantized_qwen3 as qwen3;
use tokenizers::Tokenizer;

use crate::engine::llm::backend::LlmBackend;
use crate::engine::llm::qwen35;

pub struct CandleLlmConfig {
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for CandleLlmConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
        }
    }
}

enum QuantModel {
    Qwen3(qwen3::ModelWeights),
    Llama(llama::ModelWeights),
    Qwen35(qwen35::ModelWeights),
}

impl QuantModel {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(input, offset),
            Self::Llama(m) => m.forward(input, offset),
            Self::Qwen35(m) => m.forward(input, offset),
        }
    }
}

pub struct CandleLlm {
    model: QuantModel,
    tokenizer: Tokenizer,
    device: Device,
    config: CandleLlmConfig,
    eos_token_id: u32,
    im_end_token_id: u32,
}

impl LlmBackend for CandleLlm {
    fn generate_queries(&mut self, system: &str, user_message: &str) -> Result<String> {
        let prompt = format!("{}\n\n用户说: \"{}\"", system, user_message);
        self.generate(&prompt, 32768)
    }

    fn generate_response(
        &mut self,
        system: &str,
        context: &str,
        user_message: &str,
    ) -> Result<String> {
        let system_prompt = if context.is_empty() {
            system.to_string()
        } else {
            format!("{}\n\n相关记忆:\n{}", system, context)
        };
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, user_message
        );
        self.generate(&prompt, 1024)
    }
}

impl CandleLlm {
    pub fn load(config: CandleLlmConfig) -> Result<Self> {
        let model_path = Path::new(&config.model_path);
        let mut file = File::open(model_path)
            .with_context(|| format!("打开模型文件失败: {}", config.model_path))?;

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

        let probe = gguf_file::Content::read(&mut file).context("读取 GGUF 文件头(探测)失败")?;
        let gguf_dump = dump_gguf_header(&probe);
        let arch = probe
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_default();
        let has_qk = has_qk_norm(&probe);
        drop(probe);

        file.seek(std::io::SeekFrom::Start(0))
            .context("GGUF 寻址失败")?;
        let mut content = gguf_file::Content::read(&mut file).context("读取 GGUF 文件头失败")?;
        let model: QuantModel = match (|| -> anyhow::Result<QuantModel> {
            if arch == "qwen35" {
                file.seek(std::io::SeekFrom::Start(0))
                    .context("GGUF 寻址失败")?;
                Ok(QuantModel::Qwen35(qwen35::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )?))
            } else if has_qk {
                remap_metadata(&mut content, &["qwen3"]);
                file.seek(std::io::SeekFrom::Start(0))
                    .context("GGUF 寻址失败")?;
                Ok(QuantModel::Qwen3(qwen3::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )?))
            } else {
                remap_metadata(&mut content, &["llama"]);
                file.seek(std::io::SeekFrom::Start(0))
                    .context("GGUF 寻址失败")?;
                Ok(QuantModel::Llama(llama::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )?))
            }
        })() {
            Ok(m) => m,
            Err(e) => {
                let error_log = format!("{}\n\nERROR:\n{:?}", gguf_dump, e);
                let _ = std::fs::write("soul_tune_gguf.log", &error_log);
                return Err(e);
            }
        };

        let tokenizer_path = config.tokenizer_path.clone().unwrap_or_else(|| {
            model_path
                .parent()
                .unwrap_or(Path::new("."))
                .join("tokenizer.json")
                .to_string_lossy()
                .to_string()
        });
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("加载 tokenizer 失败 ({}): {}", tokenizer_path, e))?;

        let eos = tokenizer.token_to_id("<|im_end|>");
        let eos_token_id = eos.unwrap_or_else(|| tokenizer.token_to_id("</s>").unwrap_or(0));

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            eos_token_id,
            im_end_token_id: eos_token_id,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        let mut rng = rand::rng();

        let input = Tensor::new(&tokens[..], &self.device)
            .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?
            .unsqueeze(0)?;
        let mut logits = self
            .model
            .forward(&input, 0)
            .map_err(|e| anyhow::anyhow!("Forward error: {}", e))?;

        for _ in 0..max_tokens {
            let next_token = sample_logits(&logits, self.config.temperature, &mut rng)?;

            tokens.push(next_token);

            if next_token == self.eos_token_id || next_token == self.im_end_token_id {
                break;
            }

            let pos = tokens.len() - 1;
            let input = Tensor::new(&tokens[pos..=pos], &self.device)
                .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?
                .unsqueeze(0)?;
            logits = self
                .model
                .forward(&input, pos)
                .map_err(|e| anyhow::anyhow!("Forward error: {}", e))?;
        }

        let output = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

        Ok(strip_think_tags(&output))
    }
}

fn strip_think_tags(s: &str) -> String {
    let mut result = s.to_string();
    while let Some(start) = result.find("<｜end▁of▁thinking｜>") {
        let end = result[start..]
            .find("<｜end▁of▁thinking｜>")
            .map(|p| start + p + 7)
            .or_else(|| result[start..].find("<think/>").map(|p| start + p + 8))
            .unwrap_or(result.len());
        result.replace_range(start..end, "");
    }
    result.trim().to_string()
}

fn sample_logits(logits: &Tensor, temperature: f32, rng: &mut impl rand::RngCore) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    let logits = logits.to_vec1::<f32>()?;

    if temperature <= 0.0 {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return Ok(max_idx as u32);
    }

    let logits: Vec<f32> = logits.into_iter().map(|x| x / temperature).collect();
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

    let r = rng.next_u64() as f32 / u64::MAX as f32;
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return Ok(i as u32);
        }
    }

    Ok((probs.len() - 1) as u32)
}

fn dump_gguf_header(content: &gguf_file::Content) -> String {
    let mut lines = Vec::new();
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().map(|s| s.clone()))
        .unwrap_or_default();
    lines.push(format!("GGUF architecture: {}", arch));

    let file_type = content
        .metadata
        .get("general.file_type")
        .and_then(|v| v.to_u32().ok())
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    lines.push(format!("GGUF file_type: {}", file_type));

    let size_label = content
        .metadata
        .get("general.size_label")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "unknown".to_string());
    lines.push(format!("GGUF size_label: {}", size_label));

    lines.push(format!("GGUF metadata keys: {}", content.metadata.len()));
    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        lines.push(format!("  {}", k));
    }
    lines.push(format!("GGUF tensors: {}", content.tensor_infos.len()));
    let mut tkeys: Vec<&String> = content.tensor_infos.keys().collect();
    tkeys.sort();
    for t in &tkeys {
        if let Some(info) = content.tensor_infos.get(*t) {
            lines.push(format!("  {} ({:?}, {:?})", t, info.ggml_dtype, info.shape));
        } else {
            lines.push(format!("  {}", t));
        }
    }
    lines.join("\n")
}

fn has_qk_norm(content: &gguf_file::Content) -> bool {
    content
        .tensor_infos
        .contains_key("blk.0.attn_q_norm.weight")
}

fn remap_metadata(content: &mut gguf_file::Content, target_prefix: &[&str]) {
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_default();

    let keys = [
        "attention.head_count",
        "attention.head_count_kv",
        "block_count",
        "embedding_length",
        "context_length",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "attention.key_length",
        "rope.dimension_count",
        "expert_count",
        "expert_used_count",
    ];

    for suffix in &keys {
        for tp in target_prefix {
            let dst = format!("{}.{}", tp, suffix);
            if content.metadata.contains_key(&dst) {
                continue;
            }
            let candidates = [
                if arch.is_empty() { "" } else { &arch },
                "qwen3",
                "qwen2",
                "llama",
            ];
            for prefix in &candidates {
                if prefix.is_empty() {
                    continue;
                }
                let src = format!("{}.{}", prefix, suffix);
                if let Some(val) = content.metadata.get(&src) {
                    content.metadata.insert(dst.clone(), val.clone());
                    break;
                }
            }
        }
    }

    for tp in target_prefix {
        let dim_key = format!("{tp}.rope.dimension_count");
        if !content.metadata.contains_key(&dim_key) {
            compute_key_from_others(content, &dim_key, tp);
        }
        let kl_key = format!("{tp}.attention.key_length");
        if !content.metadata.contains_key(&kl_key) {
            compute_key_from_others(content, &kl_key, tp);
        }
        let exc_key = format!("{tp}.expert_count");
        if !content.metadata.contains_key(&exc_key) {
            content.metadata.insert(exc_key, gguf_file::Value::U32(0));
        }
        let excu_key = format!("{tp}.expert_used_count");
        if !content.metadata.contains_key(&excu_key) {
            content.metadata.insert(excu_key, gguf_file::Value::U32(0));
        }
    }
}

fn compute_key_from_others(content: &mut gguf_file::Content, dst: &str, prefix: &str) {
    if content.metadata.contains_key(dst) {
        return;
    }
    let hc_key = format!("{prefix}.attention.head_count");
    let hl_key = format!("{prefix}.embedding_length");
    if let (Some(n_heads), Some(hidden)) =
        (content.metadata.get(&hc_key), content.metadata.get(&hl_key))
    {
        if let (Ok(nh), Ok(hs)) = (n_heads.to_u32(), hidden.to_u32()) {
            if nh > 0 {
                content
                    .metadata
                    .insert(dst.to_string(), gguf_file::Value::U32(hs / nh));
            }
        }
    }
}
