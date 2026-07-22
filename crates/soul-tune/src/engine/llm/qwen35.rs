use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f64,
    pub rope_freq_base: f64,
    pub rope_dim: usize,
    pub ssm_state_size: usize,
    pub ssm_inner_size: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_time_step_rank: usize,
    pub full_attention_interval: usize,
}

impl Qwen35Config {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let md = |k: &str| {
            ct.metadata
                .get(k)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing metadata: {k}")))
        };
        Ok(Self {
            n_heads: md("qwen35.attention.head_count")?.to_u32()? as usize,
            n_kv_heads: md("qwen35.attention.head_count_kv")?.to_u32()? as usize,
            head_dim: md("qwen35.attention.key_length")?.to_u32()? as usize,
            hidden_size: md("qwen35.embedding_length")?.to_u32()? as usize,
            n_layers: md("qwen35.block_count")?.to_u32()? as usize,
            max_seq_len: md("qwen35.context_length")?.to_u32()? as usize,
            rms_norm_eps: md("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            rope_freq_base: md("qwen35.rope.freq_base")?.to_f32()? as f64,
            rope_dim: md("qwen35.rope.dimension_count")?.to_u32()? as usize,
            ssm_state_size: md("qwen35.ssm.state_size")?.to_u32()? as usize,
            ssm_inner_size: md("qwen35.ssm.inner_size")?.to_u32()? as usize,
            ssm_conv_kernel: md("qwen35.ssm.conv_kernel")?.to_u32()? as usize,
            ssm_time_step_rank: md("qwen35.ssm.time_step_rank")?.to_u32()? as usize,
            full_attention_interval: md("qwen35.full_attention_interval")?.to_u32()? as usize,
        })
    }
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    half_d: usize,
}

impl RotaryEmbedding {
    fn new(dtype: DType, dim: usize, max_seq_len: usize, base: f64, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self {
            cos,
            sin,
            half_d: dim / 2,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_emb = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_emb = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_emb, k_emb))
    }
}

struct MlpWeights {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl MlpWeights {
    fn load<R: Read + Seek>(gg: &mut GgufHelper<R>, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
            up: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
            down: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.apply(&Activation::Silu)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
    }
}

struct FusedAttention {
    qkv: QMatMul,
    gate: QMatMul,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: RotaryEmbedding,
}

impl FusedAttention {
    fn load<R: Read + Seek>(
        gg: &mut GgufHelper<R>,
        prefix: &str,
        config: &Qwen35Config,
        dev: &Device,
    ) -> Result<Self> {
        let rope = RotaryEmbedding::new(
            DType::F32,
            config.rope_dim,
            config.max_seq_len,
            config.rope_freq_base,
            dev,
        )?;
        Ok(Self {
            qkv: gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?,
            gate: gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            hidden_size: config.hidden_size,
            rotary_emb: rope,
        })
    }

    fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let q = qkv.narrow(2, 0, q_dim)?;
        let k = qkv.narrow(2, q_dim, kv_dim)?;
        let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?;
        let q = q
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let n_kv_groups = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, n_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        let gate = self.gate.forward(x)?.apply(&Activation::Sigmoid)?;
        ctx * gate
    }
}

struct StandardAttention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    output: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: RotaryEmbedding,
}

impl StandardAttention {
    fn load<R: Read + Seek>(
        gg: &mut GgufHelper<R>,
        prefix: &str,
        config: &Qwen35Config,
        dev: &Device,
    ) -> Result<Self> {
        let rope = RotaryEmbedding::new(
            DType::F32,
            config.rope_dim,
            config.max_seq_len,
            config.rope_freq_base,
            dev,
        )?;
        let q_weight = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), config.rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), config.rms_norm_eps)?;
        Ok(Self {
            q: q_weight,
            k: gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
            v: gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
            output: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
            q_norm,
            k_norm,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            hidden_size: config.hidden_size,
            rotary_emb: rope,
        })
    }

    fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;
        let q = q
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let n_kv_groups = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, n_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        self.output.forward(&ctx)
    }
}

struct SsmWeights {
    a: Tensor,
    alpha: Tensor,
    beta: Tensor,
    conv: Tensor,
    dt_bias: Option<Tensor>,
    norm: RmsNorm,
    out: QMatMul,
    inner_size: usize,
    state_size: usize,
    conv_kernel: usize,
    device: Device,
}

impl SsmWeights {
    fn load<R: Read + Seek>(
        gg: &mut GgufHelper<R>,
        prefix: &str,
        config: &Qwen35Config,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            a: gg.tensor(&format!("{prefix}.ssm_a"))?.dequantize(device)?,
            alpha: gg
                .tensor(&format!("{prefix}.ssm_alpha.weight"))?
                .dequantize(device)?,
            beta: gg
                .tensor(&format!("{prefix}.ssm_beta.weight"))?
                .dequantize(device)?,
            conv: gg
                .tensor(&format!("{prefix}.ssm_conv1d.weight"))?
                .dequantize(device)?,
            dt_bias: gg
                .try_tensor(&format!("{prefix}.ssm_dt.bias"))?
                .map(|t| t.dequantize(device))
                .transpose()?,
            norm: gg.rms_norm(&format!("{prefix}.ssm_norm.weight"), config.rms_norm_eps)?,
            out: gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?,
            inner_size: config.ssm_inner_size,
            state_size: config.ssm_state_size,
            conv_kernel: config.ssm_conv_kernel,
            device: device.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, l, _h) = x.dims3()?;
        let state = self.state_size;
        let inner = self.inner_size;
        let conv_k = self.conv_kernel;
        let device = &self.device;

        let h = self.norm.forward(x)?;

        let conv_w = self.conv.unsqueeze(1)?.to_dtype(DType::F32)?;
        let x_f32 = h.to_dtype(DType::F32)?;
        let conv_out = conv1d_simple(&x_f32, &conv_w, conv_k - 1)?;
        let conv_out = conv_out.apply(&Activation::Silu)?;

        let b_proj = conv_out.narrow(2, 0, inner)?;
        let dt_bias = self
            .dt_bias
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(inner, DType::F32, device).unwrap());
        let dt = b_proj.broadcast_add(&dt_bias)?.apply(&softplus)?;
        let dt = dt.unsqueeze(2)?;

        let a = self.a.expand((1, inner, state))?;
        let a_bar = (dt * &a)?.exp()?;

        let c = conv_out.narrow(2, inner, inner)?.unsqueeze(2)?;
        let scan_out = selective_scan(&conv_out, &a_bar, &b_proj.unsqueeze(2)?, &c, state, device)?;

        self.out.forward(&scan_out)
    }
}

fn selective_scan(
    x: &Tensor,
    a_bar: &Tensor,
    b: &Tensor,
    c: &Tensor,
    state_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let (_b_sz, l, inner) = x.dims3()?;
    let mut h = Tensor::zeros((1, inner, state_size), DType::F32, device)?;
    let mut ys = Vec::with_capacity(l);
    for t in 0..l {
        let xt = x.narrow(1, t, 1)?;
        let bt = b.narrow(1, t, 1)?;
        let ct = c.narrow(1, t, 1)?;
        h = (&h * a_bar.narrow(1, t, 1)? + bt.matmul(&xt)?)?;
        let yt = ct.matmul(&h)?;
        ys.push(yt);
    }
    Tensor::cat(&ys, 1)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(x)?;
    x.neg()?.exp()?.add(&one)?.log()
}

fn conv1d_simple(x: &Tensor, w: &Tensor, padding: usize) -> Result<Tensor> {
    let (_b, l, _d) = x.dims3()?;
    let (_, kw, _) = w.dims3()?;
    let padded = x.pad_with_zeros(1, padding, kw - 1 - padding)?;
    let mut out = Vec::with_capacity(l);
    for t in 0..l {
        let slice = padded.narrow(1, t, kw)?;
        let product = (slice * w)?.sum(1)?;
        out.push(product);
    }
    Tensor::stack(&out, 1)
}

enum LayerKind {
    Hybrid(FusedAttention, SsmWeights),
    Pure(StandardAttention),
}

struct LayerWeights {
    attn_norm: RmsNorm,
    kind: LayerKind,
    ffn: MlpWeights,
}

impl LayerWeights {
    fn load<R: Read + Seek>(
        gg: &mut GgufHelper<R>,
        prefix: &str,
        config: &Qwen35Config,
        dev: &Device,
    ) -> Result<Self> {
        let attn_norm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), config.rms_norm_eps)?;
        let ffn = MlpWeights::load(gg, prefix)?;
        let has_fused = gg
            .ct
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_qkv.weight"));
        let kind = if has_fused {
            LayerKind::Hybrid(
                FusedAttention::load(gg, prefix, config, dev)?,
                SsmWeights::load(gg, prefix, config, dev)?,
            )
        } else {
            LayerKind::Pure(StandardAttention::load(gg, prefix, config, dev)?)
        };
        Ok(Self {
            attn_norm,
            kind,
            ffn,
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.attn_norm.forward(x)?;
        let attn_out = match &self.kind {
            LayerKind::Hybrid(attn, ssm) => {
                let gated = attn.forward(&h, offset)?;
                let attn_res = (residual + gated)?;
                let s = ssm.forward(&attn_res)?;
                (attn_res + s)?
            }
            LayerKind::Pure(attn) => {
                let out = attn.forward(&h, offset)?;
                (residual + out)?
            }
        };
        let h = attn_out;
        let residual = h.clone();
        let ffn_out = self.ffn.forward(&h)?;
        residual + ffn_out
    }
}

pub struct ModelWeights {
    embed: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = GgufHelper::new(ct, reader, device.clone());
        let config = Qwen35Config::from_gguf(&gg.ct)?;
        let embed_t = gg.tensor("token_embd.weight")?;
        let embed = Embedding::new(embed_t.dequantize(device)?, config.hidden_size);
        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps)?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("blk.{i}");
            layers.push(LayerWeights::load(&mut gg, &prefix, &config, device)?);
        }
        let lm_head_t = match gg.tensor("output.weight") {
            Ok(t) => t,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(Arc::new(lm_head_t))?;
        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed.forward(input)?;
        for layer in &mut self.layers {
            h = layer.forward(&h, offset)?;
        }
        let h = self.norm.forward(&h)?;
        let last = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last)?.squeeze(1)
    }
}

struct GgufHelper<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> GgufHelper<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.tensor(name)?;
        QMatMul::from_weights(Arc::new(ws))
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.tensor(name)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn try_tensor(&mut self, name: &str) -> Result<Option<QTensor>> {
        match self.tensor(name) {
            Ok(t) => Ok(Some(t)),
            Err(_) => Ok(None),
        }
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))
}
