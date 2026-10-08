use serde::{Deserialize, Serialize};

use super::Result;
use super::pickle::Dict;
use crate::Error;

/// Name and shape of one expected model parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Parameter name as stored in the checkpoint.
    pub name: String,
    /// Expected dimensions.
    pub shape: Vec<usize>,
}

impl TensorSpec {
    fn new(name: impl Into<String>, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            shape: shape.to_vec(),
        }
    }

    /// Number of elements, or `None` on overflow.
    pub fn numel(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }
}

/// Model architecture settings, the `config.model` section of the official checkpoint.
///
/// Unknown keys are ignored when parsing so that newer checkpoints keep loading;
/// every field that determines the parameter layout is required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden width of the transformer.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Hidden width of the SwiGLU feed-forward layers.
    pub ffn_hidden: usize,
    /// Dimension of the diffusion state (candle features).
    pub in_features: usize,
    /// Rotary embedding base.
    pub rope_theta: f64,
    /// Number of learnable register tokens.
    pub n_registers: usize,
    /// Number of instrument embedding slots.
    pub inst_vocab: usize,
    /// Low-rank instrument embedding width.
    pub inst_dim: usize,
    /// Number of market embedding slots.
    pub market_slots: usize,
    /// Number of sector embedding slots.
    pub sector_slots: usize,
    /// Number of timescale embedding slots.
    pub scale_slots: usize,
    /// Sinusoidal timestep embedding width.
    pub t_emb_dim: usize,
    /// Clock-time Fourier feature width.
    pub clock_dim: usize,
    /// Hidden width of the in-session position MLP.
    pub sess_hidden: usize,
    /// Day-of-year Fourier feature width.
    pub yday_dim: usize,
    /// Whether attention applies RMS normalization to queries and keys.
    pub qk_norm: bool,
    /// Attention backend requested at training time; informational only.
    #[serde(default = "default_attn_backend")]
    pub attn_backend: String,
    /// Training-time activation checkpointing flag; informational only.
    #[serde(default)]
    pub use_activation_checkpointing: bool,
}

fn default_attn_backend() -> String {
    "auto".to_string()
}

impl ModelConfig {
    pub(crate) fn from_pickle(dict: &Dict<'_>) -> Result<Self> {
        if dict.get("yday_dim").is_none() {
            return Err(Error::Checkpoint(
                "checkpoint config.model has no `yday_dim`: it predates the month/yday calendar \
                 features and would run with untrained embeddings; use the matching old code \
                 or retrain"
                    .to_string(),
            ));
        }
        let config = Self {
            d_model: dict.usize("d_model")?,
            n_layers: dict.usize("n_layers")?,
            n_heads: dict.usize("n_heads")?,
            ffn_hidden: dict.usize("ffn_hidden")?,
            in_features: dict.usize("in_features")?,
            rope_theta: dict.f64("rope_theta")?,
            n_registers: dict.usize("n_registers")?,
            inst_vocab: dict.usize("inst_vocab")?,
            inst_dim: dict.usize("inst_dim")?,
            market_slots: dict.usize("market_slots")?,
            sector_slots: dict.usize("sector_slots")?,
            scale_slots: dict.usize("scale_slots")?,
            t_emb_dim: dict.usize("t_emb_dim")?,
            clock_dim: dict.usize("clock_dim")?,
            sess_hidden: dict.usize("sess_hidden")?,
            yday_dim: dict.usize("yday_dim")?,
            qk_norm: dict.bool("qk_norm")?,
            attn_backend: dict.string_or("attn_backend", "auto")?,
            use_activation_checkpointing: dict.bool_or("use_activation_checkpointing", false)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Per-head width, `d_model / n_heads`.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Checks structural constraints required to derive the parameter layout.
    pub fn validate(&self) -> Result<()> {
        let dims = [
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("ffn_hidden", self.ffn_hidden),
            ("in_features", self.in_features),
            ("n_registers", self.n_registers),
            ("inst_vocab", self.inst_vocab),
            ("inst_dim", self.inst_dim),
            ("market_slots", self.market_slots),
            ("sector_slots", self.sector_slots),
            ("scale_slots", self.scale_slots),
            ("t_emb_dim", self.t_emb_dim),
            ("clock_dim", self.clock_dim),
            ("sess_hidden", self.sess_hidden),
            ("yday_dim", self.yday_dim),
        ];
        for (name, value) in dims {
            if value == 0 {
                return Err(Error::InvalidConfig(format!(
                    "model.{name} must be positive"
                )));
            }
            if value > (1 << 28) {
                return Err(Error::InvalidConfig(format!(
                    "model.{name} = {value} is implausibly large"
                )));
            }
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "model.rope_theta must be finite and positive, got {}",
                self.rope_theta
            )));
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(Error::InvalidConfig(format!(
                "model.d_model {} is not divisible by model.n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(Error::InvalidConfig(format!(
                "head dimension {} must be even for rotary embeddings",
                self.head_dim()
            )));
        }
        if !self.t_emb_dim.is_multiple_of(2) {
            return Err(Error::InvalidConfig(format!(
                "model.t_emb_dim {} must be even",
                self.t_emb_dim
            )));
        }
        Ok(())
    }

    /// Full ordered list of expected parameters (names and shapes) of the EMA state dict.
    pub fn tensor_specs(&self) -> Result<Vec<TensorSpec>> {
        self.validate()?;
        let d = self.d_model;
        let hd = self.head_dim();
        let per_block = if self.qk_norm { 8 } else { 6 };
        let mut specs = Vec::with_capacity(33 + self.n_layers * per_block);
        specs.push(TensorSpec::new("register", &[self.n_registers, d]));
        specs.push(TensorSpec::new("head_bias", &[2 * d]));
        specs.push(TensorSpec::new("in_proj.weight", &[d, self.in_features]));
        specs.push(TensorSpec::new("in_proj.bias", &[d]));
        specs.push(TensorSpec::new("role_emb.weight", &[2, d]));
        specs.push(TensorSpec::new(
            "calendar.clock_proj.weight",
            &[d, self.clock_dim],
        ));
        specs.push(TensorSpec::new("calendar.clock_proj.bias", &[d]));
        specs.push(TensorSpec::new(
            "calendar.sess_fc1.weight",
            &[self.sess_hidden, 2],
        ));
        specs.push(TensorSpec::new(
            "calendar.sess_fc1.bias",
            &[self.sess_hidden],
        ));
        specs.push(TensorSpec::new(
            "calendar.sess_fc2.weight",
            &[d, self.sess_hidden],
        ));
        specs.push(TensorSpec::new("calendar.sess_fc2.bias", &[d]));
        specs.push(TensorSpec::new("calendar.dow.weight", &[7, d]));
        specs.push(TensorSpec::new("calendar.event.weight", &[3, d]));
        specs.push(TensorSpec::new("calendar.month.weight", &[12, d]));
        specs.push(TensorSpec::new(
            "calendar.yday_proj.weight",
            &[d, self.yday_dim],
        ));
        specs.push(TensorSpec::new("calendar.yday_proj.bias", &[d]));
        specs.push(TensorSpec::new(
            "cond_emb.market.weight",
            &[self.market_slots, d],
        ));
        specs.push(TensorSpec::new(
            "cond_emb.sector.weight",
            &[self.sector_slots, d],
        ));
        specs.push(TensorSpec::new(
            "cond_emb.scale.weight",
            &[self.scale_slots, d],
        ));
        specs.push(TensorSpec::new(
            "cond_emb.inst.weight",
            &[self.inst_vocab, self.inst_dim],
        ));
        specs.push(TensorSpec::new(
            "cond_emb.inst_proj.weight",
            &[d, self.inst_dim],
        ));
        specs.push(TensorSpec::new("t_emb.fc1.weight", &[d, self.t_emb_dim]));
        specs.push(TensorSpec::new("t_emb.fc1.bias", &[d]));
        specs.push(TensorSpec::new("t_emb.fc2.weight", &[d, d]));
        specs.push(TensorSpec::new("t_emb.fc2.bias", &[d]));
        specs.push(TensorSpec::new("adaln.w1.weight", &[d, d]));
        specs.push(TensorSpec::new("adaln.w1.bias", &[d]));
        specs.push(TensorSpec::new("adaln.w2.weight", &[6 * d, d]));
        specs.push(TensorSpec::new("adaln.w2.bias", &[6 * d]));
        specs.push(TensorSpec::new("adaln.w_head.weight", &[2 * d, d]));
        specs.push(TensorSpec::new("adaln.w_head.bias", &[2 * d]));
        for i in 0..self.n_layers {
            let p = format!("blocks.{i}");
            specs.push(TensorSpec::new(format!("{p}.bias_i"), &[6 * d]));
            specs.push(TensorSpec::new(format!("{p}.attn.qkv.weight"), &[3 * d, d]));
            specs.push(TensorSpec::new(
                format!("{p}.attn.out_proj.weight"),
                &[d, d],
            ));
            if self.qk_norm {
                specs.push(TensorSpec::new(format!("{p}.attn.q_norm.weight"), &[hd]));
                specs.push(TensorSpec::new(format!("{p}.attn.k_norm.weight"), &[hd]));
            }
            specs.push(TensorSpec::new(
                format!("{p}.mlp.w1.weight"),
                &[self.ffn_hidden, d],
            ));
            specs.push(TensorSpec::new(
                format!("{p}.mlp.w3.weight"),
                &[self.ffn_hidden, d],
            ));
            specs.push(TensorSpec::new(
                format!("{p}.mlp.w2.weight"),
                &[d, self.ffn_hidden],
            ));
        }
        specs.push(TensorSpec::new("out_proj.weight", &[self.in_features, d]));
        specs.push(TensorSpec::new("out_proj.bias", &[self.in_features]));
        Ok(specs)
    }

    /// Total number of scalar parameters implied by this configuration.
    pub fn parameter_count(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for spec in self.tensor_specs()? {
            let n = spec.numel().ok_or_else(|| {
                Error::InvalidConfig(format!("element count of {} overflows", spec.name))
            })?;
            total = total.checked_add(n as u64).ok_or_else(|| {
                Error::InvalidConfig("total parameter count overflows".to_string())
            })?;
        }
        Ok(total)
    }
}
