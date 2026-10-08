//! Condition embeddings: timestep, global identity and per-bar calendar
//! (port of `kit_v1/model/embeddings.py`).
//!
//! Index tensors (`dow`, `event`, `month`, `market`, `sector`, `inst`, `scale`)
//! must use a dtype supported by `index_select` (`u32`, `i64` or `u8`).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear, linear_no_bias};

const MAX_PERIOD: f32 = 10000.0;

/// Embeds flow time `t` in `[0, 1]`: `sinusoidal(t * 1000) -> Linear -> SiLU -> Linear`.
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    freqs: Vec<f32>,
    fc1: Linear,
    fc2: Linear,
}

impl TimestepEmbedder {
    pub fn load(t_emb_dim: usize, d: usize, vb: VarBuilder) -> Result<Self> {
        let half = t_emb_dim / 2;
        // Same operation order as the reference: (-ln(max_period) * i) / half, then exp.
        let neg_log = -(MAX_PERIOD.ln());
        let freqs = (0..half)
            .map(|i| ((neg_log * i as f32) / half as f32).exp())
            .collect();
        Ok(Self {
            freqs,
            fc1: linear(t_emb_dim, d, vb.pp("fc1"))?,
            fc2: linear(d, d, vb.pp("fc2"))?,
        })
    }

    /// `t` `[B]` -> `[B, 2 * (t_emb_dim / 2)]`, laid out as `[cos | sin]`.
    fn sinusoidal(&self, t: &Tensor) -> Result<Tensor> {
        let dev: &Device = t.device();
        let b = t.dim(0)?;
        let half = self.freqs.len();
        let freqs = Tensor::from_slice(&self.freqs, (1, half), dev)?;
        let args = t
            .to_dtype(DType::F32)?
            .reshape((b, 1))?
            .broadcast_mul(&freqs)?;
        Tensor::cat(&[&args.cos()?, &args.sin()?], 1)
    }

    /// `t` `[B]` in `[0, 1]` -> `[B, d]`.
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let scaled = t.affine(1000.0, 0.0)?;
        let emb = self
            .sinusoidal(&scaled)?
            .to_dtype(self.fc1.weight().dtype())?;
        self.fc2.forward(&self.fc1.forward(&emb)?.silu()?)
    }
}

/// Global identity condition: `market + sector + proj(instrument) + scale` -> `[B, d]`.
///
/// Slot 0 of every table is a learnable unknown entry (no padding index).
#[derive(Debug, Clone)]
pub struct CondEmbeddings {
    market: Embedding,
    sector: Embedding,
    scale: Embedding,
    inst: Embedding,
    inst_proj: Linear,
}

impl CondEmbeddings {
    pub fn load(
        d: usize,
        market_slots: usize,
        sector_slots: usize,
        scale_slots: usize,
        inst_vocab: usize,
        inst_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            market: embedding(market_slots, d, vb.pp("market"))?,
            sector: embedding(sector_slots, d, vb.pp("sector"))?,
            scale: embedding(scale_slots, d, vb.pp("scale"))?,
            inst: embedding(inst_vocab, inst_dim, vb.pp("inst"))?,
            inst_proj: linear_no_bias(inst_dim, d, vb.pp("inst_proj"))?,
        })
    }

    /// Four `[B]` index tensors -> `c_base` `[B, d]`.
    pub fn forward(
        &self,
        market: &Tensor,
        sector: &Tensor,
        inst: &Tensor,
        scale: &Tensor,
    ) -> Result<Tensor> {
        let acc = (self.market.forward(market)? + self.sector.forward(sector)?)?;
        let acc = (acc + self.inst_proj.forward(&self.inst.forward(inst)?)?)?;
        acc + self.scale.forward(scale)?
    }
}

/// Per-bar calendar embedding `[B, L, d]`, added to token embeddings (never noised).
#[derive(Debug, Clone)]
pub struct CalendarEmbed {
    clock_proj: Linear,
    sess_fc1: Linear,
    sess_fc2: Linear,
    dow: Embedding,
    event: Embedding,
    month: Embedding,
    yday_proj: Linear,
}

impl CalendarEmbed {
    pub fn load(
        d: usize,
        clock_dim: usize,
        sess_hidden: usize,
        yday_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            clock_proj: linear(clock_dim, d, vb.pp("clock_proj"))?,
            sess_fc1: linear(2, sess_hidden, vb.pp("sess_fc1"))?,
            sess_fc2: linear(sess_hidden, d, vb.pp("sess_fc2"))?,
            dow: embedding(7, d, vb.pp("dow"))?,
            event: embedding(3, d, vb.pp("event"))?,
            month: embedding(12, d, vb.pp("month"))?,
            yday_proj: linear(yday_dim, d, vb.pp("yday_proj"))?,
        })
    }

    /// `clock [B,L,clock_dim]`, `sess [B,L,2]`, `yday [B,L,yday_dim]` float;
    /// `dow/event/month [B,L]` index tensors -> `[B,L,d]`.
    pub fn forward(
        &self,
        clock: &Tensor,
        sess: &Tensor,
        dow: &Tensor,
        event: &Tensor,
        month: &Tensor,
        yday: &Tensor,
    ) -> Result<Tensor> {
        let sess_emb = self
            .sess_fc2
            .forward(&self.sess_fc1.forward(sess)?.silu()?)?;
        let acc = (self.clock_proj.forward(clock)? + sess_emb)?;
        let acc = (acc + self.dow.forward(dow)?)?;
        let acc = (acc + self.event.forward(event)?)?;
        let acc = (acc + self.month.forward(month)?)?;
        acc + self.yday_proj.forward(yday)?
    }
}

/// Role embedding table: 0 = ctx, 1 = tgt.
pub fn load_role_embedding(d: usize, vb: VarBuilder) -> Result<Embedding> {
    embedding(2, d, vb)
}

/// Learnable register tokens `[n_registers, d]`, stored under the bare name `register`.
pub fn load_register_tokens(n_registers: usize, d: usize, vb: &VarBuilder) -> Result<Tensor> {
    vb.get((n_registers, d), "register")
}
