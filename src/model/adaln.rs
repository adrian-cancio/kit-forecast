//! Shared AdaLN trunk (port of `kit_v1/model/adaln.py`).
//!
//! Called twice per forward pass (target with the real `t`, context with `t = 0`)
//! with shared weights.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

/// `h = silu(W1(t_emb + c_base))`; returns `c_mod = W2(h)` and `head_mod = W_head(h)`.
#[derive(Debug, Clone)]
pub struct SharedAdaLn {
    w1: Linear,
    w2: Linear,
    w_head: Linear,
}

impl SharedAdaLn {
    pub fn load(d: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear(d, d, vb.pp("w1"))?,
            // shift1, scale1, gate1, shift2, scale2, gate2
            w2: linear(d, 6 * d, vb.pp("w2"))?,
            // final shift, scale
            w_head: linear(d, 2 * d, vb.pp("w_head"))?,
        })
    }

    /// `t_emb_vec [B,d]`, `c_base [B,d]` -> `(c_mod [B,6d], head_mod [B,2d])`.
    pub fn forward(&self, t_emb_vec: &Tensor, c_base: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.w1.forward(&(t_emb_vec + c_base)?)?.silu()?;
        Ok((self.w2.forward(&h)?, self.w_head.forward(&h)?))
    }
}
