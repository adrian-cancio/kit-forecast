//! DiT block with per-role AdaLN modulation (port of `kit_v1/model/block.py`).

use candle_core::{D, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::attention::MultiheadAttention;
use super::norms::RmsNorm;
use super::rope::RopeTables;
use super::swiglu::SwiGlu;

/// One DiT block: attention + SwiGLU, modulated by the shared AdaLN output plus a
/// learned per-layer bias (`bias_i`).
#[derive(Debug, Clone)]
pub struct DiTBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: MultiheadAttention,
    mlp: SwiGlu,
    bias_i: Tensor,
}

impl DiTBlock {
    pub fn load(
        d: usize,
        n_heads: usize,
        ffn_hidden: usize,
        qk_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::plain(),
            norm2: RmsNorm::plain(),
            attn: MultiheadAttention::load(d, n_heads, qk_norm, vb.pp("attn"))?,
            mlp: SwiGlu::load(d, ffn_hidden, vb.pp("mlp"))?,
            bias_i: vb.get(6 * d, "bias_i")?,
        })
    }

    /// `x [B,S,d]`; `c_mod_ctx`/`c_mod_tgt` `[B,6d]`; `role [B,S]` index tensor
    /// (0 = ctx, 1 = tgt); `pad_mask [B,S]` (non-zero = valid) or `None`.
    pub fn forward(
        &self,
        x: &Tensor,
        c_mod_ctx: &Tensor,
        c_mod_tgt: &Tensor,
        role: &Tensor,
        pad_mask: Option<&Tensor>,
        rope: &RopeTables,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let six_d = c_mod_tgt.dim(D::Minus1)?;

        // Pick the modulation per token by role: [B,S,6d].
        let is_tgt = role
            .eq(1u32)?
            .unsqueeze(D::Minus1)?
            .broadcast_as((b, s, six_d))?;
        let tgt = c_mod_tgt.unsqueeze(1)?.broadcast_as((b, s, six_d))?;
        let ctx = c_mod_ctx.unsqueeze(1)?.broadcast_as((b, s, six_d))?;
        let modulation = is_tgt.where_cond(&tgt, &ctx)?.broadcast_add(&self.bias_i)?;

        let parts = modulation.chunk(6, D::Minus1)?;
        let [shift1, scale1, gate1, shift2, scale2, gate2] = parts.as_slice() else {
            candle_core::bail!("expected 6 modulation chunks, got {}", parts.len());
        };

        let h = self
            .norm1
            .forward(x)?
            .mul(&scale1.affine(1.0, 1.0)?)?
            .add(shift1)?;
        let attn_out = self.attn.forward(&h, pad_mask, rope)?;
        let x = (x + gate1.mul(&attn_out)?)?;

        let h = self
            .norm2
            .forward(&x)?
            .mul(&scale2.affine(1.0, 1.0)?)?
            .add(shift2)?;
        let mlp_out = self.mlp.forward(&h)?;
        x + gate2.mul(&mlp_out)?
    }
}
