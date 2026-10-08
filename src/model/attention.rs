//! Bidirectional multi-head attention with QK-Norm and RoPE
//! (port of `kit_v1/model/attention.py`, plain softmax-attention path).

use candle_core::{D, DType, Result, Tensor, bail};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias, ops::softmax_last_dim};

use super::norms::RmsNorm;
use super::rope::RopeTables;

/// `pad_mask [B,S]` (non-zero = valid) -> boolean-like `u8` mask `[B,1,S,S]` (1 = attend).
///
/// Every query may attend to all valid keys and to itself: the diagonal is always
/// allowed so padded query rows never become all `-inf` (NaN).
fn dense_attn_mask(pad_mask: &Tensor, seq_len: usize) -> Result<Tensor> {
    let (b, s) = pad_mask.dims2()?;
    if s != seq_len {
        bail!("pad_mask has {s} positions but the sequence has {seq_len}");
    }
    let dev = pad_mask.device();
    let keys = pad_mask
        .to_dtype(DType::F32)?
        .reshape((b, 1, 1, s))?
        .broadcast_as((b, 1, s, s))?;
    let eye = Tensor::eye(s, DType::F32, dev)?.reshape((1, 1, s, s))?;
    keys.broadcast_add(&eye)?.gt(0f32)
}

#[derive(Debug, Clone)]
pub struct MultiheadAttention {
    n_heads: usize,
    head_dim: usize,
    qkv: Linear,
    out_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
}

impl MultiheadAttention {
    pub fn load(d: usize, n_heads: usize, qk_norm: bool, vb: VarBuilder) -> Result<Self> {
        if n_heads == 0 || !d.is_multiple_of(n_heads) {
            bail!("d={d} is not divisible by n_heads={n_heads}");
        }
        let head_dim = d / n_heads;
        // QK-Norm: RMSNorm over head_dim, weights shared across heads.
        let (q_norm, k_norm) = if qk_norm {
            (
                Some(RmsNorm::affine(head_dim, vb.pp("q_norm"))?),
                Some(RmsNorm::affine(head_dim, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            n_heads,
            head_dim,
            qkv: linear_no_bias(d, 3 * d, vb.pp("qkv"))?,
            out_proj: linear_no_bias(d, d, vb.pp("out_proj"))?,
            q_norm,
            k_norm,
        })
    }

    /// `x [B,S,d]`, `pad_mask [B,S]` (non-zero = valid, `None` = all valid),
    /// `rope` tables for the `S` assembled positions.
    pub fn forward(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        rope: &RopeTables,
    ) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, s, 3, self.n_heads, self.head_dim))?;
        let split = |i: usize| -> Result<Tensor> {
            qkv.narrow(2, i, 1)?
                .squeeze(2)?
                .transpose(1, 2)? // [B,H,S,Dh]
                .contiguous()
        };
        let (mut q, mut k, v) = (split(0)?, split(1)?, split(2)?);
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(&q)?;
            k = kn.forward(&k)?;
        }
        let (q, k) = rope.apply(&q, &k)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
        if let Some(pad) = pad_mask {
            let allowed = dense_attn_mask(pad, s)?.broadcast_as(scores.shape())?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?
                .to_dtype(scores.dtype())?;
            scores = allowed.where_cond(&scores, &neg_inf)?;
        }
        let attn = softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, d))?;
        self.out_proj.forward(&out)
    }
}
