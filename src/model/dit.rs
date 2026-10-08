//! `KlineDiT` (port of `kit_v1/model/dit.py`).
//!
//! Global conditions modulate blocks through a shared AdaLN; per-bar calendar
//! features are added to the token embeddings. The model predicts the velocity
//! for the target segment only.

use candle_core::{D, DType, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, linear};

use super::adaln::SharedAdaLn;
use super::block::DiTBlock;
use super::condition::Condition;
use super::embeddings::{
    CalendarEmbed, CondEmbeddings, TimestepEmbedder, load_register_tokens, load_role_embedding,
};
use super::norms::RmsNorm;
use super::rope::RotaryEmbedding;
use crate::checkpoint::ModelConfig;

type BlockTap<'a> = dyn FnMut(usize, &Tensor) -> candle_core::Result<()> + 'a;

/// Conditioning vectors computed inside one forward pass (upstream `return_aux`).
#[derive(Debug, Clone)]
pub struct Modulations {
    /// Global identity embedding, `[B, d]`.
    pub c_base: Tensor,
    /// Block modulation for context tokens (`t = 0`), `[B, 6d]`.
    pub c_mod_ctx: Tensor,
    /// Block modulation for target tokens (real `t`), `[B, 6d]`.
    pub c_mod_tgt: Tensor,
    /// Final-layer shift and scale before `head_bias`, `[B, 2d]`.
    pub head_mod: Tensor,
}

#[derive(Debug, Clone)]
pub struct KlineDiT {
    in_proj: Linear,
    role_emb: Embedding,
    register: Tensor,
    calendar: CalendarEmbed,
    cond_emb: CondEmbeddings,
    t_emb: TimestepEmbedder,
    adaln: SharedAdaLn,
    blocks: Vec<DiTBlock>,
    final_norm: RmsNorm,
    head_bias: Tensor,
    out_proj: Linear,
    rope: RotaryEmbedding,
    n_registers: usize,
    in_features: usize,
}

impl KlineDiT {
    /// Build the model from checkpoint tensors exposed through `vb`
    /// (root-level names such as `in_proj.weight`, `blocks.0.attn.qkv.weight`).
    pub fn load(cfg: &ModelConfig, vb: VarBuilder) -> crate::Result<Self> {
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let in_features = cfg.in_features;
        let n_registers = cfg.n_registers;
        if n_heads == 0 || !d.is_multiple_of(n_heads) {
            return Err(candle_core::Error::Msg(format!(
                "d_model={d} is not divisible by n_heads={n_heads}"
            ))
            .into());
        }

        let blocks = (0..cfg.n_layers)
            .map(|i| {
                DiTBlock::load(
                    d,
                    n_heads,
                    cfg.ffn_hidden,
                    cfg.qk_norm,
                    vb.pp(format!("blocks.{i}")),
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self {
            in_proj: linear(in_features, d, vb.pp("in_proj"))?,
            role_emb: load_role_embedding(d, vb.pp("role_emb"))?,
            register: load_register_tokens(n_registers, d, &vb)?,
            calendar: CalendarEmbed::load(
                d,
                cfg.clock_dim,
                cfg.sess_hidden,
                cfg.yday_dim,
                vb.pp("calendar"),
            )?,
            cond_emb: CondEmbeddings::load(
                d,
                cfg.market_slots,
                cfg.sector_slots,
                cfg.scale_slots,
                cfg.inst_vocab,
                cfg.inst_dim,
                vb.pp("cond_emb"),
            )?,
            t_emb: TimestepEmbedder::load(cfg.t_emb_dim, d, vb.pp("t_emb"))?,
            adaln: SharedAdaLn::load(d, vb.pp("adaln"))?,
            blocks,
            final_norm: RmsNorm::plain(),
            head_bias: vb.get(2 * d, "head_bias")?,
            out_proj: linear(d, in_features, vb.pp("out_proj"))?,
            rope: RotaryEmbedding::new(d / n_heads, cfg.rope_theta as f32)?,
            n_registers,
            in_features,
        })
    }

    /// `v_pred = model(z, t, cond, lh)`.
    ///
    /// * `z` `[B, L, F]`: clean context followed by the noisy target (`lh` bars).
    /// * `t` `[B]`: flow time in `[0, 1]`.
    /// * `lh`: target length (`0 < lh <= L`).
    ///
    /// Returns the velocity for the target segment only, `[B, lh, F]`.
    pub fn forward(
        &self,
        z: &Tensor,
        t: &Tensor,
        cond: &Condition,
        lh: usize,
    ) -> crate::Result<Tensor> {
        Ok(self.forward_with_aux(z, t, cond, lh)?.0)
    }

    /// Like [`KlineDiT::forward`], additionally returning the intermediate conditioning
    /// vectors (upstream `return_aux=True`), useful to localise numerical differences.
    pub fn forward_with_aux(
        &self,
        z: &Tensor,
        t: &Tensor,
        cond: &Condition,
        lh: usize,
    ) -> crate::Result<(Tensor, Modulations)> {
        self.forward_inner(z, t, cond, lh, None)
    }

    /// Like [KlineDiT::forward], calling `tap(i, x)` with the token stream
    /// [B, registers + L, d] before the first block (i = 0) and after every block i.
    ///
    /// Intended for localising numerical differences against another implementation.
    pub fn forward_with_taps(
        &self,
        z: &Tensor,
        t: &Tensor,
        cond: &Condition,
        lh: usize,
        tap: &mut BlockTap<'_>,
    ) -> crate::Result<Tensor> {
        Ok(self.forward_inner(z, t, cond, lh, Some(tap))?.0)
    }

    fn forward_inner(
        &self,
        z: &Tensor,
        t: &Tensor,
        cond: &Condition,
        lh: usize,
        mut tap: Option<&mut BlockTap<'_>>,
    ) -> crate::Result<(Tensor, Modulations)> {
        let (b, l, f) = z.dims3()?;
        if f != self.in_features {
            return Err(candle_core::Error::Msg(format!(
                "z has {f} features, expected {}",
                self.in_features
            ))
            .into());
        }
        if lh == 0 || lh > l {
            return Err(candle_core::Error::Msg(format!(
                "target length lh={lh} must be in 1..={l}"
            ))
            .into());
        }
        let lc = l - lh;
        let n_reg = self.n_registers;
        let dev = z.device();

        // Role per bar: 0 = context, 1 = target.
        let role_tok = Tensor::cat(
            &[
                Tensor::zeros(lc, DType::U32, dev)?,
                Tensor::ones(lh, DType::U32, dev)?,
            ],
            0,
        )?
        .unsqueeze(0)?
        .broadcast_as((b, l))?
        .contiguous()?;

        let tok = (self.in_proj.forward(z)? + self.role_emb.forward(&role_tok)?)?;
        let tok = (tok
            + self.calendar.forward(
                &cond.clock,
                &cond.sess,
                &cond.dow,
                &cond.event,
                &cond.month,
                &cond.yday,
            )?)?;

        let reg = self
            .register
            .unsqueeze(0)?
            .broadcast_as((b, n_reg, self.register.dim(1)?))?;
        let x = Tensor::cat(&[&reg.contiguous()?, &tok], 1)?;

        let full_role = Tensor::cat(
            &[&Tensor::zeros((b, n_reg), DType::U32, dev)?, &role_tok],
            1,
        )?;
        // A missing pad mask means every bar is valid; the reference then builds an
        // all-true mask, which is equivalent to applying no mask at all.
        let full_pad = match &cond.pad_mask {
            Some(pad) => {
                let pad = pad.to_dtype(DType::F32)?.gt(0f32)?;
                Some(Tensor::cat(
                    &[&Tensor::ones((b, n_reg), DType::U8, dev)?, &pad],
                    1,
                )?)
            }
            None => None,
        };
        // RoPE uses indices of the assembled [registers | context | target] sequence.
        let positions = Tensor::arange(0u32, (n_reg + l) as u32, dev)?;
        let rope = self.rope.tables(&positions)?;

        let c_base = self
            .cond_emb
            .forward(&cond.market, &cond.sector, &cond.inst, &cond.scale)?;
        let (c_mod_tgt, head_mod) = self.adaln.forward(&self.t_emb.forward(t)?, &c_base)?;
        let (c_mod_ctx, _) = self
            .adaln
            .forward(&self.t_emb.forward(&t.zeros_like()?)?, &c_base)?;

        let mut x = x;
        if let Some(tap) = tap.as_deref_mut() {
            tap(0, &x)?;
        }
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(
                &x,
                &c_mod_ctx,
                &c_mod_tgt,
                &full_role,
                full_pad.as_ref(),
                &rope,
            )?;
            if let Some(tap) = tap.as_deref_mut() {
                tap(i + 1, &x)?;
            }
        }

        let x_tgt = x.narrow(1, n_reg + lc, lh)?;
        let head = head_mod.broadcast_add(&self.head_bias)?;
        let halves = head.chunk(2, D::Minus1)?;
        let [shift_f, scale_f] = halves.as_slice() else {
            return Err(candle_core::Error::Msg(format!(
                "expected 2 head modulation chunks, got {}",
                halves.len()
            ))
            .into());
        };
        let h = self
            .final_norm
            .forward(&x_tgt)?
            .broadcast_mul(&scale_f.affine(1.0, 1.0)?.unsqueeze(1)?)?
            .broadcast_add(&shift_f.unsqueeze(1)?)?;
        let v_pred = self.out_proj.forward(&h)?;
        Ok((
            v_pred,
            Modulations {
                c_base,
                c_mod_ctx,
                c_mod_tgt,
                head_mod,
            },
        ))
    }
}
