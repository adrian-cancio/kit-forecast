use candle_core::{DType, Device, Tensor};

use crate::data::PreparedWindow;
use crate::market::{CLOCK_DIM, SESS_DIM, YDAY_DIM};
use crate::model::Condition;
use crate::{Error, Result};

const N_FEATURES: usize = crate::data::N_FEATURES;

/// Model inputs for `B` windows that share one context and horizon length.
///
/// Built with [`WindowBatch::from_windows`]; consumed by [`flow_sample`](super::flow_sample).
#[derive(Debug, Clone)]
pub struct WindowBatch {
    pub(crate) x_ctx: Tensor,
    pub(crate) cond: Condition,
    pub(crate) lh: usize,
}

impl WindowBatch {
    /// Stack prepared windows into batched tensors on `device`.
    ///
    /// All windows must have the same `Lc` and `Lh`.
    pub fn from_windows(windows: &[&PreparedWindow], device: &Device) -> Result<Self> {
        let first = windows
            .first()
            .ok_or_else(|| Error::InvalidData("cannot batch zero windows".to_string()))?;
        let (lc, lh) = (first.lc(), first.lh());
        if let Some(bad) = windows.iter().find(|w| w.lc() != lc || w.lh() != lh) {
            return Err(Error::InvalidData(format!(
                "all windows in a batch must share Lc/Lh: expected {lc}/{lh}, found {}/{}",
                bad.lc(),
                bad.lh()
            )));
        }
        let b = windows.len();
        let l = lc + lh;

        let x_ctx: Vec<f32> = windows
            .iter()
            .flat_map(|w| w.x_ctx.iter().flatten().copied())
            .collect();
        let clock: Vec<f32> = windows
            .iter()
            .flat_map(|w| w.cond.clock.iter().flatten().copied())
            .collect();
        let sess: Vec<f32> = windows
            .iter()
            .flat_map(|w| w.cond.sess.iter().flatten().copied())
            .collect();
        let yday: Vec<f32> = windows
            .iter()
            .flat_map(|w| w.cond.yday.iter().flatten().copied())
            .collect();
        let per_bar = |pick: fn(&PreparedWindow) -> &Vec<i64>| -> Result<Tensor> {
            let v = windows
                .iter()
                .flat_map(|w| pick(w).iter().map(|&x| index(x)))
                .collect::<Result<Vec<u32>>>()?;
            Ok(Tensor::from_vec(v, (b, l), device)?)
        };
        let per_window = |pick: fn(&PreparedWindow) -> i64| -> Result<Tensor> {
            let v = windows
                .iter()
                .map(|w| index(pick(w)))
                .collect::<Result<Vec<u32>>>()?;
            Ok(Tensor::from_vec(v, b, device)?)
        };
        let pad: Vec<u8> = windows
            .iter()
            .flat_map(|w| w.cond.pad_mask.iter().map(|&m| u8::from(m)))
            .collect();

        let cond = Condition {
            clock: Tensor::from_vec(clock, (b, l, CLOCK_DIM), device)?,
            sess: Tensor::from_vec(sess, (b, l, SESS_DIM), device)?,
            dow: per_bar(|w| &w.cond.dow)?,
            event: per_bar(|w| &w.cond.event)?,
            month: per_bar(|w| &w.cond.month)?,
            yday: Tensor::from_vec(yday, (b, l, YDAY_DIM), device)?,
            market: per_window(|w| w.cond.market)?,
            sector: per_window(|w| w.cond.sector)?,
            inst: per_window(|w| w.cond.inst)?,
            scale: per_window(|w| w.cond.scale)?,
            pad_mask: Some(Tensor::from_vec(pad, (b, l), device)?),
        };
        Ok(Self {
            x_ctx: Tensor::from_vec(x_ctx, (b, lc, N_FEATURES), device)?,
            cond,
            lh,
        })
    }

    /// Number of windows.
    pub fn len(&self) -> usize {
        self.x_ctx.dims().first().copied().unwrap_or(0)
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Context length `Lc`.
    pub fn context_len(&self) -> usize {
        self.x_ctx.dims().get(1).copied().unwrap_or(0)
    }

    /// Horizon length `Lh`.
    pub fn horizon_len(&self) -> usize {
        self.lh
    }

    /// Device holding the tensors.
    pub fn device(&self) -> &Device {
        self.x_ctx.device()
    }

    /// Normalized context features, [B, Lc, F].
    pub fn context(&self) -> &Tensor {
        &self.x_ctx
    }

    /// Condition tensors for the whole window, L = Lc + Lh.
    pub fn condition(&self) -> &Condition {
        &self.cond
    }
}

fn index(value: i64) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::InvalidData(format!("embedding index {value} is out of range")))
}

fn map_cond(
    cond: &Condition,
    mut f: impl FnMut(&Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Condition> {
    Ok(Condition {
        clock: f(&cond.clock)?,
        sess: f(&cond.sess)?,
        dow: f(&cond.dow)?,
        event: f(&cond.event)?,
        month: f(&cond.month)?,
        yday: f(&cond.yday)?,
        market: f(&cond.market)?,
        sector: f(&cond.sector)?,
        inst: f(&cond.inst)?,
        scale: f(&cond.scale)?,
        pad_mask: cond.pad_mask.as_ref().map(&mut f).transpose()?,
    })
}

pub(crate) fn repeat_interleave(cond: &Condition, k: usize) -> candle_core::Result<Condition> {
    map_cond(cond, |t| repeat_interleave_tensor(t, k))
}

pub(crate) fn repeat_interleave_tensor(t: &Tensor, k: usize) -> candle_core::Result<Tensor> {
    let dims = t.dims().to_vec();
    let b = dims[0];
    let mut expanded = vec![b, k];
    expanded.extend_from_slice(&dims[1..]);
    let mut unsqueezed = vec![b, 1];
    unsqueezed.extend_from_slice(&dims[1..]);
    let mut merged = vec![b * k];
    merged.extend_from_slice(&dims[1..]);
    t.reshape(unsqueezed)?
        .broadcast_as(expanded)?
        .contiguous()?
        .reshape(merged)
}

pub(crate) fn narrow(cond: &Condition, start: usize, len: usize) -> candle_core::Result<Condition> {
    map_cond(cond, |t| t.narrow(0, start, len))
}

pub(crate) fn concat(parts: &[Condition]) -> candle_core::Result<Condition> {
    let stack = |pick: fn(&Condition) -> &Tensor| -> candle_core::Result<Tensor> {
        let ts: Vec<&Tensor> = parts.iter().map(pick).collect();
        Tensor::cat(&ts, 0)
    };
    let pad_mask = if parts.iter().all(|p| p.pad_mask.is_some()) {
        let ts: Vec<&Tensor> = parts.iter().filter_map(|p| p.pad_mask.as_ref()).collect();
        Some(Tensor::cat(&ts, 0)?)
    } else {
        None
    };
    Ok(Condition {
        clock: stack(|c| &c.clock)?,
        sess: stack(|c| &c.sess)?,
        dow: stack(|c| &c.dow)?,
        event: stack(|c| &c.event)?,
        month: stack(|c| &c.month)?,
        yday: stack(|c| &c.yday)?,
        market: stack(|c| &c.market)?,
        sector: stack(|c| &c.sector)?,
        inst: stack(|c| &c.inst)?,
        scale: stack(|c| &c.scale)?,
        pad_mask,
    })
}

pub(crate) fn ensure_pad_mask(
    cond: &Condition,
    batch: usize,
    len: usize,
) -> candle_core::Result<Condition> {
    let mut out = cond.clone();
    if out.pad_mask.is_none() {
        out.pad_mask = Some(Tensor::ones((batch, len), DType::U8, cond.clock.device())?);
    }
    Ok(out)
}

pub(crate) fn null_identity(cond: &Condition) -> candle_core::Result<Condition> {
    let mut out = cond.clone();
    out.market = out.market.zeros_like()?;
    out.sector = out.sector.zeros_like()?;
    out.inst = out.inst.zeros_like()?;
    out.scale = out.scale.zeros_like()?;
    Ok(out)
}

pub(crate) fn null_history(cond: &Condition, lc: usize) -> candle_core::Result<Condition> {
    let zero_prefix = |t: &Tensor| -> candle_core::Result<Tensor> {
        let len = t.dim(1)?;
        let mut keep_dims = t.dims().to_vec();
        keep_dims[1] = lc;
        let zeros = Tensor::zeros(keep_dims, t.dtype(), t.device())?;
        Tensor::cat(&[&zeros, &t.narrow(1, lc, len - lc)?], 1)
    };
    let mut out = cond.clone();
    out.clock = zero_prefix(&cond.clock)?;
    out.sess = zero_prefix(&cond.sess)?;
    out.dow = zero_prefix(&cond.dow)?;
    out.event = zero_prefix(&cond.event)?;
    out.month = zero_prefix(&cond.month)?;
    out.yday = zero_prefix(&cond.yday)?;
    let mask = match &cond.pad_mask {
        Some(m) => m.clone(),
        None => Tensor::ones(cond.dow.dims(), DType::U8, cond.dow.device())?,
    };
    out.pad_mask = Some(zero_prefix(&mask)?);
    Ok(out)
}
