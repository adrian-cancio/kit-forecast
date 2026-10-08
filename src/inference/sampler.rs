use candle_core::{DType, Device, Tensor};

use super::batch::{
    WindowBatch, concat, ensure_pad_mask, narrow, null_history, null_identity, repeat_interleave,
    repeat_interleave_tensor,
};
use super::options::{SampleOptions, Solver};
use super::rng::TorchRng;
use crate::model::{Condition, KlineDiT};
use crate::{Error, Result};

/// The `steps + 1` grid points of `torch.linspace(1.0, 0.0, steps + 1)` in `f32`.
pub(crate) fn time_grid(steps: usize) -> Vec<f32> {
    let points = steps + 1;
    let step = (0.0f32 - 1.0f32) / (points - 1) as f32;
    let half = points / 2;
    (0..points)
        .map(|i| {
            if i < half {
                1.0f32 + step * i as f32
            } else {
                0.0f32 - step * (points - i - 1) as f32
            }
        })
        .collect()
}

struct Branches {
    cond: Condition,
    n: usize,
    identity: bool,
    history: bool,
    x_ctx: Tensor,
    x_ctx_zero: Option<Tensor>,
}

fn forward_chunked(
    model: &KlineDiT,
    z: &Tensor,
    t: &Tensor,
    cond: &Condition,
    lh: usize,
    chunk: usize,
) -> Result<Tensor> {
    let n = z.dim(0)?;
    if n <= chunk {
        return model.forward(z, t, cond, lh);
    }
    let mut outs = Vec::with_capacity(n.div_ceil(chunk));
    let mut start = 0;
    while start < n {
        let len = chunk.min(n - start);
        let sub = narrow(cond, start, len)?;
        outs.push(model.forward(
            &z.narrow(0, start, len)?,
            &t.narrow(0, start, len)?,
            &sub,
            lh,
        )?);
        start += len;
    }
    Ok(Tensor::cat(&outs, 0)?)
}

fn velocity(
    model: &KlineDiT,
    br: &Branches,
    z_tgt: &Tensor,
    t_val: f32,
    opts: &SampleOptions,
    lh: usize,
) -> Result<Tensor> {
    let bk = z_tgt.dim(0)?;
    let device = z_tgt.device();
    let z_full = Tensor::cat(&[&br.x_ctx, z_tgt], 1)?;
    let t_vec = Tensor::full(t_val, bk, device)?;
    if br.n == 1 {
        return forward_chunked(model, &z_full, &t_vec, &br.cond, lh, opts.chunk_size);
    }
    let mut zs: Vec<Tensor> = vec![z_full.clone()];
    if br.identity {
        zs.push(z_full);
    }
    if br.history {
        let zero = br
            .x_ctx_zero
            .as_ref()
            .ok_or_else(|| Error::InvalidConfig("missing zeroed context".to_string()))?;
        zs.push(Tensor::cat(&[zero, z_tgt], 1)?);
    }
    let z_all = Tensor::cat(&zs.iter().collect::<Vec<_>>(), 0)?;
    let t_all = Tensor::cat(&vec![&t_vec; br.n], 0)?;
    let v_all = forward_chunked(model, &z_all, &t_all, &br.cond, lh, opts.chunk_size)?;
    let parts = v_all.chunk(br.n, 0)?;
    let v_full = &parts[0];
    let mut v = v_full.clone();
    let mut i = 1;
    if br.identity {
        v = (v + (v_full - &parts[i])?.affine(opts.cfg_w - 1.0, 0.0)?)?;
        i += 1;
    }
    if br.history {
        v = (v + (v_full - &parts[i])?.affine(opts.hist_cfg_w - 1.0, 0.0)?)?;
    }
    Ok(v)
}

fn prepare_branches(batch: &WindowBatch, opts: &SampleOptions) -> Result<(Branches, usize)> {
    let b = batch.len();
    let k = opts.n_paths;
    let bk = b * k;
    let lc = batch.context_len();
    let l = lc + batch.lh;
    let x_ctx = repeat_interleave_tensor(&batch.x_ctx, k)?;
    let mut cond_rep = repeat_interleave(&batch.cond, k)?;
    let identity = opts.uses_identity_guidance();
    let history = opts.uses_history_guidance();
    if history {
        cond_rep = ensure_pad_mask(&cond_rep, bk, l)?;
    }
    let mut branch_conds = vec![cond_rep.clone()];
    if identity {
        branch_conds.push(null_identity(&cond_rep)?);
    }
    let x_ctx_zero = if history {
        branch_conds.push(null_history(&cond_rep, lc)?);
        Some(x_ctx.zeros_like()?)
    } else {
        None
    };
    let n = branch_conds.len();
    let cond = if n == 1 {
        cond_rep
    } else {
        concat(&branch_conds)?
    };
    Ok((
        Branches {
            cond,
            n,
            identity,
            history,
            x_ctx,
            x_ctx_zero,
        },
        bk,
    ))
}

/// Sample `opts.n_paths` future paths per window with rectified-flow integration.
///
/// Convention (identical to upstream): `z_t = (1 - t) * x + t * eps`, so `t = 1` is pure noise
/// and the model predicts the velocity `dz/dt`. The state is integrated from `t = 1` to `t = 0`
/// on a uniform grid with `z <- z - dt * v`; only the horizon part of the sequence is updated,
/// the context stays clean. With guidance enabled the guided velocity is
/// `v_full + (cfg_w - 1) (v_full - v_noid) + (hist_cfg_w - 1) (v_full - v_nohist)`.
///
/// Returns the normalized paths as an `f32` tensor of shape `[B, K, Lh, F]` on the model
/// device. The initial noise is one `randn` draw of shape `[B * K, Lh, F]` from [`TorchRng`]
/// seeded with `opts.seed`.
pub fn flow_sample(model: &KlineDiT, batch: &WindowBatch, opts: &SampleOptions) -> Result<Tensor> {
    flow_sample_observed(model, batch, opts, |_, _| Ok(()))
}

/// [`flow_sample`] that reports the state after every integration step.
///
/// observer(i, z) is called after step i (0-based) with the horizon state z of shape
/// [B * K, Lh, F]. Returning an error aborts sampling. Use it for progress reporting, to
/// record trajectories, or to localise numerical differences against another implementation.
pub fn flow_sample_observed(
    model: &KlineDiT,
    batch: &WindowBatch,
    opts: &SampleOptions,
    mut observer: impl FnMut(usize, &Tensor) -> Result<()>,
) -> Result<Tensor> {
    opts.validate()?;
    if batch.is_empty() {
        return Err(Error::InvalidData(
            "cannot sample an empty batch".to_string(),
        ));
    }
    let lh = batch.lh;
    let features = batch.x_ctx.dim(2)?;
    let (branches, bk) = prepare_branches(batch, opts)?;
    let device: &Device = batch.device();

    let mut rng = TorchRng::new(opts.seed);
    let noise = rng.randn(bk * lh * features);
    let mut z = Tensor::from_vec(noise, (bk, lh, features), device)?.to_dtype(DType::F32)?;

    let grid = time_grid(opts.steps);
    for i in 0..opts.steps {
        let t_cur = grid[i];
        let t_next = grid[i + 1];
        let dt = f64::from(t_cur) - f64::from(t_next);
        let v1 = velocity(model, &branches, &z, t_cur, opts, lh)?;
        let last = i + 1 == opts.steps;
        z = match opts.solver {
            Solver::Euler => (&z - v1.affine(dt, 0.0)?)?,
            Solver::Heun if last => (&z - v1.affine(dt, 0.0)?)?,
            Solver::Heun => {
                let z_euler = (&z - v1.affine(dt, 0.0)?)?;
                let v2 = velocity(model, &branches, &z_euler, t_next, opts, lh)?;
                (&z - (&v1 + &v2)?.affine(dt * 0.5, 0.0)?)?
            }
        };
        observer(i, &z)?;
    }
    let b = batch.len();
    Ok(z.reshape((b, opts.n_paths, lh, features))?)
}

#[cfg(test)]
mod grid_tests {
    use super::time_grid;

    #[test]
    fn grid_runs_from_one_to_zero() {
        let g = time_grid(4);
        assert_eq!(g, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
        let g = time_grid(32);
        assert_eq!(g.len(), 33);
        assert_eq!(g[0], 1.0);
        assert_eq!(g[32], 0.0);
    }
}
