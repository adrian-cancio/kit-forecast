use crate::market::{Market, Timescale};
use crate::{Error, Result};

/// Default soft-clip constant `c` in `c * tanh(x / c)`.
pub const DEFAULT_SOFT_CLIP_C: f64 = 6.0;

const MAD_CONST: f64 = 1.4826;
const MIN_SCALE: f64 = 1e-8;
const NORMALIZE_BOUND_EPS: f64 = 1e-9;
const DENORMALIZE_CLAMP_EPS: f64 = 1e-6;

/// Stats key `"<market>|<timescale>|<feature>"`.
pub fn stats_key(market: Market, timescale: Timescale, feature: &str) -> String {
    format!("{}|{}|{}", market.as_str(), timescale.as_str(), feature)
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    median_of_sorted(&sorted)
}

/// Robust scale `1.4826 * MAD`; falls back to `max(std, 1e-8)` if it is below `1e-8`.
pub fn fit_mad_scale(x: &[f64]) -> Result<f64> {
    if x.is_empty() {
        return Err(Error::InvalidData(
            "cannot fit a MAD scale on an empty sample".to_string(),
        ));
    }
    let med = median(x);
    let deviations: Vec<f64> = x.iter().map(|v| (v - med).abs()).collect();
    let scale = MAD_CONST * median(&deviations);
    if scale < MIN_SCALE {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        return Ok(var.sqrt().max(MIN_SCALE));
    }
    Ok(scale)
}

/// Soft-clipped robust normalization: `c * tanh((x / scale) / c)`, clipped to the open
/// interval `(-c, c)` by a relative margin of `1e-9`. NaN passes through.
pub fn normalize(x: f64, scale: f64, c: f64) -> f64 {
    let xp = x / scale;
    let y = c * (xp / c).tanh();
    let bound = c * (1.0 - NORMALIZE_BOUND_EPS);
    clip(y, -bound, bound)
}

/// Inverse of [`normalize`]. `y / c` is clamped to `±(1 - 1e-6)` to keep `atanh` finite.
pub fn denormalize(y: f64, scale: f64, c: f64) -> f64 {
    let r = clip(
        y / c,
        -(1.0 - DENORMALIZE_CLAMP_EPS),
        1.0 - DENORMALIZE_CLAMP_EPS,
    );
    r.atanh() * c * scale
}

/// Apply [`normalize`] to every element of `x`.
pub fn normalize_slice(x: &[f64], scale: f64, c: f64) -> Vec<f64> {
    x.iter().map(|&v| normalize(v, scale, c)).collect()
}

/// Apply [`denormalize`] to every element of `y`.
pub fn denormalize_slice(y: &[f64], scale: f64, c: f64) -> Vec<f64> {
    y.iter().map(|&v| denormalize(v, scale, c)).collect()
}

pub(crate) fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

pub(crate) fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

pub(crate) fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    np_min(np_max(v, lo), hi)
}
