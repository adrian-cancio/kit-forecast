use super::bar::{Bar, Ohlcv};
use crate::{Error, Result};

/// Number of candle features per bar.
pub const N_FEATURES: usize = 5;

/// Feature order; also the feature part of the stats keys.
pub const FEATURE_NAMES: [&str; N_FEATURES] = ["r_gap", "r_body", "r_up", "r_dn", "v"];

/// Candle features of a bar series (float64), one entry per bar.
///
/// ```text
/// r_gap  = ln(O_t / C_{t-1})             open gap (0 for the first bar)
/// r_body = ln(C_t / O_t)                 body
/// r_up   = ln(H_t / max(O_t, C_t))       upper shadow
/// r_dn   = ln(min(O_t, C_t) / L_t)       lower shadow
/// v      = ln((V_t + 1) / (EMA_t + 1))   log volume relative to its EMA
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Features {
    /// Open gap.
    pub r_gap: Vec<f64>,
    /// Candle body.
    pub r_body: Vec<f64>,
    /// Upper shadow.
    pub r_up: Vec<f64>,
    /// Lower shadow.
    pub r_dn: Vec<f64>,
    /// Log volume relative to its EMA.
    pub v: Vec<f64>,
}

impl Features {
    /// Number of bars.
    pub fn len(&self) -> usize {
        self.r_gap.len()
    }

    /// Whether there are no bars.
    pub fn is_empty(&self) -> bool {
        self.r_gap.is_empty()
    }

    /// Feature column by index in [`FEATURE_NAMES`] order.
    pub fn column(&self, index: usize) -> Option<&[f64]> {
        match index {
            0 => Some(&self.r_gap),
            1 => Some(&self.r_body),
            2 => Some(&self.r_up),
            3 => Some(&self.r_dn),
            4 => Some(&self.v),
            _ => None,
        }
    }

    /// Features of bar `t` in [`FEATURE_NAMES`] order.
    pub fn row(&self, t: usize) -> [f64; N_FEATURES] {
        [
            self.r_gap[t],
            self.r_body[t],
            self.r_up[t],
            self.r_dn[t],
            self.v[t],
        ]
    }

    /// All rows in [`FEATURE_NAMES`] order.
    pub fn to_rows(&self) -> Vec<[f64; N_FEATURES]> {
        (0..self.len()).map(|t| self.row(t)).collect()
    }

    /// Build from rows in [`FEATURE_NAMES`] order.
    pub fn from_rows(rows: &[[f64; N_FEATURES]]) -> Self {
        Self {
            r_gap: rows.iter().map(|r| r[0]).collect(),
            r_body: rows.iter().map(|r| r[1]).collect(),
            r_up: rows.iter().map(|r| r[2]).collect(),
            r_dn: rows.iter().map(|r| r[3]).collect(),
            v: rows.iter().map(|r| r[4]).collect(),
        }
    }
}

/// EMA smoothing factor `1 - 2^(-1/half_life_bars)`. The half-life must be positive.
pub fn ema_alpha(half_life_bars: f64) -> Result<f64> {
    if half_life_bars.is_nan() || half_life_bars <= 0.0 {
        return Err(Error::InvalidConfig(format!(
            "half_life_bars must be positive, got {half_life_bars}"
        )));
    }
    Ok(1.0 - (-1.0 / half_life_bars).exp2())
}

/// Volume EMA where `EMA[t]` depends only on `V[0..t-1]` and `EMA[0] = V[0]`.
///
/// The recursion reproduces pandas `ewm(alpha, adjust=False).mean()` shifted by one bar.
pub fn compute_volume_ema(volume: &[f64], half_life_bars: f64) -> Result<Vec<f64>> {
    let alpha = ema_alpha(half_life_bars)?;
    let n = volume.len();
    let mut ema = Vec::with_capacity(n);
    if n == 0 {
        return Ok(ema);
    }
    let old_wt_factor = 1.0 - alpha;
    let mut weighted = volume[0];
    ema.push(volume[0]);
    for &cur in &volume[1..n] {
        ema.push(weighted);
        if weighted != cur {
            weighted = (old_wt_factor * weighted + alpha * cur) / (old_wt_factor + alpha);
        }
    }
    Ok(ema)
}

/// OHLCV bars to the five candle features.
///
/// `half_life_bars` is the volume EMA half-life in bars. Prices must be positive.
pub fn compute_features(bars: &[Bar], half_life_bars: f64) -> Result<Features> {
    let n = bars.len();
    let volume: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let ema = compute_volume_ema(&volume, half_life_bars)?;

    let mut f = Features {
        r_gap: vec![0.0; n],
        r_body: Vec::with_capacity(n),
        r_up: Vec::with_capacity(n),
        r_dn: Vec::with_capacity(n),
        v: Vec::with_capacity(n),
    };
    for t in 0..n {
        let b = &bars[t];
        if t > 0 {
            f.r_gap[t] = (b.open / bars[t - 1].close).ln();
        }
        f.r_body.push((b.close / b.open).ln());
        f.r_up.push((b.high / b.open.max(b.close)).ln());
        f.r_dn.push((b.open.min(b.close) / b.low).ln());
        f.v.push(((b.volume + 1.0) / (ema[t] + 1.0)).ln());
    }
    Ok(f)
}

/// Rebuild OHLCV from features by chaining from the previous bar.
///
/// `prev_close` is the close before the first bar (for a round trip with
/// [`compute_features`] pass the first open). `prev_ema` is the volume EMA at the first bar
/// (for a round trip pass the first volume). `half_life_bars` must match the encoding.
/// Volume is clamped to `>= 0`.
pub fn invert_features(
    features: &Features,
    prev_close: f64,
    prev_ema: f64,
    half_life_bars: f64,
) -> Result<Ohlcv> {
    let n = features.len();
    if [
        features.r_body.len(),
        features.r_up.len(),
        features.r_dn.len(),
        features.v.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(Error::InvalidData(
            "feature columns must all have the same length".to_string(),
        ));
    }
    let alpha = ema_alpha(half_life_bars)?;
    let mut out = Ohlcv {
        open: Vec::with_capacity(n),
        high: Vec::with_capacity(n),
        low: Vec::with_capacity(n),
        close: Vec::with_capacity(n),
        volume: Vec::with_capacity(n),
    };
    let mut c_prev = prev_close;
    let mut ema = prev_ema;
    for t in 0..n {
        let o = c_prev * features.r_gap[t].exp();
        let c = o * features.r_body[t].exp();
        let h = o.max(c) * features.r_up[t].exp();
        let l = o.min(c) * (-features.r_dn[t]).exp();
        let raw_vol = (ema + 1.0) * features.v[t].exp() - 1.0;
        let vol = if 0.0 > raw_vol { 0.0 } else { raw_vol };
        ema = (1.0 - alpha) * ema + alpha * vol;
        c_prev = c;
        out.open.push(o);
        out.high.push(h);
        out.low.push(l);
        out.close.push(c);
        out.volume.push(vol);
    }
    Ok(out)
}
