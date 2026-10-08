use super::bar::Ohlcv;
use super::features::{FEATURE_NAMES, Features, N_FEATURES, invert_features};
use super::normalize::{DEFAULT_SOFT_CLIP_C, clip, denormalize, np_max, np_min};
use super::stats::Stats;
use crate::market::{Market, Timescale};
use crate::{Error, Result};

/// Quantile levels of the summary bands.
pub const QUANTILES: [f64; 5] = [0.05, 0.25, 0.50, 0.75, 0.95];

/// A set of sampled feature paths, `[K][Lh][5]` in `FEATURE_NAMES` order.
pub type FeaturePaths = Vec<Vec<[f64; N_FEATURES]>>;

/// Denormalize sampled paths per feature using `stats`; the shape is unchanged.
pub fn denormalize_paths(
    paths: &[Vec<[f32; N_FEATURES]>],
    market: Market,
    timescale: Timescale,
    stats: &Stats,
    c: f64,
) -> Result<FeaturePaths> {
    let scales = stats.feature_scales(market, timescale)?;
    Ok(paths
        .iter()
        .map(|path| {
            path.iter()
                .map(|row| {
                    let mut out = [0.0; N_FEATURES];
                    for j in 0..N_FEATURES {
                        out[j] = denormalize(f64::from(row[j]), scales[j], c);
                    }
                    out
                })
                .collect()
        })
        .collect())
}

/// [`denormalize_paths`] with the default soft-clip constant.
pub fn denormalize_paths_default(
    paths: &[Vec<[f32; N_FEATURES]>],
    market: Market,
    timescale: Timescale,
    stats: &Stats,
) -> Result<FeaturePaths> {
    denormalize_paths(paths, market, timescale, stats, DEFAULT_SOFT_CLIP_C)
}

/// Convert denormalized feature paths to OHLCV paths.
///
/// Each path is rebuilt independently from `prev_close` / `prev_ema`, with `r_up` and `r_dn`
/// clamped to `>= 0`.
pub fn paths_to_ohlcv(
    feats: &[Vec<[f64; N_FEATURES]>],
    prev_close: f64,
    prev_ema: f64,
    half_life_bars: f64,
) -> Result<Vec<Ohlcv>> {
    feats
        .iter()
        .map(|path| {
            let mut f = Features::from_rows(path);
            for x in f.r_up.iter_mut().chain(f.r_dn.iter_mut()) {
                *x = np_max(*x, 0.0);
            }
            invert_features(&f, prev_close, prev_ema, half_life_bars)
        })
        .collect()
}

/// Clamp O/H/L/C bar by bar to `[pc*(1-rate), pc*(1+rate)]`, where `pc` is the previous
/// projected close, then make H/L consistent with O/C. Returns copies unchanged when
/// `limit_rate` is `None`.
pub fn apply_limit_projection(ohlcv: &Ohlcv, prev_close: f64, limit_rate: Option<f64>) -> Ohlcv {
    let mut out = ohlcv.clone();
    let Some(rate) = limit_rate else {
        return out;
    };
    let mut pc = prev_close;
    for t in 0..out.len() {
        let lo = pc * (1.0 - rate);
        let hi = pc * (1.0 + rate);
        out.open[t] = clip(out.open[t], lo, hi);
        out.close[t] = clip(out.close[t], lo, hi);
        out.high[t] = clip(out.high[t], lo, hi);
        out.low[t] = clip(out.low[t], lo, hi);
        out.high[t] = np_max(out.high[t], np_max(out.open[t], out.close[t]));
        out.low[t] = np_min(out.low[t], np_min(out.open[t], out.close[t]));
        pc = out.close[t];
    }
    out
}

/// Per-position quantile bands along the sample axis, each of length `Lh`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Band {
    /// 5% quantile.
    pub q05: Vec<f64>,
    /// 25% quantile.
    pub q25: Vec<f64>,
    /// 50% quantile (median).
    pub q50: Vec<f64>,
    /// 75% quantile.
    pub q75: Vec<f64>,
    /// 95% quantile.
    pub q95: Vec<f64>,
}

/// Per-path scalar statistic: the samples plus their quantiles.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dist {
    /// One value per path.
    pub samples: Vec<f64>,
    /// 5% quantile.
    pub q05: f64,
    /// 25% quantile.
    pub q25: f64,
    /// 50% quantile (median).
    pub q50: f64,
    /// 75% quantile.
    pub q75: f64,
    /// 95% quantile.
    pub q95: f64,
}

/// Summary statistics of a set of paths.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PathSummary {
    /// Per-bar quantile bands of the close log return, each of length `Lh`.
    pub close_return_bands: Band,
    /// `(channel name, per-bar quantile bands)`; feature names for feature paths, or
    /// `open/high/low/close/volume` for OHLCV paths.
    pub channel_bands: Vec<(String, Band)>,
    /// Cumulative log return over `Lh`.
    pub terminal_return: Dist,
    /// Per-path maximum drawdown of the cumulative log return, starting from 0.
    pub max_drawdown: Dist,
    /// Per-path sample standard deviation (ddof = 1) of per-bar log returns; 0 when `Lh == 1`.
    pub realized_vol: Dist,
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// Linear-interpolation quantile of an unsorted sample, matching `numpy.quantile`.
/// Returns NaN if the sample contains a NaN, and an error if it is empty.
pub fn quantile(values: &[f64], q: f64) -> Result<f64> {
    if values.is_empty() {
        return Err(Error::InvalidData(
            "cannot take a quantile of an empty sample".to_string(),
        ));
    }
    if values.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let virtual_index = (sorted.len() - 1) as f64 * q;
    let lo = virtual_index.floor();
    let t = virtual_index - lo;
    let lo_i = lo as usize;
    let hi_i = (lo_i + 1).min(sorted.len() - 1);
    Ok(lerp(sorted[lo_i], sorted[hi_i], t))
}

fn band(rows: &[Vec<f64>], len: usize) -> Result<Band> {
    let mut b = Band::default();
    for pos in 0..len {
        let column: Vec<f64> = rows.iter().map(|r| r[pos]).collect();
        b.q05.push(quantile(&column, QUANTILES[0])?);
        b.q25.push(quantile(&column, QUANTILES[1])?);
        b.q50.push(quantile(&column, QUANTILES[2])?);
        b.q75.push(quantile(&column, QUANTILES[3])?);
        b.q95.push(quantile(&column, QUANTILES[4])?);
    }
    Ok(b)
}

fn dist(samples: Vec<f64>) -> Result<Dist> {
    Ok(Dist {
        q05: quantile(&samples, QUANTILES[0])?,
        q25: quantile(&samples, QUANTILES[1])?,
        q50: quantile(&samples, QUANTILES[2])?,
        q75: quantile(&samples, QUANTILES[3])?,
        q95: quantile(&samples, QUANTILES[4])?,
        samples,
    })
}

fn summarize_returns(
    r: &[Vec<f64>],
    lh: usize,
    channel_bands: Vec<(String, Band)>,
) -> Result<PathSummary> {
    let terminal: Vec<f64> = r.iter().map(|row| row.iter().sum()).collect();
    let max_dd: Vec<f64> = r
        .iter()
        .map(|row| {
            let mut cum = 0.0;
            let mut run_max = f64::NEG_INFINITY;
            let mut dd = f64::NEG_INFINITY;
            for &x in row {
                cum += x;
                run_max = np_max(run_max, np_max(cum, 0.0));
                dd = np_max(dd, run_max - cum);
            }
            dd
        })
        .collect();
    let rv: Vec<f64> = r
        .iter()
        .map(|row| {
            if lh > 1 {
                let n = row.len() as f64;
                let mean = row.iter().sum::<f64>() / n;
                let ss = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
                (ss / (n - 1.0)).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    Ok(PathSummary {
        close_return_bands: band(r, lh)?,
        channel_bands,
        terminal_return: dist(terminal)?,
        max_drawdown: dist(max_dd)?,
        realized_vol: dist(rv)?,
    })
}

fn check_nonempty(k: usize, lh: usize) -> Result<()> {
    if k == 0 || lh == 0 {
        return Err(Error::InvalidData(
            "paths must contain at least one path and one bar".to_string(),
        ));
    }
    Ok(())
}

/// Summarize denormalized feature paths `[K][Lh][5]`. The close-to-close log return is
/// `r_gap + r_body`.
pub fn summarize_feature_paths(paths: &[Vec<[f64; N_FEATURES]>]) -> Result<PathSummary> {
    let k = paths.len();
    let lh = paths.first().map_or(0, Vec::len);
    check_nonempty(k, lh)?;
    if paths.iter().any(|p| p.len() != lh) {
        return Err(Error::InvalidData(
            "all paths must have the same length".to_string(),
        ));
    }
    let r: Vec<Vec<f64>> = paths
        .iter()
        .map(|p| p.iter().map(|row| row[0] + row[1]).collect())
        .collect();
    let mut channels = Vec::with_capacity(N_FEATURES);
    for (j, name) in FEATURE_NAMES.iter().enumerate() {
        let rows: Vec<Vec<f64>> = paths
            .iter()
            .map(|p| p.iter().map(|row| row[j]).collect())
            .collect();
        channels.push(((*name).to_string(), band(&rows, lh)?));
    }
    summarize_returns(&r, lh, channels)
}

/// Summarize OHLCV paths. The first return uses `prev_close`, or `ln(C_0 / O_0)` when it is
/// `None`; later returns are differences of log closes.
pub fn summarize_ohlcv_paths(paths: &[Ohlcv], prev_close: Option<f64>) -> Result<PathSummary> {
    let k = paths.len();
    let lh = paths.first().map_or(0, Ohlcv::len);
    check_nonempty(k, lh)?;
    if paths.iter().any(|p| p.len() != lh) {
        return Err(Error::InvalidData(
            "all paths must have the same length".to_string(),
        ));
    }
    let r: Vec<Vec<f64>> = paths
        .iter()
        .map(|p| {
            let mut row = Vec::with_capacity(lh);
            row.push(match prev_close {
                Some(pc) => (p.close[0] / pc).ln(),
                None => (p.close[0] / p.open[0]).ln(),
            });
            for t in 1..lh {
                row.push(p.close[t].ln() - p.close[t - 1].ln());
            }
            row
        })
        .collect();
    let mut channels = Vec::with_capacity(5);
    for name in ["open", "high", "low", "close", "volume"] {
        let rows: Vec<Vec<f64>> = paths
            .iter()
            .map(|p| match name {
                "open" => p.open.clone(),
                "high" => p.high.clone(),
                "low" => p.low.clone(),
                "close" => p.close.clone(),
                _ => p.volume.clone(),
            })
            .collect();
        channels.push((name.to_string(), band(&rows, lh)?));
    }
    summarize_returns(&r, lh, channels)
}
