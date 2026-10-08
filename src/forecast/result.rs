use crate::data::{Bar, FeaturePaths, Ohlcv, PathSummary, summarize_ohlcv_paths};
use crate::market::Timestamp;
use crate::{Error, Result};

/// Sampled future candles for one request.
///
/// All price paths start from the last close of the supplied history ([`Forecast::prev_close`]);
/// the observed history is untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    /// Close time of the last historical bar (the anchor).
    pub anchor: Timestamp,
    /// Close times of the forecast bars, projected on the exchange calendar.
    pub timestamps: Vec<Timestamp>,
    /// Sampled OHLCV paths, `paths[k]` has one entry per forecast bar.
    pub paths: Vec<Ohlcv>,
    /// Denormalized model output per path and bar in `FEATURE_NAMES` order:
    /// `r_gap, r_body, r_up, r_dn, v` (log ratios and relative log volume).
    pub features: FeaturePaths,
    /// Last observed close.
    pub prev_close: f64,
    /// Volume EMA state at the first forecast bar.
    pub prev_ema: f64,
    /// Volume EMA half-life in bars.
    pub half_life_bars: f64,
}

impl Forecast {
    /// Number of sampled paths.
    pub fn n_paths(&self) -> usize {
        self.paths.len()
    }

    /// Number of forecast bars.
    pub fn horizon(&self) -> usize {
        self.timestamps.len()
    }

    /// Path `k` as timestamped bars.
    pub fn bars(&self, k: usize) -> Result<Vec<Bar>> {
        self.paths
            .get(k)
            .ok_or_else(|| {
                Error::InvalidData(format!("path {k} out of range ({})", self.n_paths()))
            })?
            .to_bars(&self.timestamps)
    }

    /// Cumulative close log return of every path against [`Forecast::prev_close`], `[K][Lh]`.
    pub fn cumulative_log_returns(&self) -> Vec<Vec<f64>> {
        self.paths
            .iter()
            .map(|p| p.close.iter().map(|c| (c / self.prev_close).ln()).collect())
            .collect()
    }

    /// Terminal cumulative log return of every path.
    pub fn terminal_returns(&self) -> Vec<f64> {
        self.paths
            .iter()
            .map(|p| p.close.last().map_or(0.0, |c| (c / self.prev_close).ln()))
            .collect()
    }

    /// Mean over paths of the summed per-bar log returns (the `return` signal of the upstream
    /// evaluation).
    pub fn mean_return(&self) -> f64 {
        let r = self.terminal_returns();
        r.iter().sum::<f64>() / r.len().max(1) as f64
    }

    /// Index of the representative path: the one with the median terminal return, as used by
    /// the upstream plots.
    pub fn representative_path(&self) -> usize {
        let term = self.terminal_returns();
        let mut order: Vec<usize> = (0..term.len()).collect();
        order.sort_by(|&a, &b| term[a].total_cmp(&term[b]));
        order.get(term.len() / 2).copied().unwrap_or(0)
    }

    /// Quantile bands, terminal return, drawdown and volatility distributions of the paths.
    pub fn summary(&self) -> Result<PathSummary> {
        summarize_ohlcv_paths(&self.paths, Some(self.prev_close))
    }
}
