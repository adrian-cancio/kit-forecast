use super::bar::Bar;
use super::features::{FEATURE_NAMES, N_FEATURES, compute_features, compute_volume_ema, ema_alpha};
use super::normalize::normalize;
use super::stats::Stats;
use crate::market::{
    CLOCK_DIM, Market, SESS_DIM, Timescale, Timestamp, TradingCalendar, YDAY_DIM,
    attach_window_calendar, bars_per_day, gen_future_timestamps,
};
use crate::{Error, Result};

/// Condition tensors for one window of length `L = Lc + Lh`, batch size 1.
///
/// Per-bar vectors have length `L`; padded context positions keep zero calendar features,
/// zero event code and `pad_mask == false`.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    /// Clock Fourier features, `[L][16]`.
    pub clock: Vec<[f32; CLOCK_DIM]>,
    /// In-session position features, `[L][2]`.
    pub sess: Vec<[f32; SESS_DIM]>,
    /// Day of week (Monday = 0), `[L]`.
    pub dow: Vec<i64>,
    /// Event code (0 none, 1 gap, 2 first bar), `[L]`.
    pub event: Vec<i64>,
    /// Month (January = 0), `[L]`.
    pub month: Vec<i64>,
    /// Day-of-year Fourier features, `[L][8]`.
    pub yday: Vec<[f32; YDAY_DIM]>,
    /// Market embedding id.
    pub market: i64,
    /// Sector embedding id; 0 means unknown.
    pub sector: i64,
    /// Instrument embedding id; 0 means unknown.
    pub inst: i64,
    /// Timescale embedding id.
    pub scale: i64,
    /// `true` for valid positions, `false` for padded context positions, `[L]`.
    pub pad_mask: Vec<bool>,
}

/// Model inputs built from raw K-line history (batch size 1).
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedWindow {
    /// Normalized features of the context, `[Lc][5]` in `FEATURE_NAMES` order. The first
    /// `n_pad` rows are zero padding.
    pub x_ctx: Vec<[f32; N_FEATURES]>,
    /// Condition vectors for the whole window, `L = Lc + Lh`.
    pub cond: Condition,
    /// Close of the last context bar.
    pub prev_close: f64,
    /// Volume EMA state at the first future bar.
    pub prev_ema: f64,
    /// Volume EMA half-life in bars.
    pub half_life_bars: f64,
    /// Timestamps of the valid (non-padded) context bars.
    pub ctx_timestamps: Vec<Timestamp>,
    /// Timestamps of the `Lh` future bars.
    pub future_timestamps: Vec<Timestamp>,
    /// Number of padded bars, `Lc - ctx_timestamps.len()`.
    pub n_pad: usize,
}

impl PreparedWindow {
    /// Build inputs for a single window.
    ///
    /// `bars` is the full available history in ascending time order; features use all of it
    /// so the volume EMA is warmed up, and the last `lc` bars form the context. With fewer
    /// than `lc` bars the front is padded. At least two bars are required.
    ///
    /// `half_life_days` is converted to bars with `bars_per_day(market, scale)`; `stats`
    /// must hold the five `market|scale|feature` scales; `sector_id` / `inst_id` are
    /// embedding ids (0 = unknown). `calendar` drives future timestamps and event detection.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        bars: &[Bar],
        market: Market,
        scale: Timescale,
        lc: usize,
        lh: usize,
        stats: &Stats,
        sector_id: i64,
        inst_id: i64,
        calendar: &TradingCalendar,
        soft_clip_c: f64,
        half_life_days: f64,
    ) -> Result<PreparedWindow> {
        let n_total = bars.len();
        if n_total < 2 {
            return Err(Error::InvalidData(format!(
                "history has only {n_total} bars, cannot build an inference window"
            )));
        }

        let half_life_bars = half_life_days * bars_per_day(market, scale) as f64;
        let feats = compute_features(bars, half_life_bars)?;
        let scales = stats.feature_scales(market, scale)?;
        let x_all: Vec<[f32; N_FEATURES]> = (0..n_total)
            .map(|t| {
                let row = feats.row(t);
                let mut out = [0f32; N_FEATURES];
                for j in 0..N_FEATURES {
                    out[j] = normalize(row[j], scales[j], soft_clip_c) as f32;
                }
                out
            })
            .collect();
        debug_assert_eq!(FEATURE_NAMES.len(), N_FEATURES);

        let n_eff = lc.min(n_total);
        let n_pad = lc - n_eff;
        let ctx_ts: Vec<Timestamp> = bars[n_total - n_eff..]
            .iter()
            .map(|b| b.timestamp)
            .collect();
        let mut x_ctx = vec![[0f32; N_FEATURES]; lc];
        x_ctx[n_pad..].copy_from_slice(&x_all[n_total - n_eff..]);

        let last_ctx = *ctx_ts
            .last()
            .ok_or_else(|| Error::InvalidData("context length must be positive".to_string()))?;
        let future_timestamps = gen_future_timestamps(last_ctx, lh, market, scale, calendar);
        if future_timestamps.len() != lh {
            return Err(Error::InvalidData(format!(
                "could only generate {} of {lh} future timestamps",
                future_timestamps.len()
            )));
        }
        let mut window_ts = ctx_ts.clone();
        window_ts.extend_from_slice(&future_timestamps);
        let prev_ts = if n_total > n_eff {
            Some(bars[n_total - n_eff - 1].timestamp)
        } else {
            None
        };
        let cal =
            attach_window_calendar(&window_ts, market, scale, n_eff, None, prev_ts, calendar)?;

        let total = lc + lh;
        let mut cond = Condition {
            clock: vec![[0f32; CLOCK_DIM]; total],
            sess: vec![[0f32; SESS_DIM]; total],
            dow: vec![0; total],
            event: vec![0; total],
            month: vec![0; total],
            yday: vec![[0f32; YDAY_DIM]; total],
            market: market.id(),
            sector: sector_id,
            inst: inst_id,
            scale: scale.id(),
            pad_mask: vec![true; total],
        };
        cond.clock[n_pad..].copy_from_slice(&cal.features.clock);
        cond.sess[n_pad..].copy_from_slice(&cal.features.sess);
        cond.dow[n_pad..].copy_from_slice(&cal.features.dow);
        cond.month[n_pad..].copy_from_slice(&cal.features.month);
        cond.yday[n_pad..].copy_from_slice(&cal.features.yday);
        cond.event[n_pad..].copy_from_slice(&cal.event);
        for m in cond.pad_mask.iter_mut().take(n_pad) {
            *m = false;
        }

        let volume: Vec<f64> = bars.iter().map(|b| b.volume).collect();
        let ema = compute_volume_ema(&volume, half_life_bars)?;
        let alpha = ema_alpha(half_life_bars)?;
        let last = n_total - 1;
        let prev_ema = (1.0 - alpha) * ema[last] + alpha * volume[last];

        Ok(PreparedWindow {
            x_ctx,
            cond,
            prev_close: bars[last].close,
            prev_ema,
            half_life_bars,
            ctx_timestamps: ctx_ts,
            future_timestamps,
            n_pad,
        })
    }

    /// Context length `Lc` (including padding).
    pub fn lc(&self) -> usize {
        self.x_ctx.len()
    }

    /// Horizon length `Lh`.
    pub fn lh(&self) -> usize {
        self.future_timestamps.len()
    }

    /// Total window length `L = Lc + Lh`.
    pub fn len(&self) -> usize {
        self.cond.pad_mask.len()
    }

    /// Whether the window has no positions.
    pub fn is_empty(&self) -> bool {
        self.cond.pad_mask.is_empty()
    }
}
