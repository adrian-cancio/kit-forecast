//! Data pipeline: candle features, robust normalization, model-input windows and
//! post-processing of sampled paths.
//!
//! - [`Bar`] / [`Ohlcv`]: raw OHLCV rows and columns
//! - features: [`compute_features`] and its inverse [`invert_features`]
//! - normalize: MAD scaling with tanh soft clipping, [`normalize`] / [`denormalize`]
//! - [`Stats`]: `stats.json` scales
//! - [`PreparedWindow`]: inference inputs for one window
//! - postprocess: [`denormalize_paths`], [`paths_to_ohlcv`], [`apply_limit_projection`],
//!   path summaries

mod bar;
mod features;
mod normalize;
mod postprocess;
mod stats;
#[cfg(test)]
mod tests;
mod window;

pub use bar::{Bar, Ohlcv};
pub use features::{
    FEATURE_NAMES, Features, N_FEATURES, compute_features, compute_volume_ema, ema_alpha,
    invert_features,
};
pub use normalize::{
    DEFAULT_SOFT_CLIP_C, denormalize, denormalize_slice, fit_mad_scale, normalize, normalize_slice,
    stats_key,
};
pub use postprocess::{
    Band, Dist, FeaturePaths, PathSummary, QUANTILES, apply_limit_projection, denormalize_paths,
    denormalize_paths_default, paths_to_ohlcv, quantile, summarize_feature_paths,
    summarize_ohlcv_paths,
};
pub use stats::Stats;
pub use window::{Condition, PreparedWindow};
