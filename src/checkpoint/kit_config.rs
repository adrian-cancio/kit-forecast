use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::Result;
use super::config::ModelConfig;
use super::pickle::Dict;
use crate::Error;

/// Supported timescale names, from the shortest to the longest.
pub const TIMESCALES: [&str; 7] = ["1m", "5m", "15m", "30m", "1h", "2h", "1d"];

const DEFAULT_CONTEXT_LEN: usize = 1024;
const DEFAULT_HORIZON_LEN: usize = 128;
const DEFAULT_TRUNC_MIN: usize = 256;
const DEFAULT_SOFT_CLIP_C: f64 = 6.0;
const DEFAULT_VOL_EMA_HALF_LIFE_DAYS: f64 = 5.0;

/// Context and horizon lengths of one timescale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSpec {
    /// Number of context bars (`Lc`).
    pub context_len: usize,
    /// Number of forecast bars (`Lh`).
    pub horizon_len: usize,
    /// Minimum truncated context length used in training.
    pub trunc_min: usize,
}

impl WindowSpec {
    /// Total sequence length, `context_len + horizon_len`.
    pub fn total_len(&self) -> usize {
        self.context_len + self.horizon_len
    }

    fn validate(&self, what: &str) -> Result<()> {
        if self.context_len == 0 || self.horizon_len == 0 {
            return Err(Error::InvalidConfig(format!(
                "{what}: context_len and horizon_len must be positive"
            )));
        }
        if self.trunc_min == 0 || self.trunc_min > self.context_len {
            return Err(Error::InvalidConfig(format!(
                "{what}: trunc_min {} must be in 1..={}",
                self.trunc_min, self.context_len
            )));
        }
        Ok(())
    }
}

/// Per-timescale windows with the global window as fallback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Windows {
    /// Global window from `data.Lc`, `data.Lh` and `data.trunc_min`.
    pub default: WindowSpec,
    /// Entries of `data.scale_windows`, keyed by timescale name.
    pub by_timescale: BTreeMap<String, WindowSpec>,
}

impl Windows {
    /// Builds the table from a `scale_windows` string such as
    /// `"1m:1200/120/300,1d:250/20"` and the global window values.
    ///
    /// This follows the upstream `parse_scale_windows`, `window_for` and `trunc_min_for`:
    /// a missing `trunc_min` becomes `min(global trunc_min, Lc of that scale)`, and unlisted
    /// timescales use the global values.
    pub fn parse(
        scale_windows: &str,
        context_len: usize,
        horizon_len: usize,
        trunc_min: usize,
    ) -> Result<Self> {
        let default = WindowSpec {
            context_len,
            horizon_len,
            trunc_min: trunc_min.min(context_len),
        };
        let mut by_timescale = BTreeMap::new();
        for item in scale_windows.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let Some((scale, rest)) = item.split_once(':') else {
                return Err(window_error(format!(
                    "invalid item (missing ':'): {item:?}; {HELP}"
                )));
            };
            let (scale, rest) = (scale.trim(), rest.trim());
            if !TIMESCALES.contains(&scale) {
                return Err(window_error(format!(
                    "unknown timescale {scale:?}; choose from {TIMESCALES:?}"
                )));
            }
            if by_timescale.contains_key(scale) {
                return Err(window_error(format!(
                    "timescale {scale:?} appears more than once"
                )));
            }
            if rest.is_empty() {
                return Err(window_error(format!(
                    "scale {scale:?} has no value; {HELP}"
                )));
            }
            let parts: Vec<&str> = rest.split('/').map(str::trim).collect();
            if parts.len() != 2 && parts.len() != 3 {
                return Err(window_error(format!(
                    "scale {scale:?} has value {rest:?}, should be 'Lc/Lh' or 'Lc/Lh/trunc_min'"
                )));
            }
            let mut values = [0usize; 3];
            for (slot, (raw, field)) in
                values
                    .iter_mut()
                    .zip(parts.iter().zip(["Lc", "Lh", "trunc_min"]))
            {
                let parsed: i64 = raw.parse().map_err(|_| {
                    window_error(format!("{scale}.{field} is not an integer: {raw:?}"))
                })?;
                if parsed <= 0 {
                    return Err(window_error(format!(
                        "{scale}.{field} must be a positive integer, got {parsed}"
                    )));
                }
                *slot = usize::try_from(parsed).map_err(|_| {
                    window_error(format!("{scale}.{field} = {parsed} is out of range"))
                })?;
            }
            let [lc, lh, table_trunc] = values;
            let table_trunc = if parts.len() == 3 { table_trunc } else { 0 };
            if table_trunc > lc {
                return Err(window_error(format!(
                    "{scale}.trunc_min({table_trunc}) must not exceed Lc({lc})"
                )));
            }
            let resolved = if table_trunc > 0 {
                table_trunc
            } else {
                trunc_min.min(lc)
            };
            by_timescale.insert(
                scale.to_string(),
                WindowSpec {
                    context_len: lc,
                    horizon_len: lh,
                    trunc_min: resolved,
                },
            );
        }
        let windows = Self {
            default,
            by_timescale,
        };
        windows.validate()?;
        Ok(windows)
    }

    /// Window of `timescale`, falling back to [`Windows::default`] when it has no entry.
    pub fn get(&self, timescale: &str) -> WindowSpec {
        self.by_timescale
            .get(timescale)
            .copied()
            .unwrap_or(self.default)
    }

    fn validate(&self) -> Result<()> {
        self.default.validate("windows.default")?;
        for (name, spec) in &self.by_timescale {
            if !TIMESCALES.contains(&name.as_str()) {
                return Err(Error::InvalidConfig(format!(
                    "windows.by_timescale has unknown timescale {name:?}"
                )));
            }
            spec.validate(&format!("windows.by_timescale[{name}]"))?;
        }
        Ok(())
    }
}

const HELP: &str = "format 'scale:Lc/Lh[/trunc_min]', items separated by commas";

fn window_error(message: String) -> Error {
    Error::InvalidConfig(format!("data.scale_windows: {message}"))
}

/// Sampling defaults from the `infer` section of the checkpoint configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferDefaults {
    /// Number of integration steps of the flow-matching sampler.
    pub steps: usize,
    /// Solver name, `euler` or `heun`.
    pub solver: String,
    /// Number of sampled paths.
    pub n_paths: usize,
    /// Classifier-free guidance weight; `1.0` disables it.
    pub cfg_w: f64,
    /// History guidance weight; `1.0` disables it.
    pub hist_cfg_w: f64,
}

impl Default for InferDefaults {
    fn default() -> Self {
        Self {
            steps: 32,
            solver: "euler".to_string(),
            n_paths: 64,
            cfg_w: 1.0,
            hist_cfg_w: 1.0,
        }
    }
}

impl InferDefaults {
    fn from_pickle(dict: &Dict<'_>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            steps: dict.usize_or("steps", defaults.steps)?,
            solver: dict.string_or("solver", &defaults.solver)?,
            n_paths: dict.usize_or("n_paths", defaults.n_paths)?,
            cfg_w: dict.f64_or("cfg_w", defaults.cfg_w)?,
            hist_cfg_w: dict.f64_or("hist_cfg_w", defaults.hist_cfg_w)?,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.steps == 0 || self.n_paths == 0 {
            return Err(Error::InvalidConfig(
                "infer.steps and infer.n_paths must be positive".to_string(),
            ));
        }
        for (name, value) in [("cfg_w", self.cfg_w), ("hist_cfg_w", self.hist_cfg_w)] {
            if !value.is_finite() {
                return Err(Error::InvalidConfig(format!(
                    "infer.{name} must be finite, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// Everything the checkpoint's `config` section says about inference: architecture,
/// per-timescale windows, normalization constants and sampling defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KitConfig {
    /// Model architecture (`config.model`).
    pub model: ModelConfig,
    /// Context and horizon windows (`config.data` `Lc`, `Lh`, `trunc_min`, `scale_windows`).
    pub windows: Windows,
    /// Soft clip constant `c` in `c * tanh(x / c)` (`config.data.soft_clip_c`).
    pub soft_clip_c: f64,
    /// Volume EMA half-life in trading days (`config.data.vol_ema_half_life_days`).
    pub vol_ema_half_life_days: f64,
    /// Sampling defaults (`config.infer`).
    pub infer: InferDefaults,
}

impl KitConfig {
    /// Window of `timescale` such as `"1m"`; unlisted timescales get the global window.
    pub fn window(&self, timescale: &str) -> WindowSpec {
        self.windows.get(timescale)
    }

    /// Converts the unpickled `config` dictionary of a checkpoint.
    pub(crate) fn from_pickle(config: &Dict<'_>) -> Result<Self> {
        let model = ModelConfig::from_pickle(&config.dict("model")?)?;
        let data = config.dict_or_empty("data")?;
        let infer = config.dict_or_empty("infer")?;
        let windows = Windows::parse(
            &data.string_or("scale_windows", "")?,
            data.usize_or("Lc", DEFAULT_CONTEXT_LEN)?,
            data.usize_or("Lh", DEFAULT_HORIZON_LEN)?,
            data.usize_or("trunc_min", DEFAULT_TRUNC_MIN)?,
        )?;
        let parsed = Self {
            model,
            windows,
            soft_clip_c: data.f64_or("soft_clip_c", DEFAULT_SOFT_CLIP_C)?,
            vol_ema_half_life_days: data
                .f64_or("vol_ema_half_life_days", DEFAULT_VOL_EMA_HALF_LIFE_DAYS)?,
            infer: InferDefaults::from_pickle(&infer)?,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    /// Parses and validates a JSON document produced by [`KitConfig::to_json`].
    pub fn from_json(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Serializes to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Checks value ranges and cross-field constraints.
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.windows.validate()?;
        for (name, value) in [
            ("soft_clip_c", self.soft_clip_c),
            ("vol_ema_half_life_days", self.vol_ema_half_life_days),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "data.{name} must be finite and positive, got {value}"
                )));
            }
        }
        self.infer.validate()
    }
}
