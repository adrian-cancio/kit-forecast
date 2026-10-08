use crate::inference::{SampleOptions, Solver};
use crate::{Error, Result};

/// Options of one forecast.
///
/// [`ForecastOptions::default`] reproduces the settings of the official evaluation scripts
/// (`eval/infer_for_pred.py`): 16 sampled paths, 32 Euler steps, no guidance, seed 0, and a
/// context of exactly `Lc` bars.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastOptions {
    /// Number of sampled future paths `K`.
    pub paths: usize,
    /// Number of integration steps.
    pub steps: usize,
    /// ODE solver.
    pub solver: Solver,
    /// Identity guidance weight; `1.0` disables it. Upstream notes it hurts calibration.
    pub cfg_w: f64,
    /// History guidance weight; `1.0` disables it.
    pub hist_cfg_w: f64,
    /// Seed of the initial noise. Equal seeds give equal forecasts on the same device.
    pub seed: u64,
    /// Maximum number of sequences per forward pass; lower it to save memory.
    pub chunk_size: usize,
    /// Optional daily price-limit band (for example `0.1` for a 10% limit) applied bar by bar
    /// to the generated OHLC values, as in the upstream backtest. `None` leaves paths as
    /// generated.
    pub limit_rate: Option<f64>,
    /// Accept histories shorter than `Lc`. The front of the context is then padded and masked,
    /// which the model saw during training for lengths of at least the window's `trunc_min`.
    /// The official scripts reject such histories, so this is off by default.
    pub allow_padding: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        let sample = SampleOptions::default();
        Self {
            paths: sample.n_paths,
            steps: sample.steps,
            solver: sample.solver,
            cfg_w: sample.cfg_w,
            hist_cfg_w: sample.hist_cfg_w,
            seed: sample.seed,
            chunk_size: sample.chunk_size,
            limit_rate: None,
            allow_padding: false,
        }
    }
}

impl ForecastOptions {
    pub(crate) fn sample_options(&self) -> SampleOptions {
        SampleOptions {
            steps: self.steps,
            solver: self.solver,
            n_paths: self.paths,
            cfg_w: self.cfg_w,
            hist_cfg_w: self.hist_cfg_w,
            seed: self.seed,
            chunk_size: self.chunk_size,
        }
    }

    /// Checks that every option is usable.
    pub fn validate(&self) -> Result<()> {
        self.sample_options().validate()?;
        if let Some(rate) = self.limit_rate
            && !(rate.is_finite() && rate > 0.0 && rate < 1.0)
        {
            return Err(Error::InvalidConfig(format!(
                "limit_rate must be in (0, 1), got {rate}"
            )));
        }
        Ok(())
    }
}
