use std::path::Path;
use std::time::Instant;

use candle_core::Tensor;

use crate::checkpoint::{Checkpoint, CheckpointInfo, KitConfig, ModelWeights, WindowSpec};
use crate::data::{
    PreparedWindow, Stats, apply_limit_projection, denormalize_paths, paths_to_ohlcv,
};
use crate::inference::{WindowBatch, flow_sample};
use crate::market::{Timescale, TradingCalendar};
use crate::model::KlineDiT;
use crate::{Error, Result};

use super::device::Device;
use super::options::ForecastOptions;
use super::request::{ForecastRequest, validate_bars};
use super::result::Forecast;

/// A loaded KiT model together with its configuration, ready to forecast.
///
/// Loading is the expensive step (the official checkpoint is 1.1 GB); keep one `Kit` alive and
/// call [`Kit::forecast`] repeatedly. `Kit` is immutable after loading and can be shared by
/// reference between threads.
#[derive(Debug)]
pub struct Kit {
    model: KlineDiT,
    weights: ModelWeights,
    config: KitConfig,
    info: CheckpointInfo,
    stats: Stats,
    calendar: TradingCalendar,
    device: Device,
    backend: candle_core::Device,
}

impl Kit {
    /// Load an official PyTorch checkpoint (`step_market_284M.pt`) directly, without Python.
    pub fn from_pt(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let backend = device.to_backend()?;
        let checkpoint = Checkpoint::load_pt(path, &backend)?;
        Self::from_checkpoint(checkpoint, device, backend)
    }

    /// Load a directory written by [`Kit::save_dir`] (`model.safetensors` and `config.json`).
    pub fn from_dir(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let backend = device.to_backend()?;
        let checkpoint = Checkpoint::load_dir(path, &backend)?;
        Self::from_checkpoint(checkpoint, device, backend)
    }

    /// Download `step_market_284M.pt` (or another file) from a Hugging Face model repository and
    /// load it. Files are cached by the Hugging Face cache; later calls work offline.
    ///
    /// The official repository is `Lucifer744/KiT_model`. Requires the `hf-hub` feature.
    #[cfg(feature = "hf-hub")]
    pub fn from_hub(repo_id: &str, file: &str, device: Device) -> Result<Self> {
        let path = super::hub::download(repo_id, file)?;
        Self::from_pt(path, device)
    }

    fn from_checkpoint(
        checkpoint: Checkpoint,
        device: Device,
        backend: candle_core::Device,
    ) -> Result<Self> {
        let Checkpoint {
            config,
            weights,
            info,
        } = checkpoint;
        let model = KlineDiT::load(&config.model, weights.var_builder())?;
        Ok(Self {
            model,
            weights,
            config,
            info,
            stats: Stats::official_ashare(),
            calendar: TradingCalendar::weekends_only(),
            device,
            backend,
        })
    }

    /// Replace the normalization scales (default: [`Stats::official_ashare`]).
    #[must_use]
    pub fn with_stats(mut self, stats: Stats) -> Self {
        self.stats = stats;
        self
    }

    /// Replace the trading-day calendar used to project forecast timestamps and detect gaps
    /// (default: weekends only, so exchange holidays are not skipped).
    ///
    /// Holidays change the calendar features the model sees, so supply the official
    /// `trade_dates.csv` for A-shares whenever the forecast horizon may cross a holiday.
    #[must_use]
    pub fn with_calendar(mut self, calendar: TradingCalendar) -> Self {
        self.calendar = calendar;
        self
    }

    /// The underlying network, for lower-level use with [`flow_sample`](crate::inference::flow_sample).
    pub fn model(&self) -> &KlineDiT {
        &self.model
    }

    /// Configuration stored in the checkpoint.
    pub fn config(&self) -> &KitConfig {
        &self.config
    }

    /// Provenance of the loaded weights (training step, EMA decay, ...).
    pub fn checkpoint_info(&self) -> &CheckpointInfo {
        &self.info
    }

    /// Normalization scales in use.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Device the model runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Context and horizon lengths the checkpoint was trained with for `timescale`.
    ///
    /// The model only supports these windows.
    pub fn window(&self, timescale: Timescale) -> WindowSpec {
        self.config.window(timescale.as_str())
    }

    /// Write the weights and configuration as `model.safetensors` + `config.json`.
    pub fn save_dir(&self, path: impl AsRef<Path>) -> Result<()> {
        Checkpoint {
            config: self.config.clone(),
            weights: self.weights.clone(),
            info: self.info.clone(),
        }
        .save_dir(path)
    }

    /// Turn raw history into model inputs without running the model.
    ///
    /// Useful to inspect the exact tensors, to batch several requests with
    /// [`Kit::forecast_batch`], or to benchmark preprocessing separately.
    pub fn prepare(
        &self,
        request: &ForecastRequest,
        options: &ForecastOptions,
    ) -> Result<PreparedWindow> {
        request.validate()?;
        let window = self.window(request.timescale);
        let history = request.bars.len();
        let required = if options.allow_padding {
            2
        } else {
            window.context_len
        };
        if history < required {
            return Err(Error::InsufficientHistory {
                required,
                actual: history,
            });
        }
        if options.allow_padding && history < window.trunc_min {
            return Err(Error::InsufficientHistory {
                required: window.trunc_min,
                actual: history,
            });
        }
        PreparedWindow::prepare(
            &request.bars,
            request.market,
            request.timescale,
            window.context_len,
            window.horizon_len,
            &self.stats,
            i64::from(request.instrument.sector),
            i64::from(request.instrument.id),
            &self.calendar,
            self.config.soft_clip_c,
            self.config.vol_ema_half_life_days,
        )
    }

    /// Forecast the next `Lh` bars (the window length of the request's timescale).
    ///
    /// ```no_run
    /// use kit_forecast::forecast::{Device, ForecastOptions, ForecastRequest, Kit};
    /// use kit_forecast::market::{Market, Timescale};
    ///
    /// # fn main() -> kit_forecast::Result<()> {
    /// let kit = Kit::from_pt("step_market_284M.pt", Device::Cpu)?;
    /// let bars = Vec::new(); // at least 250 daily bars for Timescale::Day1
    /// let request = ForecastRequest::new(bars, Market::Ashare, Timescale::Day1);
    /// let forecast = kit.forecast(&request, &ForecastOptions::default())?;
    /// println!("{} paths x {} bars", forecast.n_paths(), forecast.horizon());
    /// # Ok(()) }
    /// ```
    pub fn forecast(
        &self,
        request: &ForecastRequest,
        options: &ForecastOptions,
    ) -> Result<Forecast> {
        let mut out = self.forecast_batch(std::slice::from_ref(request), options)?;
        out.pop()
            .ok_or_else(|| Error::InvalidData("forecast produced no result".to_string()))
    }

    /// Forecast several requests in one batched pass.
    ///
    /// All requests must share market and timescale. Results come back in request order. Each
    /// request draws its paths from one shared noise tensor seeded with `options.seed`, so a
    /// request's paths depend on its position in the batch; use [`Kit::forecast`] for results
    /// that do not depend on batch composition.
    pub fn forecast_batch(
        &self,
        requests: &[ForecastRequest],
        options: &ForecastOptions,
    ) -> Result<Vec<Forecast>> {
        options.validate()?;
        let first = requests
            .first()
            .ok_or_else(|| Error::InvalidData("no requests to forecast".to_string()))?;
        if let Some(other) = requests
            .iter()
            .find(|r| r.market != first.market || r.timescale != first.timescale)
        {
            return Err(Error::InvalidData(format!(
                "a batch must share market and timescale: {}/{} vs {}/{}",
                first.market, first.timescale, other.market, other.timescale
            )));
        }
        let windows = requests
            .iter()
            .map(|r| self.prepare(r, options))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&PreparedWindow> = windows.iter().collect();
        let batch = WindowBatch::from_windows(&refs, &self.backend)?;
        let paths = flow_sample(&self.model, &batch, &options.sample_options())?;
        self.assemble(first, &windows, &paths, options)
    }

    fn assemble(
        &self,
        first: &ForecastRequest,
        windows: &[PreparedWindow],
        paths: &Tensor,
        options: &ForecastOptions,
    ) -> Result<Vec<Forecast>> {
        let (b, k, lh, f) = paths.dims4()?;
        let flat = paths
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut out = Vec::with_capacity(b);
        for (wi, window) in windows.iter().enumerate() {
            let per_path: Vec<Vec<[f32; crate::data::N_FEATURES]>> = (0..k)
                .map(|pi| {
                    (0..lh)
                        .map(|t| {
                            let base = ((wi * k + pi) * lh + t) * f;
                            let mut row = [0f32; crate::data::N_FEATURES];
                            row.copy_from_slice(&flat[base..base + f]);
                            row
                        })
                        .collect()
                })
                .collect();
            let features = denormalize_paths(
                &per_path,
                first.market,
                first.timescale,
                &self.stats,
                self.config.soft_clip_c,
            )?;
            let mut ohlcv = paths_to_ohlcv(
                &features,
                window.prev_close,
                window.prev_ema,
                window.half_life_bars,
            )?;
            if options.limit_rate.is_some() {
                for p in &mut ohlcv {
                    *p = apply_limit_projection(p, window.prev_close, options.limit_rate);
                }
            }
            let anchor = window.ctx_timestamps.last().copied().ok_or_else(|| {
                Error::InvalidData("window has no context timestamps".to_string())
            })?;
            out.push(Forecast {
                anchor,
                timestamps: window.future_timestamps.clone(),
                paths: ohlcv,
                features,
                prev_close: window.prev_close,
                prev_ema: window.prev_ema,
                half_life_bars: window.half_life_bars,
            });
        }
        Ok(out)
    }

    /// Validate bars without needing a model, for early input checking.
    pub fn validate_bars(bars: &[crate::data::Bar]) -> Result<()> {
        validate_bars(bars)
    }

    /// Run `forecast` and report wall-clock seconds; convenience for quick profiling.
    pub fn forecast_timed(
        &self,
        request: &ForecastRequest,
        options: &ForecastOptions,
    ) -> Result<(Forecast, f64)> {
        let start = Instant::now();
        let forecast = self.forecast(request, options)?;
        Ok((forecast, start.elapsed().as_secs_f64()))
    }
}
