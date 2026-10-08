//! Native Rust port of **KiT**, a flow-matching diffusion transformer for candlestick
//! (K-line) forecasting.
//!
//! KiT ([paper](https://arxiv.org/abs/2609.34507), [code](https://github.com/Luciferbobo/KiT))
//! encodes OHLCV bars as five log-ratio features, conditions a DiT backbone on calendar,
//! instrument and timescale information, and generates the next `Lh` bars by integrating a
//! rectified flow from noise. This crate runs the official pretrained checkpoint without
//! Python or PyTorch, on [Candle](https://github.com/huggingface/candle).
//!
//! This is an independent port, not the official implementation. See the repository `NOTICE`
//! and `README.md` for attribution, licensing and the numerical-parity status.
//!
//! # Forecasting
//!
//! ```no_run
//! use kit_forecast::data::Bar;
//! use kit_forecast::forecast::{Device, ForecastOptions, ForecastRequest, Instrument, Kit};
//! use kit_forecast::market::{Market, Timescale};
//!
//! # fn main() -> kit_forecast::Result<()> {
//! // Official checkpoint from https://huggingface.co/Lucifer744/KiT_model
//! let kit = Kit::from_pt("step_market_284M.pt", Device::Cpu)?;
//!
//! // Daily bars need 250 bars of history; the model then predicts the next 20.
//! let bars: Vec<Bar> = load_daily_bars();
//! let request = ForecastRequest::new(bars, Market::Ashare, Timescale::Day1)
//!     .with_instrument(Instrument::new(0, 1817));
//!
//! let forecast = kit.forecast(&request, &ForecastOptions::default())?;
//! for (t, ts) in forecast.timestamps.iter().enumerate() {
//!     let median = forecast.summary()?.channel_bands.iter()
//!         .find(|(name, _)| name == "close")
//!         .map(|(_, band)| band.q50[t]);
//!     println!("{ts}: median close {median:?}");
//! }
//! # Ok(()) }
//! # fn load_daily_bars() -> Vec<Bar> { Vec::new() }
//! ```
//!
//! # Windows
//!
//! The model only supports the windows it was trained with. The history must contain at least
//! `Lc` bars and the horizon is fixed to `Lh`:
//!
//! | timescale | 1m | 5m | 15m | 30m | 1h | 2h | 1d |
//! |---|---|---|---|---|---|---|---|
//! | history `Lc` | 1200 | 960 | 800 | 640 | 400 | 360 | 250 |
//! | horizon `Lh` | 120 | 96 | 64 | 40 | 20 | 20 | 20 |
//!
//! [`forecast::Kit::window`] reports the values stored in the loaded checkpoint.
//!
//! # Layers
//!
//! * [`forecast`]: the high-level API ([`forecast::Kit`], requests, options, results).
//! * [`data`]: bars, candle features, normalisation, model-input windows, post-processing.
//! * [`market`]: markets, timescales, trading sessions, calendars and calendar features.
//! * [`inference`]: the rectified-flow sampler for callers who manage tensors themselves.
//! * [`model`]: the `KlineDiT` network.
//! * [`checkpoint`]: loading of official `.pt` files and the safetensors exchange layout.
//!
//! # Features
//!
//! * `cuda`: NVIDIA GPU support through Candle (requires the CUDA toolkit at build time).
//! * `metal`: Apple GPU support through Candle (macOS only; not exercised by the authors of
//!   this port).
//! * `hf-hub`: [`forecast::Kit::from_hub`] downloads checkpoints from the Hugging Face Hub.

pub mod checkpoint;
pub mod data;
mod error;
pub mod forecast;
pub mod inference;
pub mod market;
pub mod model;

pub use error::{Error, Result};
