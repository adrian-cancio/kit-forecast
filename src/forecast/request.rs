use crate::data::Bar;
use crate::market::{Market, Timescale};
use crate::{Error, Result};

use super::instruments::Instrument;

const PRICE_TOLERANCE: f64 = 1e-9;

/// Historical market data and identity of the series to forecast.
///
/// `bars` must be in strictly ascending time order and end at the anchor: the forecast starts
/// with the bar that follows the last one supplied.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastRequest {
    /// Historical bars, oldest first. Timestamps are bar close times in exchange-local time.
    pub bars: Vec<Bar>,
    /// Market the instrument trades on; selects the session template and normalization scales.
    pub market: Market,
    /// Bar size.
    pub timescale: Timescale,
    /// Embedding ids of the instrument; [`Instrument::UNKNOWN`] if it is not in the vocabulary.
    pub instrument: Instrument,
}

impl ForecastRequest {
    /// Request for an instrument without a learned identity.
    pub fn new(bars: Vec<Bar>, market: Market, timescale: Timescale) -> Self {
        Self {
            bars,
            market,
            timescale,
            instrument: Instrument::UNKNOWN,
        }
    }

    /// Set the instrument identity.
    #[must_use]
    pub fn with_instrument(mut self, instrument: Instrument) -> Self {
        self.instrument = instrument;
        self
    }

    /// Check the bars for problems that would silently corrupt the model input.
    pub fn validate(&self) -> Result<()> {
        validate_bars(&self.bars)
    }
}

pub(crate) fn validate_bars(bars: &[Bar]) -> Result<()> {
    for (i, bar) in bars.iter().enumerate() {
        let prices = [bar.open, bar.high, bar.low, bar.close];
        if prices.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err(Error::InvalidData(format!(
                "bar {i} ({}): prices must be finite and positive, got O={} H={} L={} C={}",
                bar.timestamp, bar.open, bar.high, bar.low, bar.close
            )));
        }
        if !bar.volume.is_finite() || bar.volume < 0.0 {
            return Err(Error::InvalidData(format!(
                "bar {i} ({}): volume must be finite and non-negative, got {}",
                bar.timestamp, bar.volume
            )));
        }
        let body_hi = bar.open.max(bar.close);
        let body_lo = bar.open.min(bar.close);
        if bar.high < body_hi * (1.0 - PRICE_TOLERANCE) {
            return Err(Error::InvalidData(format!(
                "bar {i} ({}): high {} is below max(open, close) {body_hi}",
                bar.timestamp, bar.high
            )));
        }
        if bar.low > body_lo * (1.0 + PRICE_TOLERANCE) {
            return Err(Error::InvalidData(format!(
                "bar {i} ({}): low {} is above min(open, close) {body_lo}",
                bar.timestamp, bar.low
            )));
        }
        if i > 0 && bars[i - 1].timestamp >= bar.timestamp {
            return Err(Error::InvalidData(format!(
                "bar {i}: timestamp {} does not follow {}; bars must be strictly ascending",
                bar.timestamp,
                bars[i - 1].timestamp
            )));
        }
    }
    Ok(())
}
