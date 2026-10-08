use crate::market::Timestamp;
use crate::{Error, Result};

/// One OHLCV bar. `timestamp` is the bar's close time in exchange-local time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    /// Close time of the bar.
    pub timestamp: Timestamp,
    /// Opening price.
    pub open: f64,
    /// Highest price.
    pub high: f64,
    /// Lowest price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Traded volume.
    pub volume: f64,
}

impl Bar {
    /// Build a bar from its parts.
    pub fn new(
        timestamp: Timestamp,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }
}

/// Column-wise OHLCV values without timestamps, as produced by the inverse transforms.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ohlcv {
    /// Opening prices.
    pub open: Vec<f64>,
    /// Highest prices.
    pub high: Vec<f64>,
    /// Lowest prices.
    pub low: Vec<f64>,
    /// Closing prices.
    pub close: Vec<f64>,
    /// Volumes.
    pub volume: Vec<f64>,
}

impl Ohlcv {
    /// Number of bars.
    pub fn len(&self) -> usize {
        self.open.len()
    }

    /// Whether there are no bars.
    pub fn is_empty(&self) -> bool {
        self.open.is_empty()
    }

    /// Attach close timestamps, one per bar.
    pub fn to_bars(&self, timestamps: &[Timestamp]) -> Result<Vec<Bar>> {
        let n = self.len();
        let lens = [
            self.high.len(),
            self.low.len(),
            self.close.len(),
            self.volume.len(),
            timestamps.len(),
        ];
        if lens.iter().any(|&l| l != n) {
            return Err(Error::InvalidData(format!(
                "OHLCV columns and timestamps must all have length {n}"
            )));
        }
        Ok((0..n)
            .map(|i| {
                Bar::new(
                    timestamps[i],
                    self.open[i],
                    self.high[i],
                    self.low[i],
                    self.close[i],
                    self.volume[i],
                )
            })
            .collect())
    }
}
