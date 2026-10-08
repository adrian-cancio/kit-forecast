use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{Map, Number, Value};

use super::features::FEATURE_NAMES;
use super::normalize::stats_key;
use crate::market::{Market, Timescale};
use crate::{Error, Result};

/// Per-bucket normalization scales keyed by `"<market>|<timescale>|<feature>"`
/// (the `stats.json` format).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stats {
    scales: BTreeMap<String, f64>,
}

impl Stats {
    /// Empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stats from an existing key/scale map.
    pub fn from_map(scales: BTreeMap<String, f64>) -> Self {
        Self { scales }
    }

    /// Parse the text of a `stats.json` file: a JSON object of numeric scales.
    /// Numeric strings are accepted like Python's `float(...)`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text).map_err(Error::Json)?;
        let Value::Object(map) = value else {
            return Err(Error::InvalidData(
                "stats JSON must be an object of scales".to_string(),
            ));
        };
        let mut scales = BTreeMap::new();
        for (key, v) in map {
            let scale = match &v {
                Value::Number(n) => n.as_f64(),
                Value::String(s) => s.trim().parse::<f64>().ok(),
                Value::Bool(b) => Some(f64::from(u8::from(*b))),
                _ => None,
            };
            let scale = scale.ok_or_else(|| {
                Error::InvalidData(format!("stats value for {key:?} is not a number: {v}"))
            })?;
            scales.insert(key, scale);
        }
        Ok(Self { scales })
    }

    /// Read a `stats.json` file.
    pub fn from_json_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref()).map_err(Error::Io)?;
        Self::from_json_str(&text)
    }

    /// Serialize with sorted keys and two-space indentation, like the upstream `save_stats`.
    pub fn to_json_string(&self) -> Result<String> {
        let mut map = Map::new();
        for (key, &scale) in &self.scales {
            let number = Number::from_f64(scale).ok_or_else(|| {
                Error::InvalidData(format!("stats value for {key:?} is not finite"))
            })?;
            map.insert(key.clone(), Value::Number(number));
        }
        serde_json::to_string_pretty(&Value::Object(map)).map_err(Error::Json)
    }

    /// Write a `stats.json` file, creating parent directories.
    pub fn save_json_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(Error::Io)?;
        }
        fs::write(path, self.to_json_string()?).map_err(Error::Io)
    }

    /// Insert or replace a scale.
    pub fn insert(&mut self, key: impl Into<String>, scale: f64) {
        self.scales.insert(key.into(), scale);
    }

    /// Scale for a raw key.
    pub fn get(&self, key: &str) -> Option<f64> {
        self.scales.get(key).copied()
    }

    /// Scale for a (market, timescale, feature name) bucket.
    pub fn scale(&self, market: Market, timescale: Timescale, feature: &str) -> Result<f64> {
        let key = stats_key(market, timescale, feature);
        self.get(&key)
            .ok_or_else(|| Error::InvalidData(format!("missing stats key {key:?}")))
    }

    /// Scales of the five features of a bucket in [`FEATURE_NAMES`] order.
    pub fn feature_scales(&self, market: Market, timescale: Timescale) -> Result<[f64; 5]> {
        let mut out = [0.0; 5];
        for (slot, name) in out.iter_mut().zip(FEATURE_NAMES) {
            *slot = self.scale(market, timescale, name)?;
        }
        Ok(out)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.scales.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    /// Iterate over `(key, scale)` in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.scales.iter().map(|(k, &v)| (k.as_str(), v))
    }
}

const OFFICIAL_ASHARE_STATS: &str = include_str!("ashare_stats.json");

impl Stats {
    /// The normalization scales published with the official KiT demo data (`stats.json`).
    ///
    /// They cover the `ashare` market at all seven timescales and were fitted by the KiT
    /// authors on their training data. No scales were published for other markets, so
    /// forecasts for them need [`Stats`] you fitted yourself.
    pub fn official_ashare() -> Self {
        Self::from_json_str(OFFICIAL_ASHARE_STATS)
            .unwrap_or_else(|_| unreachable!("embedded stats.json is valid"))
    }
}
