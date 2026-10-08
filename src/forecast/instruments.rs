use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::{Error, Result};

/// Embedding ids that identify one instrument to the model.
///
/// Id `0` is the learned "unknown" entry in both tables; the model was trained with
/// instrument dropout, so [`Instrument::UNKNOWN`] is a valid input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize)]
pub struct Instrument {
    /// Sector embedding id (`0` = unknown).
    #[serde(default)]
    pub sector: u32,
    /// Instrument embedding id, called `vocab_index` in the official `instruments.json`
    /// (`0` = unknown).
    #[serde(default, alias = "vocab_index")]
    pub id: u32,
}

impl Instrument {
    /// Instrument without a learned identity.
    pub const UNKNOWN: Self = Self { sector: 0, id: 0 };

    /// Build from explicit embedding ids.
    pub const fn new(sector: u32, id: u32) -> Self {
        Self { sector, id }
    }
}

#[derive(Deserialize)]
struct RawEntry {
    #[serde(default)]
    sector: u32,
    vocab_index: u32,
}

/// Lookup table from instrument codes (for example `"600681.SH"`) to [`Instrument`] ids,
/// read from the official `instruments.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstrumentTable {
    entries: BTreeMap<String, Instrument>,
}

impl InstrumentTable {
    /// Parse the text of an `instruments.json` file.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let raw: BTreeMap<String, RawEntry> = serde_json::from_str(text)?;
        let entries = raw
            .into_iter()
            .map(|(code, e)| {
                (
                    code,
                    Instrument {
                        sector: e.sector,
                        id: e.vocab_index,
                    },
                )
            })
            .collect();
        Ok(Self { entries })
    }

    /// Read an `instruments.json` file.
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.display()),
            ))
        })?;
        Self::from_json_str(&text)
    }

    /// Ids of `code`, if present.
    pub fn get(&self, code: &str) -> Option<Instrument> {
        self.entries.get(code).copied()
    }

    /// Number of instruments.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(code, ids)` in code order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Instrument)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), *v))
    }
}
