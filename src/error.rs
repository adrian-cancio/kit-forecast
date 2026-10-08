//! Error type shared by the whole crate.

use std::io;

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong when preparing data, loading a model or forecasting.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Market data is malformed, too short, or inconsistent with the requested window.
    #[error("invalid market data: {0}")]
    InvalidData(String),
    /// Fewer bars were supplied than the model needs to build its context window.
    #[error("insufficient history: {actual} bars supplied, at least {required} required")]
    InsufficientHistory {
        /// Minimum number of bars required.
        required: usize,
        /// Number of bars supplied.
        actual: usize,
    },
    /// A configuration value (model, forecast options, metadata) is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A checkpoint is unsupported, malformed, or does not match the model configuration.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    /// A tensor required by the model is absent from the checkpoint.
    #[error("missing tensor `{0}` in checkpoint")]
    MissingTensor(String),
    /// A tensor has a different shape than the model configuration requires.
    #[error("tensor `{name}` has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Shape found in the checkpoint.
        actual: Vec<usize>,
        /// Shape required by the configuration.
        expected: Vec<usize>,
    },
    /// Downloading a model from a remote hub failed.
    #[error("model download failed: {0}")]
    Download(String),
    /// The requested compute device or feature is unavailable in this build or machine.
    #[error("unsupported hardware: {0}")]
    UnsupportedHardware(String),
    /// The tensor backend reported a failure.
    #[error("backend error: {0}")]
    Backend(#[from] candle_core::Error),
    /// An I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// A JSON document could not be parsed or written.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
