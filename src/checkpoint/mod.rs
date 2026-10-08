//! Direct loading of official KiT PyTorch checkpoints (`.pt`) in pure Rust.
//!
//! A `.pt` file is a zip archive with a pickled dictionary (`step`, `config`, `ema`, ...) and
//! one raw storage per tensor. [`Checkpoint::load_pt`] parses the pickle without executing it,
//! reads the configuration, picks the `ema.shadow` weights (falling back to `model`), validates
//! them strictly against [`ModelConfig::tensor_specs`] and streams every tensor onto the
//! target device. No Python and no conversion step are involved.
//!
//! [`Checkpoint::save_dir`] and [`Checkpoint::load_dir`] offer an optional Rust-friendly
//! exchange layout: `model.safetensors` plus `config.json`.

mod config;
mod kit_config;
mod loader;
mod pickle;
mod pt;
mod weights;

#[cfg(test)]
mod tests;

pub use config::{ModelConfig, TensorSpec};
pub use kit_config::{InferDefaults, KitConfig, TIMESCALES, WindowSpec, Windows};
pub use loader::{CONFIG_FILE, Checkpoint, CheckpointInfo, INFO_FILE, SourceKind, WEIGHTS_FILE};
pub use weights::ModelWeights;

pub(crate) type Result<T> = std::result::Result<T, crate::Error>;
