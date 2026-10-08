//! Rectified-flow sampling of future candle paths.
//!
//! - [`WindowBatch`]: model inputs stacked from [`PreparedWindow`](crate::data::PreparedWindow)s
//! - [`SampleOptions`] / [`Solver`]: sampler parameters (defaults reproduce upstream)
//! - [`flow_sample`]: the Euler / Heun integrator with optional guidance
//! - [`TorchRng`]: PyTorch-compatible noise generator

mod batch;
mod options;
mod rng;
mod sampler;

#[cfg(test)]
mod tests;

pub use batch::WindowBatch;
pub use options::{SampleOptions, Solver};
pub use rng::TorchRng;
pub use sampler::{flow_sample, flow_sample_observed};
