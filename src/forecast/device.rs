use crate::{Error, Result};

/// Compute device used to run the model.
///
/// [`Device::Cpu`] is always available. [`Device::Cuda`] and [`Device::Metal`] require the
/// crate features `cuda` / `metal` and matching hardware; requesting them otherwise returns
/// [`Error::UnsupportedHardware`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host CPU.
    #[default]
    Cpu,
    /// NVIDIA GPU with the given ordinal (feature `cuda`).
    Cuda(usize),
    /// Apple GPU with the given ordinal (feature `metal`).
    Metal(usize),
}

impl Device {
    /// The fastest device this build can use: CUDA, then Metal, then CPU.
    pub fn best_available() -> Self {
        if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            Self::Cuda(0)
        } else if cfg!(feature = "metal") && candle_core::utils::metal_is_available() {
            Self::Metal(0)
        } else {
            Self::Cpu
        }
    }

    pub(crate) fn to_backend(self) -> Result<candle_core::Device> {
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            Self::Cuda(ordinal) => {
                if !cfg!(feature = "cuda") {
                    return Err(Error::UnsupportedHardware(
                        "this build of kit-forecast was compiled without the `cuda` feature"
                            .to_string(),
                    ));
                }
                candle_core::Device::new_cuda(ordinal).map_err(|e| {
                    Error::UnsupportedHardware(format!("CUDA device {ordinal} unavailable: {e}"))
                })
            }
            Self::Metal(ordinal) => {
                if !cfg!(feature = "metal") {
                    return Err(Error::UnsupportedHardware(
                        "this build of kit-forecast was compiled without the `metal` feature"
                            .to_string(),
                    ));
                }
                candle_core::Device::new_metal(ordinal).map_err(|e| {
                    Error::UnsupportedHardware(format!("Metal device {ordinal} unavailable: {e}"))
                })
            }
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(i) => write!(f, "cuda:{i}"),
            Self::Metal(i) => write!(f, "metal:{i}"),
        }
    }
}
