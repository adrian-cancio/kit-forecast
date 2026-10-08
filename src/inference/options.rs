use crate::checkpoint::InferDefaults;
use crate::{Error, Result};

/// ODE solver used to integrate the flow from noise (`t = 1`) to data (`t = 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Solver {
    /// First-order Euler: `z <- z - dt * v(z, t)`.
    #[default]
    Euler,
    /// Second-order Heun (explicit trapezoid); the final step is always Euler.
    Heun,
}

impl Solver {
    /// Parse the upstream solver name (`"euler"` or `"heun"`).
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "euler" => Ok(Self::Euler),
            "heun" => Ok(Self::Heun),
            other => Err(Error::InvalidConfig(format!(
                "solver must be \"euler\" or \"heun\", got {other:?}"
            ))),
        }
    }

    /// Upstream name of the solver.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Heun => "heun",
        }
    }
}

/// Parameters of one call to [`flow_sample`](super::flow_sample).
///
/// The defaults match the official evaluation scripts: 32 uniform Euler steps, 16 paths,
/// no guidance, seed 0.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleOptions {
    /// Number of integration steps on a uniform grid from `t = 1` to `t = 0`.
    pub steps: usize,
    /// Integration scheme.
    pub solver: Solver,
    /// Number of sampled paths per window.
    pub n_paths: usize,
    /// Identity guidance weight; `1.0` disables it.
    pub cfg_w: f64,
    /// History guidance weight; `1.0` disables it.
    pub hist_cfg_w: f64,
    /// Seed of the initial noise (reproduces `torch.Generator().manual_seed(seed)`).
    pub seed: u64,
    /// Maximum number of sequences per forward pass.
    pub chunk_size: usize,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            steps: 32,
            solver: Solver::Euler,
            n_paths: 16,
            cfg_w: 1.0,
            hist_cfg_w: 1.0,
            seed: 0,
            chunk_size: 16,
        }
    }
}

impl SampleOptions {
    /// Options taken from the `infer` section stored in a checkpoint.
    pub fn from_infer_defaults(defaults: &InferDefaults) -> Result<Self> {
        Ok(Self {
            steps: defaults.steps,
            solver: Solver::parse(&defaults.solver)?,
            n_paths: defaults.n_paths,
            cfg_w: defaults.cfg_w,
            hist_cfg_w: defaults.hist_cfg_w,
            ..Self::default()
        })
    }

    /// Checks that every parameter is usable.
    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 {
            return Err(Error::InvalidConfig("steps must be at least 1".to_string()));
        }
        if self.n_paths == 0 {
            return Err(Error::InvalidConfig(
                "n_paths must be at least 1".to_string(),
            ));
        }
        if self.chunk_size == 0 {
            return Err(Error::InvalidConfig(
                "chunk_size must be at least 1".to_string(),
            ));
        }
        for (name, w) in [("cfg_w", self.cfg_w), ("hist_cfg_w", self.hist_cfg_w)] {
            if !w.is_finite() {
                return Err(Error::InvalidConfig(format!(
                    "{name} must be finite, got {w}"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn uses_identity_guidance(&self) -> bool {
        self.cfg_w != 1.0
    }

    pub(crate) fn uses_history_guidance(&self) -> bool {
        self.hist_cfg_w != 1.0
    }
}
