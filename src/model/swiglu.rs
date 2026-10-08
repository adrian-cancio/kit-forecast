//! SwiGLU feed-forward (port of `kit_v1/model/swiglu.py`).

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// `w2(silu(w1 x) * w3 x)`, all projections without bias.
#[derive(Debug, Clone)]
pub struct SwiGlu {
    w1: Linear,
    w3: Linear,
    w2: Linear,
}

impl SwiGlu {
    pub fn load(d: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(d, hidden, vb.pp("w1"))?,
            w3: linear_no_bias(d, hidden, vb.pp("w3"))?,
            w2: linear_no_bias(hidden, d, vb.pp("w2"))?,
        })
    }
}

impl Module for SwiGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let value = self.w3.forward(x)?;
        self.w2.forward(&(gate * value)?)
    }
}
