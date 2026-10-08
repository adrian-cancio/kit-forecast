//! RMSNorm (port of `kit_v1/model/norms.py`).

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// Epsilon used by every RMSNorm in the model.
pub const RMS_EPS: f64 = 1e-6;

/// RMSNorm computed in fp32 and cast back to the input dtype.
///
/// The non-affine variant is used where AdaLN supplies scale/shift; the affine
/// variant (single `weight` vector) is used for QK-Norm.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    /// Norm without learnable parameters.
    pub fn plain() -> Self {
        Self {
            weight: None,
            eps: RMS_EPS,
        }
    }

    /// Norm with a learnable `weight` of shape `[dim]`, read from `vb`.
    pub fn affine(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self {
            weight: Some(weight),
            eps: RMS_EPS,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let mean_sq = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_rms = mean_sq.affine(1.0, self.eps)?.sqrt()?.recip()?;
        let mut out = xf.broadcast_mul(&inv_rms)?;
        if let Some(w) = &self.weight {
            out = out.broadcast_mul(&w.to_dtype(DType::F32)?)?;
        }
        out.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn plain_norm_matches_manual_formula() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 4), &dev)?;
        let y = RmsNorm::plain()
            .forward(&x)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let ms = (1. + 4. + 9. + 16.) / 4.0f32;
        let inv = 1.0 / (ms + 1e-6f32).sqrt();
        for (got, v) in y.iter().zip([1f32, 2., 3., 4.]) {
            assert!((got - v * inv).abs() < 1e-6);
        }
        Ok(())
    }
}
