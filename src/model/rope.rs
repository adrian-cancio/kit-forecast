//! 1D rotary position embedding, LLaMA-style `rotate_half`
//! (port of `kit_v1/model/rope.py`).
//!
//! The reference caches an fp32 cos/sin table and gathers rows by position.
//! Because `cos(pos * inv_freq)` is evaluated elementwise in fp32 either way,
//! this port evaluates the tables directly from the positions (no cache, no
//! maximum position limit).

use candle_core::{D, DType, Result, Tensor, bail};

/// cos/sin tables for a set of positions, shaped `[1, 1, S, head_dim]` (fp32).
#[derive(Debug, Clone)]
pub struct RopeTables {
    cos: Tensor,
    sin: Tensor,
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    head_dim: usize,
    inv_freq: Vec<f32>,
}

/// `(x1, x2) -> (-x2, x1)` over the two contiguous halves of the last dim.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, last - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, theta: f32) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            bail!("head_dim must be even and positive, got {head_dim}");
        }
        let inv_freq = (0..head_dim / 2)
            .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        Ok(Self { head_dim, inv_freq })
    }

    /// Build tables for `positions` (any integer or float dtype, shape `[S]`).
    pub fn tables(&self, positions: &Tensor) -> Result<RopeTables> {
        let dev = positions.device();
        let s = positions.dim(0)?;
        let half = self.head_dim / 2;
        let inv_freq = Tensor::from_slice(&self.inv_freq, (1, half), dev)?;
        let pos = positions.to_dtype(DType::F32)?.reshape((s, 1))?;
        let freqs = pos.broadcast_mul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?.reshape((1, 1, s, self.head_dim))?;
        let sin = emb.sin()?.reshape((1, 1, s, self.head_dim))?;
        Ok(RopeTables { cos, sin })
    }
}

impl RopeTables {
    /// Rotate `q`/`k` (`[B, H, S, Dh]`); math is done in fp32 and cast back.
    pub fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let rotate = |x: &Tensor| -> Result<Tensor> {
            let xf = x.to_dtype(DType::F32)?;
            let out =
                (xf.broadcast_mul(&self.cos)? + rotate_half(&xf)?.broadcast_mul(&self.sin)?)?;
            out.to_dtype(x.dtype())
        };
        Ok((rotate(q)?, rotate(k)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rotate_half_splits_into_halves() -> Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &Device::Cpu)?;
        let y = rotate_half(&x)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(y, vec![-3., -4., 1., 2.]);
        Ok(())
    }

    #[test]
    fn position_zero_is_identity() -> Result<()> {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 10000.0)?;
        let pos = Tensor::from_vec(vec![0u32], 1, &dev)?;
        let t = rope.tables(&pos)?;
        let q = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 1, 4), &dev)?;
        let (qr, kr) = t.apply(&q, &q)?;
        assert_eq!(qr.flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
        assert_eq!(kr.flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
        Ok(())
    }

    #[test]
    fn odd_head_dim_is_rejected() {
        assert!(RotaryEmbedding::new(3, 10000.0).is_err());
    }
}
