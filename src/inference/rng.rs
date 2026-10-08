//! Reproduction of the PyTorch CPU random number generator for `torch.randn`.
//!
//! Upstream draws the initial flow noise with `torch.randn(shape, generator=Generator().manual_seed(s))`.
//! [`TorchRng`] follows the same Mersenne Twister stream and the same Box-Muller layout as
//! PyTorch's CPU kernels for contiguous `float32` tensors, so a seed reproduces upstream noise.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// Tensors with at least this many elements use the vectorised Box-Muller fill.
const VECTORIZED_MIN_LEN: usize = 16;

/// Mersenne Twister (MT19937) with PyTorch's normal-sampling front end.
#[derive(Debug, Clone)]
pub struct TorchRng {
    state: [u32; N],
    index: usize,
    cached_normal: Option<f64>,
}

impl TorchRng {
    /// Seed the generator like `torch.Generator().manual_seed(seed)`.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = seed as u32;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state,
            index: N,
            cached_normal: None,
        }
    }

    fn refill(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.refill();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
    }

    fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() & ((1 << 24) - 1)) as f32 * (1.0 / 16_777_216.0)
    }

    fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() & ((1u64 << 53) - 1)) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    fn scalar_normal(&mut self) -> f64 {
        if let Some(cached) = self.cached_normal.take() {
            return cached;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.cached_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Fill `out` like `torch.randn(len)` on a contiguous CPU `float32` tensor.
    pub fn fill_randn(&mut self, out: &mut [f32]) {
        let len = out.len();
        if len < VECTORIZED_MIN_LEN {
            for x in out.iter_mut() {
                *x = self.scalar_normal() as f32;
            }
            return;
        }
        for x in out.iter_mut() {
            *x = self.uniform_f32();
        }
        let mut i = 0;
        while i + 16 <= len {
            box_muller_16(&mut out[i..i + 16]);
            i += 16;
        }
        if !len.is_multiple_of(16) {
            let tail = &mut out[len - 16..];
            for x in tail.iter_mut() {
                *x = self.uniform_f32();
            }
            box_muller_16(tail);
        }
    }

    /// Draw `len` standard normal values like `torch.randn(len)`.
    pub fn randn(&mut self, len: usize) -> Vec<f32> {
        let mut out = vec![0f32; len];
        self.fill_randn(&mut out);
        out
    }
}

fn box_muller_16(data: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * f64::from(u1).ln()).sqrt() as f32;
        let theta = 2.0 * std::f32::consts::PI * u2;
        let theta64 = f64::from(theta);
        data[j] = radius * theta64.cos() as f32;
        data[j + 8] = radius * theta64.sin() as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_reference_output() {
        let mut rng = TorchRng::new(5489);
        assert_eq!(rng.next_u32(), 3_499_211_612);
        assert_eq!(rng.next_u32(), 581_869_302);
        let mut rng = TorchRng::new(5489);
        for _ in 0..9999 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 4_123_659_995);
    }

    #[test]
    fn same_seed_same_noise_and_different_seed_differs() {
        let a = TorchRng::new(7).randn(100);
        let b = TorchRng::new(7).randn(100);
        let c = TorchRng::new(8).randn(100);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn noise_is_standard_normal() {
        let x = TorchRng::new(1).randn(100_000);
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = x
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }
}
