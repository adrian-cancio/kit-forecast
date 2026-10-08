use std::collections::HashMap;

use candle_core::{Device, Tensor, safetensors};

use super::TorchRng;

const MAX_ULP: i64 = 4;

fn fixture(name: &str) -> HashMap<String, Tensor> {
    let path = format!("{}/tests/fixtures/tiny/{name}", env!("CARGO_MANIFEST_DIR"));
    safetensors::load(path, &Device::Cpu).expect("fixture loads")
}

fn values(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn ulp_distance(a: f32, b: f32) -> i64 {
    (i64::from(a.to_bits() as i32) - i64::from(b.to_bits() as i32)).abs()
}

fn assert_close_ulps(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let same_sign = a.is_sign_negative() == b.is_sign_negative();
        assert!(
            a == b || (same_sign && ulp_distance(*a, *b) <= MAX_ULP),
            "{what}[{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn small_draws_are_bit_exact() {
    let g = fixture("rng.safetensors");
    for (name, tensor) in &g {
        let Some(rest) = name.strip_prefix("randn.s") else {
            continue;
        };
        let want = values(tensor);
        if want.len() >= 16 {
            continue;
        }
        let (seed, _) = rest.split_once('.').expect("name has a shape suffix");
        let got = TorchRng::new(seed.parse().unwrap()).randn(want.len());
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn vectorised_draws_match_torch_within_a_few_ulps() {
    let g = fixture("rng.safetensors");
    let mut checked = 0;
    for (name, tensor) in &g {
        let Some(rest) = name.strip_prefix("randn.s") else {
            continue;
        };
        let want = values(tensor);
        let (seed, _) = rest.split_once('.').expect("name has a shape suffix");
        let got = TorchRng::new(seed.parse().unwrap()).randn(want.len());
        assert_close_ulps(&got, &want, name);
        checked += 1;
    }
    assert!(checked >= 18);
}

#[test]
fn successive_calls_share_one_stream() {
    let g = fixture("rng.safetensors");
    let mut rng = TorchRng::new(7);
    for (i, n) in [3usize, 17, 16, 40].into_iter().enumerate() {
        let key = format!("seq.s7.{i}.n{n}");
        assert_close_ulps(&rng.randn(n), &values(&g[&key]), &key);
    }
}

#[test]
fn sampler_noise_fixture_is_reproduced() {
    let g = fixture("rng.safetensors");
    let want = values(&g["noise_seed7_3x16x5"]);
    assert_close_ulps(&TorchRng::new(7).randn(want.len()), &want, "noise_seed7");
}
