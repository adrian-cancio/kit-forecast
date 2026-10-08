#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use common::{assert_close, fixture_bars};
use kit_forecast::forecast::{Device, ForecastOptions, ForecastRequest, Instrument, Kit};
use kit_forecast::market::{Market, Timescale};

fn kit(device: Device) -> Kit {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_ckpt.pt");
    Kit::from_pt(path, device).expect("tiny checkpoint loads")
}

#[test]
fn cuda_matches_cpu_on_the_tiny_model() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("no CUDA device; skipping");
        return;
    }
    let request = ForecastRequest::new(fixture_bars(100), Market::Ashare, Timescale::Min30)
        .with_instrument(Instrument::new(3, 5));
    let options = ForecastOptions {
        paths: 3,
        steps: 4,
        seed: 11,
        ..ForecastOptions::default()
    };
    let cpu = kit(Device::Cpu).forecast(&request, &options).unwrap();
    let gpu = kit(Device::Cuda(0)).forecast(&request, &options).unwrap();
    assert_eq!(cpu.timestamps, gpu.timestamps);
    for (a, b) in cpu.features.iter().zip(&gpu.features) {
        let a: Vec<f32> = a.iter().flatten().map(|&x| x as f32).collect();
        let b: Vec<f32> = b.iter().flatten().map(|&x| x as f32).collect();
        assert_close(&b, &a, 2e-3, 2e-3, "cuda vs cpu features");
    }
}
