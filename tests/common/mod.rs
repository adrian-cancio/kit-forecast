#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use kit_forecast::checkpoint::{ModelConfig, ModelWeights};
use kit_forecast::data::{Bar, Stats};
use kit_forecast::market::Timestamp;
use kit_forecast::model::KlineDiT;
use serde_json::Value;

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

pub fn read_json(name: &str) -> Value {
    let text = std::fs::read_to_string(fixture_dir().join(name)).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

pub fn load_tensors(name: &str) -> HashMap<String, Tensor> {
    load_tensors_from(fixture_dir().join(name))
}

pub fn load_tensors_from(path: PathBuf) -> HashMap<String, Tensor> {
    use safetensors::tensor::{Dtype, SafeTensors};
    let bytes = std::fs::read(&path).expect("tensor file readable");
    let file = SafeTensors::deserialize(&bytes).expect("valid safetensors");
    let mut out = HashMap::new();
    for (name, view) in file.tensors() {
        let dtype = match view.dtype() {
            Dtype::F32 => DType::F32,
            Dtype::F64 => DType::F64,
            Dtype::I64 => DType::I64,
            _ => continue,
        };
        let tensor = Tensor::from_raw_buffer(view.data(), dtype, view.shape(), &Device::Cpu)
            .expect("tensor decodes");
        out.insert(name, tensor);
    }
    out
}

pub fn tiny_model() -> KlineDiT {
    let text = std::fs::read_to_string(fixture_dir().join("config.json")).unwrap();
    let cfg: ModelConfig = serde_json::from_str(&text).unwrap();
    let tensors = load_tensors("model.safetensors");
    let weights = ModelWeights::from_tensors(tensors, &cfg).unwrap();
    KlineDiT::load(&cfg, weights.var_builder()).unwrap()
}

pub fn parse_ts(s: &str) -> Timestamp {
    Timestamp::parse_from_str(s, "%Y-%m-%d %H:%M:%S").expect("timestamp parses")
}

pub fn fixture_bars(n: usize) -> Vec<Bar> {
    let v = read_json("bars.json");
    v.as_array()
        .unwrap()
        .iter()
        .take(n)
        .map(|b| {
            let f = |k: &str| b[k].as_f64().unwrap();
            Bar::new(
                parse_ts(b["datetime"].as_str().unwrap()),
                f("open"),
                f("high"),
                f("low"),
                f("close"),
                f("volume"),
            )
        })
        .collect()
}

pub fn fixture_stats() -> Stats {
    Stats::from_json_path(fixture_dir().join("stats.json")).unwrap()
}

pub fn flat(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[track_caller]
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        let allowed = atol + rtol * w.abs();
        worst = worst.max(err);
        assert!(
            err <= allowed && g.is_finite(),
            "{what}[{i}]: got {g}, want {w}, |diff| {err} > {allowed}"
        );
    }
    println!("{what}: max |diff| {worst:e} over {} values", got.len());
}
