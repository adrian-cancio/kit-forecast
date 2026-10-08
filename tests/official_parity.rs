//! Parity against the official checkpoint and the upstream Python reference.
//!
//! Needs `references/ckpt/step_market_284M.pt`, `references/data/` and the dumps written by
//! `tools/parity/make_reference.py` into `tools/cache/reference/`. Run with
//! `cargo test --release --test official_parity -- --ignored --nocapture`.
//! Select cases with `KIT_PARITY_CASES=1d,2h` (default: `1d`).

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use common::{assert_close, flat, max_abs_diff, parse_ts};
use kit_forecast::data::Bar;
use kit_forecast::forecast::{
    Device as KitDevice, ForecastOptions, ForecastRequest, Instrument, Kit,
};
use kit_forecast::inference::{SampleOptions, WindowBatch, flow_sample_observed};
use kit_forecast::market::{Market, Timescale, TradingCalendar};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_bars(path: &PathBuf) -> Vec<Bar> {
    let text = std::fs::read_to_string(path).expect("bars csv");
    text.lines()
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let n = |i: usize| f[i].parse::<f64>().unwrap();
            Bar::new(parse_ts(f[0]), n(1), n(2), n(3), n(4), n(5))
        })
        .collect()
}

fn tensors(path: PathBuf) -> HashMap<String, Tensor> {
    common::load_tensors_from(path)
}

fn timescale(s: &str) -> Timescale {
    s.parse().expect("timescale")
}

fn run_case(kit: &Kit, dir: PathBuf) {
    let inputs: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("inputs.json")).unwrap()).unwrap();
    let scale = timescale(inputs["scale"].as_str().unwrap());
    let lc = inputs["Lc"].as_u64().unwrap() as usize;
    let lh = inputs["Lh"].as_u64().unwrap() as usize;
    let k = inputs["paths"].as_u64().unwrap() as usize;
    let steps = inputs["steps"].as_u64().unwrap() as usize;
    let seed = inputs["seed"].as_u64().unwrap();
    println!(
        "== {} (Lc={lc}, Lh={lh}, K={k}, steps={steps}, seed={seed})",
        inputs["case"]
    );

    let request = ForecastRequest::new(read_bars(&dir.join("bars.csv")), Market::Ashare, scale)
        .with_instrument(Instrument::new(
            inputs["sector_id"].as_u64().unwrap() as u32,
            inputs["inst_id"].as_u64().unwrap() as u32,
        ));
    let options = ForecastOptions {
        paths: k,
        steps,
        seed,
        chunk_size: 64,
        ..ForecastOptions::default()
    };

    let window = kit.prepare(&request, &options).expect("prepare");
    let golden = tensors(dir.join("window.safetensors"));
    let x: Vec<f32> = window.x_ctx.iter().flatten().copied().collect();
    assert_close(&x, &flat(&golden["x_ctx"]), 5e-6, 5e-6, "x_ctx");
    let sess: Vec<f32> = window.cond.sess.iter().flatten().copied().collect();
    assert_close(&sess, &flat(&golden["cond.sess"]), 1e-6, 0.0, "cond.sess");
    let clock: Vec<f32> = window.cond.clock.iter().flatten().copied().collect();
    assert_close(
        &clock,
        &flat(&golden["cond.clock"]),
        1e-6,
        0.0,
        "cond.clock",
    );
    let yday: Vec<f32> = window.cond.yday.iter().flatten().copied().collect();
    assert_close(&yday, &flat(&golden["cond.yday"]), 1e-6, 0.0, "cond.yday");
    for (name, got) in [
        ("cond.dow", &window.cond.dow),
        ("cond.event", &window.cond.event),
        ("cond.month", &window.cond.month),
    ] {
        let want: Vec<i64> = golden[name]
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(got, &want, "{name}");
    }
    assert_eq!(window.prev_close, inputs["prev_close"].as_f64().unwrap());
    let ema = inputs["prev_ema"].as_f64().unwrap();
    assert!(
        (window.prev_ema - ema).abs() <= 1e-9 * ema.abs(),
        "prev_ema"
    );
    let fut: Vec<String> = window
        .future_timestamps
        .iter()
        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    let want_fut: Vec<String> = inputs["future_timestamps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    assert_eq!(fut, want_fut, "future timestamps");

    let batch = WindowBatch::from_windows(&[&window], &Device::Cpu).unwrap();
    let ref_steps = tensors(dir.join("steps.safetensors"));
    let z_in = &ref_steps["z_in"];
    let mut worst_step = 0.0f32;
    let sample = SampleOptions {
        steps,
        n_paths: k,
        seed,
        chunk_size: 64,
        ..SampleOptions::default()
    };
    let started = std::time::Instant::now();
    let paths = flow_sample_observed(kit_model(kit), &batch, &sample, |i, z| {
        if i + 1 < steps {
            let want = flat(&z_in.get(i + 1).unwrap());
            let d = max_abs_diff(&flat(z), &want);
            worst_step = worst_step.max(d);
            if i == 0 || i + 2 == steps || (i + 1) % 8 == 0 {
                println!("  step {:>2}: max |dz| {d:.3e}", i + 1);
            }
        }
        Ok(())
    })
    .expect("sampling");
    println!("  sampling took {:.1}s", started.elapsed().as_secs_f64());

    let want_paths = flat(&tensors(dir.join("paths.safetensors"))["paths"]);
    let got_paths = flat(&paths);
    let final_diff = max_abs_diff(&got_paths, &want_paths);
    println!(
        "  final normalized paths: max |diff| {final_diff:.3e} (worst intermediate {worst_step:.3e})"
    );
    assert_close(&got_paths, &want_paths, 2e-4, 2e-4, "paths");

    let fp64_path = dir.join("velocity_fp64.safetensors");
    if fp64_path.exists() {
        let fp64 = tensors(fp64_path);
        let noise = &tensors(dir.join("noise.safetensors"))["noise"];
        let copies: Vec<&_> = (0..k).map(|_| &window).collect();
        let kb = WindowBatch::from_windows(&copies, &Device::Cpu).unwrap();
        let z = Tensor::cat(&[kb.context(), noise], 1).unwrap();
        let t1 = Tensor::ones(k, DType::F32, &Device::Cpu).unwrap();
        let v_rust = kit_model(kit).forward(&z, &t1, kb.condition(), lh).unwrap();
        let v64: Vec<f32> = fp64["v64"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let rust_err = max_abs_diff(&flat(&v_rust), &v64);
        let torch_err = max_abs_diff(&flat(&fp64["v32"]), &v64);
        let cross = max_abs_diff(&flat(&v_rust), &flat(&fp64["v32"]));
        println!(
            "  velocity at t=1 vs float64 truth: rust {rust_err:.3e}, torch {torch_err:.3e}, rust-vs-torch {cross:.3e}"
        );
    }
    let forecast = kit.forecast(&request, &options).expect("forecast");
    let ohlcv = tensors(dir.join("ohlcv.safetensors"));
    let mut worst_rel = 0.0f64;
    type Pick = fn(&kit_forecast::data::Ohlcv) -> &Vec<f64>;
    let columns: [(&str, Pick); 5] = [
        ("open", |p| &p.open),
        ("high", |p| &p.high),
        ("low", |p| &p.low),
        ("close", |p| &p.close),
        ("volume", |p| &p.volume),
    ];
    for (name, got) in columns {
        let want = ohlcv[name]
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let have: Vec<f64> = forecast
            .paths
            .iter()
            .flat_map(|p| got(p).iter().copied())
            .collect();
        for (a, b) in have.iter().zip(&want) {
            worst_rel = worst_rel.max((a - b).abs() / b.abs().max(1.0));
        }
    }
    println!("  final OHLCV: max relative difference {worst_rel:.3e}");
    assert!(
        worst_rel < 5e-5,
        "OHLCV differs from upstream by {worst_rel}"
    );
}

fn kit_model(kit: &Kit) -> &kit_forecast::model::KlineDiT {
    kit.model()
}

#[test]
#[ignore = "needs the 1.1 GB official checkpoint and tools/cache/reference dumps"]
fn official_checkpoint_matches_upstream() {
    let ckpt = root().join("references/ckpt/step_market_284M.pt");
    let calendar =
        TradingCalendar::from_csv_path(root().join("references/data/trade_dates.csv")).unwrap();
    let kit = Kit::from_pt(&ckpt, KitDevice::Cpu)
        .unwrap()
        .with_calendar(calendar);
    let wanted = std::env::var("KIT_PARITY_CASES").unwrap_or_else(|_| "1d".to_string());
    let wanted: Vec<&str> = wanted.split(',').map(str::trim).collect();
    let base = root().join("tools/cache/reference");
    let mut ran = 0;
    for entry in std::fs::read_dir(&base).expect("reference dumps present") {
        let dir = entry.unwrap().path();
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let scale = name.split('_').nth(1).unwrap_or("");
        if wanted.contains(&scale) || wanted.contains(&"all") {
            run_case(&kit, dir);
            ran += 1;
        }
    }
    assert!(ran > 0, "no reference case matched {wanted:?}");
}

#[test]
#[ignore = "diagnostic: needs tools/parity/make_block_trace.py output for the 5m case"]
fn block_trace_against_float64_truth() {
    let root = root();
    let ckpt = root.join("references/ckpt/step_market_284M.pt");
    let calendar =
        TradingCalendar::from_csv_path(root.join("references/data/trade_dates.csv")).unwrap();
    let kit = Kit::from_pt(&ckpt, KitDevice::Cpu)
        .unwrap()
        .with_calendar(calendar);
    let base = root.join("tools/cache/reference");
    let scale = std::env::var("KIT_PARITY_CASES").unwrap_or_else(|_| "5m".to_string());
    let dir = std::fs::read_dir(&base)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name().unwrap().to_string_lossy().split('_').nth(1) == Some(scale.as_str())
        })
        .expect("case dump");
    let inputs: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("inputs.json")).unwrap()).unwrap();
    let k = inputs["paths"].as_u64().unwrap() as usize;
    let lh = inputs["Lh"].as_u64().unwrap() as usize;
    let request = ForecastRequest::new(
        read_bars(&dir.join("bars.csv")),
        Market::Ashare,
        timescale(inputs["scale"].as_str().unwrap()),
    )
    .with_instrument(Instrument::new(
        inputs["sector_id"].as_u64().unwrap() as u32,
        inputs["inst_id"].as_u64().unwrap() as u32,
    ));
    let window = kit.prepare(&request, &ForecastOptions::default()).unwrap();
    let blocks_path = dir.join("blocks.safetensors");
    if blocks_path.exists() {
        let blocks = tensors(blocks_path);
        let noise = &tensors(dir.join("noise.safetensors"))["noise"];
        let copies: Vec<&_> = (0..k).map(|_| &window).collect();
        let kb = WindowBatch::from_windows(&copies, &Device::Cpu).unwrap();
        let z = Tensor::cat(&[kb.context(), noise], 1).unwrap();
        let t1 = Tensor::ones(k, DType::F32, &Device::Cpu).unwrap();
        println!("  block   rust-vs-fp64   torch-vs-fp64   rust-vs-torch");
        kit_model(&kit)
            .forward_with_taps(&z, &t1, kb.condition(), lh, &mut |i, x| {
                let truth = flat(&blocks[&format!("fp64.{i}")]);
                let torch = flat(&blocks[&format!("fp32.{i}")]);
                let mine = flat(x);
                if i % 4 == 0 || i == 21 {
                    println!(
                        "  {i:>5}   {:>12.3e}   {:>13.3e}   {:>13.3e}",
                        max_abs_diff(&mine, &truth),
                        max_abs_diff(&torch, &truth),
                        max_abs_diff(&mine, &torch)
                    );
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
#[ignore = "diagnostic: needs tools/parity/make_gemm_probe.py output"]
fn single_gemm_against_float64_truth() {
    let root = root();
    let ckpt = kit_forecast::checkpoint::Checkpoint::load_pt(
        root.join("references/ckpt/step_market_284M.pt"),
        &Device::Cpu,
    )
    .unwrap();
    let w = ckpt
        .weights
        .get("blocks.0.attn.qkv.weight")
        .unwrap()
        .clone();
    let dir = std::fs::read_dir(root.join("tools/cache/reference"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.file_name().unwrap().to_string_lossy().split('_').nth(1) == Some("5m"))
        .unwrap();
    let g = tensors(dir.join("gemm.safetensors"));
    let x = &g["x"];
    let y_rust = x.broadcast_matmul(&w.t().unwrap()).unwrap();
    let truth = flat(&g["y64"]);
    println!(
        "  qkv GEMM [K=1024] vs float64: rust {:.3e}, torch {:.3e}, rust-vs-torch {:.3e}, max|y| {:.2}",
        max_abs_diff(&flat(&y_rust), &truth),
        max_abs_diff(&flat(&g["y32"]), &truth),
        max_abs_diff(&flat(&y_rust), &flat(&g["y32"])),
        truth.iter().fold(0.0f32, |m, v| m.max(v.abs())),
    );
}
