mod common;

use candle_core::{DType, Device, Tensor};
use common::*;
use kit_forecast::data::PreparedWindow;
use kit_forecast::inference::{SampleOptions, Solver, WindowBatch, flow_sample};
use kit_forecast::market::{Market, Timescale, TradingCalendar};
use kit_forecast::model::Condition;

fn window(name: &str) -> PreparedWindow {
    let n = if name == "main" { 120 } else { 40 };
    PreparedWindow::prepare(
        &fixture_bars(n),
        Market::Ashare,
        Timescale::Min30,
        64,
        16,
        &fixture_stats(),
        3,
        5,
        &TradingCalendar::weekends_only(),
        6.0,
        5.0,
    )
    .unwrap()
}

fn nested_f32(v: &serde_json::Value, out: &mut Vec<f32>) {
    match v {
        serde_json::Value::Array(a) => a.iter().for_each(|x| nested_f32(x, out)),
        serde_json::Value::Number(n) => out.push(n.as_f64().unwrap() as f32),
        serde_json::Value::Bool(b) => out.push(f32::from(u8::from(*b))),
        other => panic!("unexpected {other}"),
    }
}

fn json_f32(v: &serde_json::Value) -> Vec<f32> {
    let mut out = Vec::new();
    nested_f32(v, &mut out);
    out
}

#[test]
fn preprocessing_matches_upstream() {
    let golden = read_json("window.json");
    for name in ["main", "pad"] {
        let w = window(name);
        let g = &golden[name];
        let x: Vec<f32> = w.x_ctx.iter().flatten().copied().collect();
        assert_close(
            &x,
            &json_f32(&g["x_ctx"]),
            2e-6,
            1e-6,
            &format!("{name}.x_ctx"),
        );
        let cond = &g["cond"];
        let flat2 = |v: &Vec<[f32; 16]>| v.iter().flatten().copied().collect::<Vec<_>>();
        assert_close(
            &flat2(&w.cond.clock),
            &json_f32(&cond["clock"]),
            1e-6,
            0.0,
            "clock",
        );
        let sess: Vec<f32> = w.cond.sess.iter().flatten().copied().collect();
        assert_close(&sess, &json_f32(&cond["sess"]), 1e-6, 0.0, "sess");
        let yday: Vec<f32> = w.cond.yday.iter().flatten().copied().collect();
        assert_close(&yday, &json_f32(&cond["yday"]), 1e-6, 0.0, "yday");
        for (key, got) in [
            ("dow", &w.cond.dow),
            ("event", &w.cond.event),
            ("month", &w.cond.month),
        ] {
            let want: Vec<i64> = json_f32(&cond[key]).iter().map(|&x| x as i64).collect();
            assert_eq!(got, &want, "{name}.{key}");
        }
        let pad: Vec<bool> = json_f32(&cond["pad_mask"])
            .iter()
            .map(|&x| x != 0.0)
            .collect();
        assert_eq!(w.cond.pad_mask, pad, "{name}.pad_mask");
        assert_eq!(w.n_pad as u64, g["n_pad"].as_u64().unwrap());
        assert_eq!(w.prev_close, g["prev_close"].as_f64().unwrap());
        let ema = g["prev_ema"].as_f64().unwrap();
        assert!((w.prev_ema - ema).abs() <= 1e-9 * ema.abs(), "prev_ema");
        assert_eq!(w.half_life_bars, g["half_life_bars"].as_f64().unwrap());
        let fut: Vec<String> = w
            .future_timestamps
            .iter()
            .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
            .collect();
        let want: Vec<String> = g["future_timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        assert_eq!(fut, want, "{name}.future_timestamps");
    }
}

fn batch_condition(w: &PreparedWindow, b: usize) -> (Tensor, Condition) {
    let batch = WindowBatch::from_windows(&vec![w; b], &Device::Cpu).unwrap();
    (batch.context().clone(), batch.condition().clone())
}

#[test]
fn model_forward_matches_upstream_at_every_stage() {
    let model = tiny_model();
    let reference = load_tensors("reference.safetensors");
    for name in ["main", "pad"] {
        let w = window(name);
        let (_, cond) = batch_condition(&w, 2);
        let z = &reference[&format!("{name}.z")];
        for tag in ["t1p0", "t0p5", "tmix"] {
            let t = &reference[&format!("{name}.{tag}.t")];
            let (v, aux) = model.forward_with_aux(z, t, &cond, 16).unwrap();
            let p = format!("{name}.{tag}");
            assert_close(
                &flat(&aux.c_base),
                &flat(&reference[&format!("{p}.c_base")]),
                2e-7,
                2e-7,
                &format!("{p}.c_base"),
            );
            assert_close(
                &flat(&aux.c_mod_ctx),
                &flat(&reference[&format!("{p}.c_mod_ctx")]),
                5e-7,
                5e-7,
                &format!("{p}.c_mod_ctx"),
            );
            assert_close(
                &flat(&aux.c_mod_tgt),
                &flat(&reference[&format!("{p}.c_mod_tgt")]),
                5e-7,
                5e-7,
                &format!("{p}.c_mod_tgt"),
            );
            assert_close(
                &flat(&aux.head_mod),
                &flat(&reference[&format!("{p}.head_mod")]),
                5e-7,
                5e-7,
                &format!("{p}.head_mod"),
            );
            assert_close(
                &flat(&v),
                &flat(&reference[&format!("{p}.v_pred")]),
                3e-6,
                3e-6,
                &format!("{p}.v_pred"),
            );
        }
    }
}

fn sample_case(name: &str, variant: &str, opts: SampleOptions) {
    let model = tiny_model();
    let reference = load_tensors("reference.safetensors");
    let w = window(name);
    let batch = WindowBatch::from_windows(&[&w], &Device::Cpu).unwrap();
    let paths = flow_sample(&model, &batch, &opts).unwrap();
    assert_eq!(paths.dims(), &[1, 3, 16, 5]);
    assert_eq!(paths.dtype(), DType::F32);
    let key = format!("sample.{name}.{variant}");
    assert_close(&flat(&paths), &flat(&reference[&key]), 5e-6, 5e-6, &key);
}

fn opts(solver: Solver, cfg_w: f64, hist_cfg_w: f64) -> SampleOptions {
    SampleOptions {
        steps: 4,
        solver,
        n_paths: 3,
        cfg_w,
        hist_cfg_w,
        seed: 7,
        chunk_size: 16,
    }
}

#[test]
fn euler_sampling_matches_upstream() {
    sample_case("main", "euler", opts(Solver::Euler, 1.0, 1.0));
}

#[test]
fn heun_sampling_matches_upstream() {
    sample_case("main", "heun", opts(Solver::Heun, 1.0, 1.0));
}

#[test]
fn identity_guidance_matches_upstream() {
    sample_case("main", "euler_idcfg", opts(Solver::Euler, 1.5, 1.0));
}

#[test]
fn history_guidance_matches_upstream() {
    sample_case("main", "euler_histcfg", opts(Solver::Euler, 1.0, 1.3));
}

#[test]
fn combined_guidance_matches_upstream_for_both_solvers() {
    sample_case("main", "euler_cfg", opts(Solver::Euler, 1.5, 1.3));
    sample_case("main", "heun_cfg", opts(Solver::Heun, 1.5, 1.3));
}

#[test]
fn padded_windows_match_upstream() {
    sample_case("pad", "euler", opts(Solver::Euler, 1.0, 1.0));
    sample_case("pad", "euler_cfg", opts(Solver::Euler, 1.5, 1.3));
}

#[test]
fn chunked_forward_equals_unchunked() {
    let model = tiny_model();
    let w = window("main");
    let batch = WindowBatch::from_windows(&[&w], &Device::Cpu).unwrap();
    let whole = flow_sample(&model, &batch, &opts(Solver::Euler, 1.5, 1.3)).unwrap();
    let mut small = opts(Solver::Euler, 1.5, 1.3);
    small.chunk_size = 2;
    let chunked = flow_sample(&model, &batch, &small).unwrap();
    assert_close(&flat(&chunked), &flat(&whole), 5e-6, 5e-6, "chunked");
}
