mod common;

use std::path::PathBuf;

use common::{fixture_bars, parse_ts};
use kit_forecast::Error;
use kit_forecast::data::Bar;
use kit_forecast::forecast::{
    Device, ForecastOptions, ForecastRequest, Instrument, InstrumentTable, Kit,
};
use kit_forecast::inference::Solver;
use kit_forecast::market::{Market, Timescale};

fn tiny_kit() -> Kit {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_ckpt.pt");
    Kit::from_pt(path, Device::Cpu).expect("tiny checkpoint loads")
}

fn request(n: usize) -> ForecastRequest {
    ForecastRequest::new(fixture_bars(n), Market::Ashare, Timescale::Min30)
        .with_instrument(Instrument::new(3, 5))
}

fn small_options() -> ForecastOptions {
    ForecastOptions {
        paths: 3,
        steps: 4,
        seed: 11,
        ..ForecastOptions::default()
    }
}

#[test]
fn forecast_has_the_documented_shape_and_invariants() {
    let kit = tiny_kit();
    let window = kit.window(Timescale::Min30);
    assert_eq!((window.context_len, window.horizon_len), (48, 8));
    let forecast = kit.forecast(&request(100), &small_options()).unwrap();
    assert_eq!(forecast.n_paths(), 3);
    assert_eq!(forecast.horizon(), 8);
    assert_eq!(forecast.timestamps.len(), 8);
    assert_eq!(forecast.anchor, fixture_bars(100).last().unwrap().timestamp);
    assert!(forecast.timestamps[0] > forecast.anchor);
    assert!(forecast.timestamps.windows(2).all(|w| w[0] < w[1]));
    let mut last_close = None;
    for path in &forecast.paths {
        for t in 0..8 {
            let (o, h, l, c) = (path.open[t], path.high[t], path.low[t], path.close[t]);
            assert!([o, h, l, c, path.volume[t]].iter().all(|x| x.is_finite()));
            assert!(h >= o.max(c) * (1.0 - 1e-12), "high below body at {t}");
            assert!(l <= o.min(c) * (1.0 + 1e-12), "low above body at {t}");
            assert!(path.volume[t] >= 0.0);
        }
        last_close = path.close.last().copied();
    }
    assert!(last_close.is_some());
    assert_eq!(forecast.prev_close, fixture_bars(100).last().unwrap().close);
    assert_eq!(forecast.terminal_returns().len(), 3);
    assert!(forecast.representative_path() < 3);
    assert_eq!(forecast.bars(0).unwrap().len(), 8);
    assert!(forecast.bars(3).is_err());
    let summary = forecast.summary().unwrap();
    assert_eq!(summary.terminal_return.samples.len(), 3);
}

#[test]
fn same_seed_is_deterministic_and_different_seed_differs() {
    let kit = tiny_kit();
    let a = kit.forecast(&request(100), &small_options()).unwrap();
    let b = kit.forecast(&request(100), &small_options()).unwrap();
    assert_eq!(a, b);
    let other = ForecastOptions {
        seed: 12,
        ..small_options()
    };
    let c = kit.forecast(&request(100), &other).unwrap();
    assert_ne!(a.paths, c.paths);
}

#[test]
fn context_beyond_the_window_is_ignored() {
    let kit = tiny_kit();
    let long = kit.forecast(&request(120), &small_options()).unwrap();
    let mut bars = fixture_bars(120);
    bars.drain(..20);
    let short = ForecastRequest::new(bars, Market::Ashare, Timescale::Min30)
        .with_instrument(Instrument::new(3, 5));
    let cut = kit.forecast(&short, &small_options()).unwrap();
    assert_eq!(long.timestamps, cut.timestamps);
    let max_rel = long.paths[0]
        .close
        .iter()
        .zip(&cut.paths[0].close)
        .map(|(a, b)| ((a - b) / a).abs())
        .fold(0.0, f64::max);
    assert!(
        max_rel < 0.2,
        "volume EMA warm-up should change little: {max_rel}"
    );
}

#[test]
fn batch_of_one_equals_single_forecast() {
    let kit = tiny_kit();
    let single = kit.forecast(&request(100), &small_options()).unwrap();
    let batch = kit
        .forecast_batch(&[request(100)], &small_options())
        .unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0], single);
}

#[test]
fn batch_returns_one_forecast_per_request_in_order() {
    let kit = tiny_kit();
    let out = kit
        .forecast_batch(
            &[request(100), request(110), request(120)],
            &small_options(),
        )
        .unwrap();
    assert_eq!(out.len(), 3);
    let anchors: Vec<_> = out.iter().map(|f| f.anchor).collect();
    let bars = fixture_bars(120);
    assert_eq!(
        anchors,
        vec![bars[99].timestamp, bars[109].timestamp, bars[119].timestamp]
    );
}

#[test]
fn limit_projection_keeps_paths_inside_the_band() {
    let kit = tiny_kit();
    let options = ForecastOptions {
        limit_rate: Some(0.002),
        ..small_options()
    };
    let f = kit.forecast(&request(100), &options).unwrap();
    for path in &f.paths {
        let mut pc = f.prev_close;
        for t in 0..path.len() {
            let (lo, hi) = (pc * 0.998 - 1e-9, pc * 1.002 + 1e-9);
            assert!(path.open[t] >= lo && path.open[t] <= hi);
            assert!(path.close[t] >= lo && path.close[t] <= hi);
            pc = path.close[t];
        }
    }
}

#[test]
fn heun_and_guidance_run_end_to_end() {
    let kit = tiny_kit();
    let options = ForecastOptions {
        solver: Solver::Heun,
        cfg_w: 1.5,
        hist_cfg_w: 1.2,
        ..small_options()
    };
    let f = kit.forecast(&request(100), &options).unwrap();
    assert!(
        f.paths
            .iter()
            .all(|p| p.close.iter().all(|c| c.is_finite()))
    );
}

#[test]
fn short_history_is_rejected_unless_padding_is_allowed() {
    let kit = tiny_kit();
    let err = kit
        .forecast(&request(30), &small_options())
        .expect_err("30 < 48 bars");
    assert!(
        matches!(
            err,
            Error::InsufficientHistory {
                required: 48,
                actual: 30
            }
        ),
        "{err}"
    );

    let padded = ForecastOptions {
        allow_padding: true,
        ..small_options()
    };
    kit.forecast(&request(30), &padded).unwrap();
    let too_short = kit
        .forecast(&request(10), &padded)
        .expect_err("below trunc_min");
    assert!(matches!(
        too_short,
        Error::InsufficientHistory { required: 16, .. }
    ));
}

#[test]
fn malformed_bars_are_rejected_with_context() {
    let kit = tiny_kit();
    let opts = small_options();

    let mut bars = fixture_bars(100);
    bars[40].high = bars[40].low * 0.5;
    let err = kit
        .forecast(
            &ForecastRequest::new(bars, Market::Ashare, Timescale::Min30),
            &opts,
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidData(m) if m.contains("bar 40")),
        "{err}"
    );

    let mut bars = fixture_bars(100);
    bars[10].close = f64::NAN;
    let err = kit
        .forecast(
            &ForecastRequest::new(bars, Market::Ashare, Timescale::Min30),
            &opts,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidData(_)));

    let mut bars = fixture_bars(100);
    bars.swap(20, 21);
    let err = kit
        .forecast(
            &ForecastRequest::new(bars, Market::Ashare, Timescale::Min30),
            &opts,
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidData(m) if m.contains("ascending")),
        "{err}"
    );

    let mut bars: Vec<Bar> = fixture_bars(100);
    bars[5].volume = -1.0;
    assert!(matches!(
        Kit::validate_bars(&bars),
        Err(Error::InvalidData(_))
    ));

    let err = kit
        .forecast(
            &ForecastRequest::new(Vec::new(), Market::Ashare, Timescale::Min30),
            &opts,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InsufficientHistory { actual: 0, .. }));
}

#[test]
fn invalid_options_are_rejected() {
    let kit = tiny_kit();
    for options in [
        ForecastOptions {
            paths: 0,
            ..small_options()
        },
        ForecastOptions {
            steps: 0,
            ..small_options()
        },
        ForecastOptions {
            chunk_size: 0,
            ..small_options()
        },
        ForecastOptions {
            cfg_w: f64::NAN,
            ..small_options()
        },
        ForecastOptions {
            limit_rate: Some(0.0),
            ..small_options()
        },
        ForecastOptions {
            limit_rate: Some(1.5),
            ..small_options()
        },
    ] {
        let err = kit.forecast(&request(100), &options).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)), "{err}");
    }
}

#[test]
fn missing_normalization_scales_are_reported() {
    let kit = tiny_kit();
    let req = ForecastRequest::new(fixture_bars(100), Market::Us, Timescale::Min30);
    let err = kit.forecast(&req, &small_options()).unwrap_err();
    assert!(
        matches!(&err, Error::InvalidData(m) if m.contains("us|30m")),
        "{err}"
    );
}

#[test]
fn mixed_batches_are_rejected() {
    let kit = tiny_kit();
    let a = request(100);
    let mut b = request(100);
    b.timescale = Timescale::Day1;
    let err = kit.forecast_batch(&[a, b], &small_options()).unwrap_err();
    assert!(matches!(err, Error::InvalidData(_)));
    assert!(matches!(
        kit.forecast_batch(&[], &small_options()),
        Err(Error::InvalidData(_))
    ));
}

#[test]
fn exported_directory_reloads_to_identical_forecasts() {
    let kit = tiny_kit();
    let dir = std::env::temp_dir().join(format!("kit-forecast-api-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    kit.save_dir(&dir).unwrap();
    let reloaded = Kit::from_dir(&dir, Device::Cpu).unwrap();
    let a = kit.forecast(&request(100), &small_options()).unwrap();
    let b = reloaded.forecast(&request(100), &small_options()).unwrap();
    assert_eq!(a, b);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn unavailable_devices_report_unsupported_hardware() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_ckpt.pt");
    if !cfg!(feature = "cuda") {
        let err = Kit::from_pt(&path, Device::Cuda(0)).unwrap_err();
        assert!(matches!(err, Error::UnsupportedHardware(_)), "{err}");
    }
    if !cfg!(feature = "metal") {
        let err = Kit::from_pt(&path, Device::Metal(0)).unwrap_err();
        assert!(matches!(err, Error::UnsupportedHardware(_)), "{err}");
    }
}

#[test]
fn bad_checkpoint_files_report_errors() {
    let missing = Kit::from_pt("/definitely/not/here.pt", Device::Cpu).unwrap_err();
    assert!(
        matches!(missing, Error::Io(_) | Error::Checkpoint(_)),
        "{missing}"
    );
    let dir = std::env::temp_dir().join(format!("kit-forecast-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let junk = dir.join("junk.pt");
    std::fs::write(&junk, b"not a zip archive").unwrap();
    let err = Kit::from_pt(&junk, Device::Cpu).unwrap_err();
    assert!(matches!(err, Error::Checkpoint(_)), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn instrument_table_reads_official_layout() {
    let table = InstrumentTable::from_json_str(
        r#"{"600681.SH": {"market": "ashare", "sector": 0, "vocab_index": 1817}}"#,
    )
    .unwrap();
    assert_eq!(table.get("600681.SH"), Some(Instrument::new(0, 1817)));
    assert_eq!(table.get("nope"), None);
    assert_eq!(table.len(), 1);
    let _ = parse_ts("2026-01-05 15:00:00");
}
