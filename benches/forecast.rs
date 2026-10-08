//! Wall-clock benchmarks of the inference paths.
//!
//! ```text
//! KIT_CKPT=references/ckpt/step_market_284M.pt \
//! KIT_BENCH_DATA=tools/cache/reference \
//! cargo bench --bench forecast                  # CPU
//! cargo bench --bench forecast --features cuda  # CUDA, if built with the toolkit
//! ```
//!
//! `KIT_BENCH_DATA` points at the per-case directories written by
//! `tools/parity/make_reference.py` (each holds `bars.csv`). `KIT_BENCH_SCALES` selects
//! timescales (default `1d,1h,30m`), `KIT_BENCH_PATHS` the number of paths (default 2),
//! `KIT_BENCH_DEVICE=cuda` the GPU. Results are printed as a table.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device as BackendDevice;
use kit_forecast::data::Bar;
use kit_forecast::forecast::{Device, ForecastOptions, ForecastRequest, Instrument, Kit};
use kit_forecast::inference::{SampleOptions, WindowBatch};
use kit_forecast::market::{Market, Timescale, Timestamp};

fn read_bars(path: &PathBuf) -> Vec<Bar> {
    let text = std::fs::read_to_string(path).expect("bars.csv readable");
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let n = |i: usize| f[i].parse::<f64>().expect("number");
            let ts = Timestamp::parse_from_str(f[0], "%Y-%m-%d %H:%M:%S").expect("timestamp");
            Bar::new(ts, n(1), n(2), n(3), n(4), n(5))
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        std::hint::black_box(f());
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn main() {
    let Ok(ckpt) = std::env::var("KIT_CKPT") else {
        eprintln!("set KIT_CKPT to the checkpoint path (and KIT_BENCH_DATA to the bar dumps)");
        return;
    };
    let data =
        PathBuf::from(std::env::var("KIT_BENCH_DATA").unwrap_or("tools/cache/reference".into()));
    let scales = std::env::var("KIT_BENCH_SCALES").unwrap_or("1d,1h,30m".into());
    let paths: usize = std::env::var("KIT_BENCH_PATHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let device = match std::env::var("KIT_BENCH_DEVICE").as_deref() {
        Ok("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };

    let load_start = Instant::now();
    let kit = Kit::from_pt(&ckpt, device).expect("checkpoint loads");
    println!(
        "device {device}: checkpoint load {:.2}s",
        load_start.elapsed().as_secs_f64()
    );
    println!(
        "{:<6} {:>5} {:>4} {:>6} {:>12} {:>14} {:>16} {:>14}",
        "scale",
        "Lc",
        "Lh",
        "paths",
        "prepare [ms]",
        "1 forward [s]",
        "32-step total [s]",
        "per step [s]"
    );

    for scale in scales.split(',').map(str::trim) {
        let timescale: Timescale = scale.parse().expect("timescale");
        let dir = std::fs::read_dir(&data)
            .expect("KIT_BENCH_DATA readable")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.split('_').nth(1) == Some(scale))
            })
            .unwrap_or_else(|| panic!("no dump for scale {scale} in {}", data.display()));
        let request =
            ForecastRequest::new(read_bars(&dir.join("bars.csv")), Market::Ashare, timescale)
                .with_instrument(Instrument::new(0, 1));
        let options = ForecastOptions {
            paths,
            chunk_size: 8,
            ..ForecastOptions::default()
        };
        let window = kit.window(timescale);

        let prepare = time(5, || kit.prepare(&request, &options).expect("prepare"));
        let prepared = kit.prepare(&request, &options).expect("prepare");

        let backend = match device {
            Device::Cpu => BackendDevice::Cpu,
            Device::Cuda(i) => BackendDevice::new_cuda(i).expect("cuda"),
            Device::Metal(i) => BackendDevice::new_metal(i).expect("metal"),
        };
        let batch = WindowBatch::from_windows(&[&prepared], &backend).expect("batch");
        let one = SampleOptions {
            steps: 1,
            n_paths: paths,
            chunk_size: 8,
            ..SampleOptions::default()
        };
        let forward = time(3, || {
            let t =
                kit_forecast::inference::flow_sample(kit.model(), &batch, &one).expect("one step");
            t.to_device(&BackendDevice::Cpu).expect("sync")
        });

        let total = time(1, || kit.forecast(&request, &options).expect("forecast"));
        println!(
            "{:<6} {:>5} {:>4} {:>6} {:>12.2} {:>14.3} {:>16.2} {:>14.3}",
            scale,
            window.context_len,
            window.horizon_len,
            paths,
            prepare * 1e3,
            forward,
            total,
            total / 32.0
        );
    }
}
