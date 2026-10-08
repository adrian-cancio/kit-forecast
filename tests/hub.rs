#![cfg(feature = "hf-hub")]

use kit_forecast::forecast::{Device, Kit};
use kit_forecast::market::Timescale;

#[test]
#[ignore = "downloads the 1.1 GB official checkpoint from Hugging Face"]
fn downloads_and_loads_the_second_official_checkpoint() {
    let kit = Kit::from_hub(
        "Lucifer744/KiT_model",
        "step_market_284M_train260410.pt",
        Device::Cpu,
    )
    .expect("download and load");
    let info = kit.checkpoint_info();
    println!("step {:?}, ema decay {:?}", info.step, info.ema_decay);
    assert_eq!(kit.config().model.n_layers, 21);
    assert_eq!(kit.config().model.d_model, 1024);
    let w = kit.window(Timescale::Day1);
    assert_eq!((w.context_len, w.horizon_len), (250, 20));
    let w = kit.window(Timescale::Min1);
    assert_eq!((w.context_len, w.horizon_len), (1200, 120));
}

#[test]
#[ignore = "needs the cached second checkpoint; run after the download test"]
fn second_checkpoint_forecasts_sane_paths() {
    use kit_forecast::data::Bar;
    use kit_forecast::forecast::{ForecastOptions, ForecastRequest, Instrument};
    use kit_forecast::market::{Market, Timestamp};

    let kit = Kit::from_hub(
        "Lucifer744/KiT_model",
        "step_market_284M_train260410.pt",
        Device::Cpu,
    )
    .unwrap();
    let csv = std::env::var("KIT_SANITY_BARS").expect("path to a 1d bars.csv");
    let text = std::fs::read_to_string(csv).unwrap();
    let bars: Vec<Bar> = text
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let n = |i: usize| f[i].parse::<f64>().unwrap();
            let ts = Timestamp::parse_from_str(f[0], "%Y-%m-%d %H:%M:%S").unwrap();
            Bar::new(ts, n(1), n(2), n(3), n(4), n(5))
        })
        .collect();
    let request = ForecastRequest::new(bars, Market::Ashare, Timescale::Day1)
        .with_instrument(Instrument::UNKNOWN);
    let options = ForecastOptions {
        paths: 4,
        steps: 8,
        ..ForecastOptions::default()
    };
    let f = kit.forecast(&request, &options).unwrap();
    let summary = f.summary().unwrap();
    println!(
        "terminal return q05 {:+.3} q50 {:+.3} q95 {:+.3}",
        summary.terminal_return.q05, summary.terminal_return.q50, summary.terminal_return.q95
    );
    for p in &f.paths {
        assert!(p.close.iter().all(|c| c.is_finite() && *c > 0.0));
        assert!(p.volume.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
    assert!(
        summary.terminal_return.q05.abs() < 1.0,
        "implausible 20-day return"
    );
}
