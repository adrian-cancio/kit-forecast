//! Forecast from a CSV file of bars.
//!
//! ```text
//! cargo run --release --example forecast_csv -- \
//!     step_market_284M.pt bars.csv 1d 300368.SZ_instruments.json
//! ```
//!
//! `bars.csv` has the header `datetime,open,high,low,close,volume` with timestamps like
//! `2026-01-05 15:00:00` (bar close time, exchange-local). Arguments: checkpoint, CSV,
//! timescale (`1m 5m 15m 30m 1h 2h 1d`), and optionally `sector,id` embedding ids.

use std::process::ExitCode;

use kit_forecast::data::Bar;
use kit_forecast::forecast::{Device, ForecastOptions, ForecastRequest, Instrument, Kit};
use kit_forecast::market::{Market, Timescale, Timestamp};

fn read_bars(path: &str) -> Result<Vec<Bar>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut bars = Vec::new();
    for (line_no, line) in text.lines().enumerate().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 6 {
            return Err(format!("line {}: expected 6 columns", line_no + 1).into());
        }
        let ts = Timestamp::parse_from_str(f[0], "%Y-%m-%d %H:%M:%S")?;
        let n = |i: usize| f[i].parse::<f64>();
        bars.push(Bar::new(ts, n(1)?, n(2)?, n(3)?, n(4)?, n(5)?));
    }
    Ok(bars)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "usage: forecast_csv <checkpoint.pt> <bars.csv> <timescale> [sector,id]".into(),
        );
    }
    let timescale: Timescale = args[2].parse()?;
    let instrument = match args.get(3) {
        Some(ids) => {
            let (s, i) = ids
                .split_once(',')
                .ok_or("ids must look like `sector,id`")?;
            Instrument::new(s.parse()?, i.parse()?)
        }
        None => Instrument::UNKNOWN,
    };

    let device = Device::best_available();
    let kit = Kit::from_pt(&args[0], device)?;
    let request = ForecastRequest::new(read_bars(&args[1])?, Market::Ashare, timescale)
        .with_instrument(instrument);
    let options = ForecastOptions {
        paths: 16,
        ..ForecastOptions::default()
    };
    let (forecast, seconds) = kit.forecast_timed(&request, &options)?;

    println!(
        "device {device}, {} paths x {} bars in {seconds:.1}s, anchor {}",
        forecast.n_paths(),
        forecast.horizon(),
        forecast.anchor
    );
    let summary = forecast.summary()?;
    println!(
        "terminal return: median {:+.2}%  q05 {:+.2}%  q95 {:+.2}%",
        100.0 * summary.terminal_return.q50,
        100.0 * summary.terminal_return.q05,
        100.0 * summary.terminal_return.q95
    );
    let rep = forecast.bars(forecast.representative_path())?;
    println!("representative path:");
    for b in rep.iter().take(10) {
        println!(
            "  {}  O {:.3} H {:.3} L {:.3} C {:.3} V {:.0}",
            b.timestamp, b.open, b.high, b.low, b.close, b.volume
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
