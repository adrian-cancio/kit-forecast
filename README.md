# kit-forecast

A native Rust port of **KiT** ("K-line Diffusion Transformer"), a foundation model that
forecasts candlestick (OHLCV) time series by flow matching. It loads the **official pretrained
PyTorch checkpoint directly** and runs inference on [Candle](https://github.com/huggingface/candle).
No Python, PyTorch or conversion step is needed at runtime.

> **This is an independent port, not the official implementation.** KiT is by Boyu Zhang and
> Haorui Li ([paper](https://arxiv.org/abs/2609.34507),
> [code](https://github.com/Luciferbobo/KiT)). See [Licensing](#licensing) and `NOTICE`.

## What KiT is

KiT turns an OHLCV bar into five log-ratio features
`(r_gap, r_body, r_up, r_dn, v)`, normalises them per (market, timescale, feature) with a robust
MAD scale and a `6·tanh(x/6)` soft clip, and models the next `Lh` bars as **conditional path
generation with rectified flow**. A single-stream DiT (21 layers, width 1024, 8 heads, QK-norm,
RoPE, SwiGLU, 4 register tokens, shared AdaLN) sees `[registers | history | horizon]`. Calendar
features (time of day, session position, weekday, month, day of year, gap events) are added to
every token; market, sector, instrument and timescale ids modulate every block through AdaLN.
The history passes through unchanged and only the horizon is generated.

## What this crate provides

* The complete upstream inference path: calendar and session logic for five markets, feature
  engineering with the pandas-compatible volume EMA, normalisation, the `KlineDiT` network,
  Euler/Heun rectified-flow sampling with identity and history guidance, denormalisation,
  OHLCV reconstruction, price-limit projection and path summaries.
* A pure-Rust reader for the official `.pt` file (zip + restricted pickle parser; nothing in
  the pickle is executed).
* A reimplementation of PyTorch's CPU `randn` stream, so seeds reproduce upstream noise.
* Batching, typed errors, optional CUDA and Metal features, optional Hugging Face download.

Not ported: training code (upstream has none yet), plotting, the A-share backtest and the
evaluation metric scripts.

## Quick start

```toml
[dependencies]
kit-forecast = "0.1"
```

Download `step_market_284M.pt` (1.1 GB) from
<https://huggingface.co/Lucifer744/KiT_model>, then:

```rust,no_run
use kit_forecast::data::Bar;
use kit_forecast::forecast::{Device, ForecastOptions, ForecastRequest, Instrument, Kit};
use kit_forecast::market::{Market, Timescale};

fn main() -> kit_forecast::Result<()> {
    let kit = Kit::from_pt("step_market_284M.pt", Device::Cpu)?;

    let bars: Vec<Bar> = vec![]; // 250+ daily bars, oldest first, ending at the anchor
    let request = ForecastRequest::new(bars, Market::Ashare, Timescale::Day1)
        .with_instrument(Instrument::new(0, 1817)); // ids from instruments.json, or UNKNOWN

    let forecast = kit.forecast(&request, &ForecastOptions::default())?;
    println!("{} paths of {} bars", forecast.n_paths(), forecast.horizon());
    println!("mean return {:+.2}%", 100.0 * forecast.mean_return());
    Ok(())
}
```

A runnable CSV-driven version is in `examples/forecast_csv.rs`.

### Input data

* `Vec<Bar>`: `timestamp` (bar **close** time, exchange-local, `chrono::NaiveDateTime`), `open`,
  `high`, `low`, `close`, `volume` as `f64`. Prices must be positive and adjusted; `high ≥
  max(open, close)`, `low ≤ min(open, close)`; timestamps strictly ascending. Violations return
  `Error::InvalidData` naming the bar.
* The last bar is the **anchor**; the forecast starts at the next bar on the exchange calendar.
* The model supports fixed windows (history `Lc`, horizon `Lh`); fewer than `Lc` bars gives
  `Error::InsufficientHistory` unless `ForecastOptions::allow_padding` is set.

| timescale | 1m | 5m | 15m | 30m | 1h | 2h | 1d |
|---|---|---|---|---|---|---|---|
| history `Lc` | 1200 | 960 | 800 | 640 | 400 | 360 | 250 |
| horizon `Lh` | 120 | 96 | 64 | 40 | 20 | 20 | 20 |

Supply at least a few days more history than `Lc` if you can: features use all supplied bars
so the volume EMA is warmed up (upstream keeps `Lc + 12 · 5 · bars_per_day + 8` bars).

### Calendar, instruments and normalisation

* `Kit::with_calendar(TradingCalendar::from_csv_path("trade_dates.csv")?)` makes forecast
  timestamps skip exchange holidays. The default skips weekends only. Holidays change the
  calendar features the model sees, so use the real calendar for A-shares.
* `InstrumentTable::from_json_path("instruments.json")` maps codes such as `600681.SH` to the
  sector and instrument embedding ids the model was trained with. Unknown instruments use
  `Instrument::UNKNOWN` (embedding slot 0, which the model trained with dropout).
* `Stats::official_ashare()` (default) holds the 35 normalisation scales published with the
  demo data. No scales exist for other markets: for `Us`, `Hk`, `Futures` or `Crypto` you must
  fit your own `Stats` (`Kit::with_stats`); the request otherwise fails with a clear error. The
  published checkpoint was trained on A-shares, crypto and US equities per the paper, but only
  the A-share scales and calendar templates are verifiable from public artefacts.

### Forecast options

`ForecastOptions::default()` equals the official `eval/infer_for_pred.py` settings except for
the path count default (16, as in `infer_for_eval.py`):

| option | default | meaning |
|---|---|---|
| `paths` | 16 | sampled paths `K` |
| `steps` | 32 | uniform integration steps from `t=1` to `t=0` |
| `solver` | `Euler` | or `Heun` (second order; last step is Euler) |
| `cfg_w` / `hist_cfg_w` | 1.0 | identity / history guidance (1.0 = off) |
| `seed` | 0 | noise seed (reproduces `torch.Generator().manual_seed`) |
| `chunk_size` | 16 | max sequences per forward pass (memory knob) |
| `limit_rate` | `None` | clamp to a daily price-limit band as in the upstream backtest |
| `allow_padding` | `false` | accept histories shorter than `Lc` (≥ `trunc_min`) |

Lower-level access: `Kit::prepare` (inputs only), `inference::flow_sample[_observed]`,
`inference::WindowBatch`, `model::KlineDiT`.

### Loading

* `Kit::from_pt(path, device)`: the official `.pt`.
* `Kit::save_dir` / `Kit::from_dir`: optional safetensors + JSON layout (faster to reload).
* `Kit::from_hub("Lucifer744/KiT_model", "step_market_284M.pt", device)`: with feature `hf-hub`.

Supported checkpoints: the two official files `step_market_284M.pt` and
`step_market_284M_train260410.pt` (same architecture: 21 layers, width 1024, 284M parameters).
The first was used for all parity testing. The second (training step 63000) was downloaded through `Kit::from_hub`, loads with the same strict validation and produces finite, plausible forecasts (see `tests/hub.rs`, ignored by default); it was not compared against upstream numerically. The loader validates every tensor name, shape, dtype and finiteness against
the checkpoint's own configuration and reports `MissingTensor`, `ShapeMismatch` or
`Checkpoint` errors.

## Hardware

| backend | status |
|---|---|
| CPU | supported, tested |
| CUDA (`--features cuda`) | built and tested on an RTX 4050 (sm_89, CUDA 12.6, MSVC): CPU-vs-CUDA agreement on the tiny model (6e-8) and the benchmarks below. The full official-checkpoint parity suite was run on CPU only. |
| Metal (`--features metal`) | exposed through Candle; **cannot be built or tested on Windows** (`objc2` only compiles for Apple targets), so it is unverified |

Models run in `f32` on every backend (upstream's default evaluation runs in bf16 on GPU; see
limitations).

## Numerical parity with upstream

Measured against upstream PyTorch (CPU, fp32) on the official checkpoint with real data
(K = 2 paths, 32 Euler steps, same seeds). See `tools/parity/` to regenerate everything.

| timescale | tokens | steps compared | max abs diff of normalized paths | max rel diff of OHLCV |
|---|---|---|---|---|
| 1d  | 274  | 32 | 3.1e-6 | 8.9e-7 |
| 2h  | 384  | 32 | 8.3e-6 | 1.7e-6 |
| 1h  | 424  | 32 | 4.3e-6 | 8.9e-7 |
| 30m | 684  | 32 | 4.2e-6 | 7.5e-7 |
| 15m | 868  | 32 | 6.7e-6 | 7.7e-7 |
| 5m  | 1060 | 32 | 7.1e-5 | 8.6e-6 |
| 1m  | 1324 | 32 | 2.1e-6 | 2.7e-6 |

Normalized paths have magnitude up to about 6, so 7e-5 is a relative error of ~1e-5. Every
Euler step was compared with the recorded upstream trajectory; the largest intermediate
difference was 3.3e-5 (1m, step ~24) and the error never grows beyond that.

**The 5m deviation is rounding, not logic.** To localise it we compared both float32
implementations with a float64 evaluation of the same upstream model. For one linear layer
(`blocks.0.attn.qkv`, K = 1024, identical inputs and weights, verified bit-identical) the error
against float64 is 3.8e-6 for PyTorch (MKL) and 1.8e-5 for Candle's CPU GEMM: Candle
accumulates float32 dot products less carefully. Through 21 blocks this reaches ~2e-4 in the
token stream (PyTorch: ~2e-5), which gives ~3.5e-5 in the velocity at `t = 1`. RoPE tables
(angles bit-identical, cos/sin within 6e-8) were checked separately; the remaining operations were validated end to end on the tiny model.
The tiny-model tests show every operation matches to float32 rounding. We did not tighten
this by changing the accumulation order, because that would only trade speed for matching
PyTorch's particular rounding. Diagnostics: `tools/parity/make_fp64_velocity.py`,
`make_block_trace.py`, `make_op_trace.py`, `make_gemm_probe.py` and the ignored tests in
`tests/official_parity.rs`.

On the synthetic tiny model (random non-zero weights in every tensor, `d_model` 32), every
stage matches to float32 rounding: AdaLN vectors and the velocity to ≤ 4e-7, Euler/Heun
sampling with and without both guidance branches to ≤ 5e-7, padded windows included.
Preprocessing (features, normalisation, calendar, events, timestamps) is bit-identical on the
real data. The noise generator matches `torch.randn` bit-for-bit for fewer than 16 values
and within 3 ulp for larger tensors (see limitations).

## Known limitations

* **Precision.** Everything runs in `f32`. Upstream's evaluation uses bf16 autocast on CUDA, so
  GPU results from the Python scripts differ from ours at bf16 rounding level; the fp32 CPU
  path of upstream is what we match.
* **Noise.** `TorchRng` reproduces the Mersenne-Twister stream and layout of PyTorch's CPU
  `randn`. For tensors of 16 or more values PyTorch uses a vectorised `logf/cosf/sinf`; the
  Rust port evaluates the same formula in extended precision and differs by at most 3 ulp
  (~1e-7), which does not matter for the sampler but means noise is not bit-exact. CUDA noise
  in PyTorch uses a different generator and cannot be reproduced; this port always draws on
  the CPU, as upstream does (`gen_device` is CPU unless a CUDA generator is passed).
* **Attention** materialises the full `S × S` score matrix (upstream's `math` backend), so
  memory grows quadratically in sequence length. The 1m model (1324 tokens) needs about
  8 × 1324² × 4 B ≈ 56 MB per sequence per layer for the scores.
* **Calendar.** The upstream `detect_events` and `gen_future_timestamps` use slightly different
  weekend rules for 24/7 markets; both are reproduced as is. Without a holiday table, holidays
  are treated as trading days.
* **Markets.** Only A-share normalisation scales are public.
* **Speed.** Candle's CPU matmul is far slower than PyTorch's MKL path; see Performance.

## Performance

Measured on an Intel Core i7-12650H (12 threads), 16 GB RAM, NVIDIA RTX 4050 Laptop GPU (6 GB),
Windows 11, Rust 1.96 release build, f32, K = 2 paths, 32 Euler steps, median of 3-5 runs for
the small items and a single run for the full forecast. Checkpoint load: 1.4-2.7 s on CPU
(warm file cache), 3.7 s to CUDA.

| scale | Lc / Lh | device | preprocessing | 1 model evaluation | full forecast | per step |
|---|---|---|---|---|---|---|
| 1d  | 250 / 20  | CPU  | 0.12 ms | 2.50 s | 84.1 s  | 2.63 s |
| 1h  | 400 / 20  | CPU  | 0.19 ms | 4.31 s | 140.4 s | 4.39 s |
| 30m | 640 / 40  | CPU  | 0.42 ms | 8.69 s | 284.8 s | 8.90 s |
| 5m  | 960 / 96  | CPU  | 1.37 ms | 15.24 s | 494.0 s | 15.44 s |
| 1d  | 250 / 20  | CUDA | 0.10 ms | 0.18 s | 5.30 s  | 0.17 s |
| 1h  | 400 / 20  | CUDA | 0.13 ms | 0.25 s | 8.61 s  | 0.27 s |
| 30m | 640 / 40  | CUDA | 0.23 ms | 0.46 s | 15.6 s  | 0.49 s |
| 5m  | 960 / 96  | CUDA | 0.77 ms | 0.89 s | 30.1 s  | 0.94 s |

Not measured cleanly: CPU for 15m, 2h and 1m (the parity runs took 160-790 s per case but shared the machine with builds), and CUDA for 15m, 2h and 1m. The GPU
stopped responding ("GPU is lost") during the 1m run and needs a reboot. The CPU numbers show
that Candle's CPU path is much slower than PyTorch's; this port is practical on CPU only for
short forecasts or small `paths`. "1 model evaluation" is one batched forward over all paths
(`steps = 1` run), so a full forecast is roughly `steps` times that. The bench is
`benches/forecast.rs`. Run-to-run noise was visible (the first 1d CPU run measured 161 s
while the machine was busy with a build), so treat these as indicative.

## Testing

```text
cargo test                                   # unit + tiny-model parity + API tests (~1 min)
cargo test --release --test official_parity -- --ignored --nocapture
                                             # real checkpoint vs. upstream dumps (slow)
cargo test --release --features cuda --test cuda_smoke
```

The official-checkpoint tests need the 1.1 GB checkpoint and the dumps produced by
`tools/parity/make_reference.py`; both live under the git-ignored `references/` and
`tools/cache/` directories and are never packaged.

## Licensing

* **Code:** AGPL-3.0-only. This crate is a direct port of the AGPL-3.0 KiT sources, so it is
  distributed under the same licence (SPDX `AGPL-3.0-only`; the upstream licence file is the
  unmodified GNU AGPLv3 text, and neither the README nor any source file adds "or later"). If
  you offer this library's functionality to users over a network, AGPL section 13 requires you
  to offer them the corresponding source.
* **Weights and data:** not included and **not covered by the above**. Neither Hugging Face
  repository (`Lucifer744/KiT_model`, `Lucifer744/KiT_data_demo`) declares a licence; their cards
  say "See the main repository for license information", and the code repository's licence does
  not mention weights or data. Obtain the checkpoint yourself and clarify terms with the
  authors before redistributing weights/data or using them commercially.
* `src/data/ashare_stats.json` is a verbatim copy of the demo dataset's `stats.json`
  (35 numeric constants required to interpret the model's inputs and outputs).
* Test fixtures under `tests/fixtures/` come from synthetic data and random weights.

See `NOTICE` for attribution and the list of modifications.
