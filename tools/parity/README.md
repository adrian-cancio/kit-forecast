# Parity tooling

Dev-only Python that runs the **upstream** `kit_v1` code (`references/KiT`, never modified) to produce
deterministic reference data for the Rust port. Run from the repo root with the reference venv:

```
set PYTHONDONTWRITEBYTECODE=1
references/.venv/Scripts/python.exe tools/parity/make_tiny_fixture.py      # committed, ~1 MB
references/.venv/Scripts/python.exe tools/parity/make_reference.py         # git-ignored, real ckpt
```

`common.py` holds shared helpers. Floats in JSON are float64 reprs of the float32 values (round-trip exact).
All timestamps are `YYYY-MM-DD HH:MM:SS` strings. The tiny fixture script is deterministic (re-running yields byte-identical files); the reference script is seeded and fp32/CPU.

## A. `make_tiny_fixture.py` -> `tests/fixtures/tiny/`

Tiny model: d_model 32, 2 layers, 4 heads, ffn 48, `attn_backend='math'`, fp32 CPU. Every parameter is
re-drawn from `normal(0, 0.1)` (seed 4321, after `manual_seed(1234)` init), except the QK-norm gains
(`*.q_norm.weight`, `*.k_norm.weight`) which are `1 + normal(0, 0.1)`: near-zero gains would make attention uniform and hide bugs.

| File | Content |
|---|---|
| `model.safetensors`, `config.json` | f32 state_dict (49 tensors, upstream names); model config dict |
| `stats.json` | the 5 real `ashare\|30m\|*` scales from `references/data/stats.json` |
| `bars.json` | 120 synthetic ashare 30m bars `[{datetime,open,high,low,close,volume}]` |
| `window.json` | `params`; `main` (120 bars, Lc=64, Lh=16) and `pad` (first 40 bars, `n_pad=24`) windows; `features_raw_all_bars`, `features_roundtrip_ohlcv` |
| `reference.safetensors` | model/sampler stage outputs (below) |
| `rng.safetensors` | `torch.randn` bit-exact checks (below) |
| `postprocess.json` | denormalize / OHLCV / limit projection / summary goldens, `normalize` goldens |
| `calendar.json` | 5 markets x 7 scales calendar goldens |

Window params: market `ashare`, scale `30m`, sector_id 3, inst_id 5, `trading_days=None`, soft_clip_c 6, half-life 5 days.
Each window dict in `window.json` has: `x_ctx` [1,64,5], `cond` {`clock` [1,80,16], `sess` [1,80,2], `yday` [1,80,8] (f32);
`dow`,`event`,`month` [1,80], `market`,`sector`,`inst`,`scale` [1] (i64); `pad_mask` [1,80] bool}, `prev_close`, `prev_ema`,
`half_life_bars`, `n_pad`, `ctx_timestamps` (valid ctx bars only), `future_timestamps`.

### `reference.safetensors` keys (all f32)

`<w>` is `main` or `pad`. Batch B=2 (window repeated), L=Lc+Lh=80, d=32.

- `<w>.z` [2,80,5]: ctx (clean) + target noise (`randn(2,16,5)`, seed 99 main / 100 pad)
- `<w>.{t1p0,t0p5,tmix}.t` [2] (1.0 / 0.5 / [0.25, 0.9]); `.v_pred` [2,16,5]; `.c_mod_ctx` [2,192]; `.c_mod_tgt` [2,192]; `.head_mod` [2,64]; `.c_base` [2,32]
- `<w>.dbg.in_proj` [2,80,32], `.dbg.calendar` [2,80,32], `.dbg.t_emb_t` / `.dbg.t_emb_zero` [2,32] (TimestepEmbedder at t=1.0 / 0), `.dbg.block0` / `.dbg.block1` [2,84,32] (token stream incl. 4 registers; computed at t=1.0)
- `sample.<w>.<variant>` [1,3,16,5]: `flow_sample` result, steps=4, K=3, `Generator().manual_seed(7)`, fp32
- `sample.<w>.<variant>.v_calls` [n_calls, B', 16, 5]: raw output of every model forward during sampling (B' = 3 x n_branches; branch order full, noid, nohist)
- variants: `main`: `euler`, `heun` (7 calls), `euler_cfg`, `heun_cfg` (cfg_w 1.5 + hist_cfg_w 1.3), `euler_idcfg` (cfg_w 1.5), `euler_histcfg` (hist_cfg_w 1.3); `pad`: `euler`, `euler_cfg`
- initial noise for the sampler = `rng.safetensors:noise_seed7_3x16x5`

### `rng.safetensors` keys (all f32)

`noise_seed7_3x16x5` [3,16,5]; `randn.s{0,7,12345}.{1x3,16,17,1000,2x16x5,48x16x5}`; `seq.s7.{0..3}.n{3,17,16,40}` (successive `randn(n)` calls on one generator, seed 7).

### `postprocess.json`

`main_euler` / `pad_euler` (from `sample.<w>.euler`, window scalars included): `feats` [K,Lh,5] f64, `ohlcv`, `limit_0.1`, `limit_0.002`
(a tight rate that actually clips), `limit_changed_values`, `summary_feats`, `summary_ohlcv`, `summary_ohlcv_no_prev_close`,
`summary_limit_0.1` (bands q05..q95, terminal_return, max_drawdown, realized_vol). `normalize`: per-feature `scale`, `x`->`normalize`, `y`->`denormalize`.

### `calendar.json`

`trading_days` (synthetic table: weekdays 2025-12-01..2026-03-31 minus 2026-01-01 and 2026-01-09); `future` entries
(`market, scale, kind, last_ts, n, no_table, table`); `calendar` entries (`timestamps` plus `clock, sess, dow, month, yday`, `bars_per_day`,
`event_first/_prev/_prev_table/_table`, `window_event`); `offgrid` entries (timestamps snapped to session edges).

## B. `make_reference.py` -> `tools/cache/reference/<code>_<scale>_<anchor>/`

Uses the real checkpoint (`references/ckpt/step_market_284M.pt`, EMA weights), mirrors `eval/infer_for_pred.py` (it imports its
`parse_case`/`load_case_frame`/`DEFAULT_CASES`; real `trade_dates.csv`, instruments, stats), fp32 CPU, case `i` uses seed `seed+i`
(i = index in the upstream default list, even when `--cases` selects a subset).

Options: `--cases 1d,2h,...|all|0,3` `--paths 2` `--steps 32` `--solver euler|heun` `--seed 0` `--out` `--data-root` `--ckpt`.

Per case: `inputs.json` (all scalars, timestamps, seed, timings), `bars.csv` (exact history rows fed in), `future.csv` (real following bars),
`window.safetensors` (`x_ctx` f32 [1,Lc,5], `cond.*` as in the tiny window), `noise.safetensors` (`noise` f32 [K,Lh,5]),
`velocity_first.safetensors` (`v_step0`, `v_last`, `z_in_last`, `z_after_step1`, `z_after_step16`, `z_final`: f32 [K,Lh,5]; `t_step0`, `t_last` f32 scalars),
`steps.safetensors` (`z_in`, `v`: f32 [steps,K,Lh,5], model input target part / output of every step), `paths.safetensors` (`paths` f32 [1,K,Lh,5]),
`feats.safetensors` (`feats` f64 [K,Lh,5]), `ohlcv.safetensors` (`open/high/low/close/volume` and `limit0.1.*`, f64 [K,Lh]), `summary.json`.
The script asserts that the recorded step-0 input equals the replayed noise and that an Euler replay of the recorded velocities reproduces `paths` exactly.
