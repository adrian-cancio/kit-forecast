#!/usr/bin/env python
"""Generate the committed tiny parity fixtures in tests/fixtures/tiny/ from the upstream kit_v1 code.

Run from the repo root:
    references/.venv/Scripts/python.exe tools/parity/make_tiny_fixture.py

Everything is deterministic (fixed seeds, fp32, CPU, attn_backend='math').
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import (KIT_ROOT, ROOT, fmt_ts, fmt_ts_list, to_py, write_json,  # noqa: F401 (sets sys.path)
                    write_safetensors)

import numpy as np
import pandas as pd
import torch

from kit_v1.config import ModelConfig
from kit_v1.data.calendar import (MARKET_IDS, TIMESCALE_IDS, _daily_grid_minutes,
                                  attach_window_calendar, bars_per_day, detect_events,
                                  gen_calendar, gen_future_timestamps)
from kit_v1.data.features import FEATURE_NAMES, compute_features, invert_features
from kit_v1.data.normalize import denormalize, load_stats, normalize, stats_key
from kit_v1.infer import (apply_limit_projection, denormalize_paths, flow_sample,
                          paths_to_ohlcv, prepare_inference_window, summarize_paths)
from kit_v1.model import KlineDiT

MARKET, SCALE = "ashare", "30m"
LC, LH = 64, 16
SECTOR_ID, INST_ID = 3, 5
SOFT_CLIP_C, HALF_LIFE_DAYS = 6.0, 5.0
N_BARS, N_BARS_PAD = 120, 40
MODEL_SEED, PARAM_SEED = 1234, 4321
SAMPLER_SEED = 7
NORM_WEIGHT_NAMES = ("q_norm.weight", "k_norm.weight")

TINY_MODEL = dict(
    d_model=32, n_layers=2, n_heads=4, ffn_hidden=48, in_features=5, rope_theta=10000.0,
    n_registers=4, inst_vocab=16, inst_dim=8, market_slots=16, sector_slots=64,
    scale_slots=16, t_emb_dim=16, clock_dim=16, sess_hidden=16, yday_dim=8,
    qk_norm=True, attn_backend="math", use_activation_checkpointing=False,
)


# ----------------------------------------------------------------------------- model

def build_model() -> KlineDiT:
    """Tiny upstream model with every parameter re-randomized (zero-init params would hide bugs).

    Parameters are drawn from normal(0, 0.1); the QK-norm gains are drawn from 1 + normal(0, 0.1)
    instead, since near-zero gains would collapse attention to a uniform average.
    """
    torch.manual_seed(MODEL_SEED)
    model = KlineDiT(ModelConfig(**TINY_MODEL))
    g = torch.Generator().manual_seed(PARAM_SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.1)
            if name.endswith(NORM_WEIGHT_NAMES):
                p.add_(1.0)
    return model.eval()


# ----------------------------------------------------------------------------- bars / window

def make_bars(n: int) -> pd.DataFrame:
    """Synthetic ashare 30m OHLCV: seeded random walk on the real session grid (close-stamped)."""
    rng = np.random.default_rng(2024)
    grid = [600, 630, 660, 690, 810, 840, 870, 900]
    days = pd.bdate_range("2025-12-15", periods=math.ceil(n / len(grid)))
    stamps = [d + pd.Timedelta(minutes=m) for d in days for m in grid][:n]
    close, rows = 25.0, []
    for i, ts in enumerate(stamps):
        o = close * math.exp(rng.normal(0.0, 0.002))
        c = o * math.exp(rng.normal(0.0, 0.004))
        h = max(o, c) * math.exp(abs(rng.normal(0.0, 0.002)))
        l = min(o, c) * math.exp(-abs(rng.normal(0.0, 0.002)))
        o, h, l, c = (round(x, 2) for x in (o, h, l, c))
        h, l = max(h, o, c), min(l, o, c)
        u_shape = 1.0 + 1.5 * (abs(i % len(grid) - 3.5) / 3.5) ** 2
        vol = float(round(1.0e6 * u_shape * math.exp(rng.normal(0.0, 0.35))))
        rows.append((ts, o, h, l, c, vol))
        close = c
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def window_to_dict(win: dict) -> dict:
    cond = {k: v for k, v in win["cond"].items()}
    return {
        "x_ctx": win["x_ctx"], "cond": cond,
        "prev_close": win["prev_close"], "prev_ema": win["prev_ema"],
        "half_life_bars": win["half_life_bars"], "n_pad": win["n_pad"],
        "ctx_timestamps": fmt_ts_list(win["ctx_timestamps"]),
        "future_timestamps": fmt_ts_list(win["future_timestamps"]),
    }


# ----------------------------------------------------------------------------- reference tensors

class Recorder:
    """Records every output of a module's forward calls (and optionally the inputs)."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.outputs: list = []
        self.handle = module.register_forward_hook(lambda m, a, o: self.outputs.append(o.detach().clone()))

    def close(self) -> None:
        self.handle.remove()


def fmt_t(t: float) -> str:
    return f"{t:.2f}".replace(".", "p")


@torch.no_grad()
def forward_refs(model, name, win, noise_seed, out: dict) -> torch.Tensor:
    """Stage outputs of the model on a B=2 batch built from one window."""
    B = 2
    g = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(B, LH, 5, generator=g)
    x_ctx = win["x_ctx"].repeat(B, 1, 1)
    z = torch.cat([x_ctx, noise], dim=1)
    cond = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in win["cond"].items()}
    out[f"{name}.z"] = z
    ts = [("t1p0", [1.0, 1.0]), ("t0p5", [0.5, 0.5]), ("tmix", [0.25, 0.9])]
    for tag, tv in ts:
        t = torch.tensor(tv, dtype=torch.float32)
        rec = None
        if tag == "t1p0":
            hooks = {"in_proj": model.in_proj, "calendar": model.calendar, "t_emb": model.t_emb}
            hooks.update({f"block{i}": b for i, b in enumerate(model.blocks)})
            rec = {k: Recorder(m) for k, m in hooks.items()}
        v, aux = model(z, t, cond, LH, return_aux=True)
        out[f"{name}.{tag}.t"] = t
        out[f"{name}.{tag}.v_pred"] = v
        for k, a in aux.items():
            out[f"{name}.{tag}.{k}"] = a
        if rec:
            for k, r in rec.items():
                if k == "t_emb":
                    out[f"{name}.dbg.t_emb_t"], out[f"{name}.dbg.t_emb_zero"] = r.outputs
                else:
                    out[f"{name}.dbg.{k}"] = r.outputs[0]
                r.close()
    return z


@torch.no_grad()
def sampler_refs(model, name, win, variants, out: dict) -> dict:
    """flow_sample results (and the raw velocity calls) per variant; returns paths by variant."""
    paths_by = {}
    for tag, kw in variants.items():
        calls = Recorder(model)
        paths = flow_sample(model, win["x_ctx"], win["cond"], Lh=LH, steps=4, n_paths=3,
                            generator=torch.Generator().manual_seed(SAMPLER_SEED),
                            device=torch.device("cpu"), precision="fp32", **kw)
        calls.close()
        out[f"sample.{name}.{tag}"] = paths
        out[f"sample.{name}.{tag}.v_calls"] = torch.stack(calls.outputs)   # [n_calls, B', Lh, 5]
        paths_by[tag] = paths
    return paths_by


def rng_refs() -> dict:
    out = {}
    out["noise_seed7_3x16x5"] = torch.randn(3, LH, 5, generator=torch.Generator().manual_seed(SAMPLER_SEED))
    shapes = [(1, 3), (16,), (17,), (1000,), (2, 16, 5), (48, 16, 5)]
    for seed in (0, 7, 12345):
        for shp in shapes:
            g = torch.Generator().manual_seed(seed)
            out[f"randn.s{seed}.{'x'.join(map(str, shp))}"] = torch.randn(*shp, generator=g)
    g = torch.Generator().manual_seed(SAMPLER_SEED)
    for i, n in enumerate((3, 17, 16, 40)):       # successive calls on one generator
        out[f"seq.s7.{i}.n{n}"] = torch.randn(n, generator=g)
    return out


# ----------------------------------------------------------------------------- postprocess

def postprocess_goldens(paths, win, stats) -> dict:
    feats = denormalize_paths(paths, MARKET, SCALE, stats, c=SOFT_CLIP_C)[0]            # [K,Lh,5]
    ohlcv = paths_to_ohlcv(feats, win["prev_close"], win["prev_ema"], win["half_life_bars"])
    limited = {rate: apply_limit_projection(ohlcv, win["prev_close"], rate) for rate in (0.1, 0.002)}
    binding = {str(r): bool(any(not np.array_equal(limited[r][k], ohlcv[k]) for k in ohlcv)) for r in limited}
    return {
        "prev_close": win["prev_close"], "prev_ema": win["prev_ema"], "half_life_bars": win["half_life_bars"],
        "feats": feats, "ohlcv": ohlcv,
        "limit_0.1": limited[0.1], "limit_0.002": limited[0.002], "limit_changed_values": binding,
        "summary_feats": summarize_paths(feats),
        "summary_ohlcv": summarize_paths(ohlcv, win["prev_close"]),
        "summary_ohlcv_no_prev_close": summarize_paths(ohlcv),
        "summary_limit_0.1": summarize_paths(limited[0.1], win["prev_close"]),
    }


def normalize_goldens(stats) -> dict:
    xs = np.array([-1.0, -0.2, -0.01, -1e-4, 0.0, 1e-4, 0.003, 0.02, 0.5, 3.0, 1e3, -1e3])
    ys = np.array([-6.0, -5.9999999, -5.5, -3.0, -0.5, 0.0, 1e-5, 0.7, 2.0, 5.99, 5.9999999, 6.0, 7.0])
    out = {}
    for name in FEATURE_NAMES:
        s = float(stats[stats_key(MARKET, SCALE, name)])
        out[name] = {"scale": s, "x": xs, "normalize": normalize(xs, s, c=SOFT_CLIP_C),
                     "y": ys, "denormalize": denormalize(ys, s, c=SOFT_CLIP_C)}
    return out


# ----------------------------------------------------------------------------- calendar

def trading_days_table() -> np.ndarray:
    days = pd.bdate_range("2025-12-01", "2026-03-31")
    days = days.drop([pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-09")])
    return days.to_numpy().astype("datetime64[D]")


def sample_series(market: str, scale: str) -> pd.DatetimeIndex:
    """Chronological bars around day starts / ends / session breaks of Wed 01-07, Thu 01-08, Mon 01-12."""
    grid = _daily_grid_minutes(market, scale)
    from kit_v1.data.calendar import _market_session_grids
    breaks, n = [], 0
    for g in _market_session_grids(market, scale)[:-1]:
        n += len(g)
        breaks.append(n)
    picks = {0, 1, len(grid) - 2, len(grid) - 1}
    for b in breaks:
        picks |= {b - 2, b - 1, b, b + 1}
    idx = sorted(i for i in picks if 0 <= i < len(grid))
    out = []
    for d in ("2026-01-07", "2026-01-08", "2026-01-12"):
        day = pd.Timestamp(d)
        out += [day + pd.Timedelta(minutes=grid[i]) for i in idx]
    return pd.DatetimeIndex(out)


def calendar_goldens() -> dict:
    td = trading_days_table()
    out = {"trading_days": [str(d) for d in td], "future": [], "calendar": [], "offgrid": []}
    for market in MARKET_IDS:
        for scale in TIMESCALE_IDS:
            grid = _daily_grid_minutes(market, scale)
            day = pd.Timestamp("2026-01-08")
            lasts = {"thu_end": day + pd.Timedelta(minutes=grid[-1]),
                     "thu_mid": day + pd.Timedelta(minutes=grid[len(grid) // 2]),
                     "table_end": pd.Timestamp("2026-03-31") + pd.Timedelta(minutes=grid[-1])}
            for kind, last in lasts.items():
                out["future"].append({
                    "market": market, "scale": scale, "kind": kind, "last_ts": fmt_ts(last), "n": 6,
                    "no_table": fmt_ts_list(gen_future_timestamps(last, 6, market, scale)),
                    "table": fmt_ts_list(gen_future_timestamps(last, 6, market, scale, trading_days=td))})
            ts = sample_series(market, scale)
            prev = pd.Timestamp("2026-01-06") + pd.Timedelta(minutes=grid[-1])
            cal = gen_calendar(ts, market, scale)
            out["calendar"].append({
                "market": market, "scale": scale, "timestamps": fmt_ts_list(ts),
                **cal,
                "bars_per_day": bars_per_day(market, scale),
                "prev_ts": fmt_ts(prev),
                "event_first": detect_events(ts, market, scale),
                "event_prev": detect_events(ts, market, scale, prev_ts=prev),
                "event_prev_table": detect_events(ts, market, scale, prev_ts=prev, trading_days=td),
                "event_table": detect_events(ts, market, scale, trading_days=td),
                "window_event_n_ctx": len(ts) // 2,
                "window_event": attach_window_calendar(ts, market, scale, n_ctx=len(ts) // 2,
                                                       prev_ts=prev, trading_days=td)["event"]})
    off = {"ashare": ["2026-01-08 09:31:00", "2026-01-08 11:45:00", "2026-01-08 12:10:00", "2026-01-08 15:20:00"],
           "us": ["2026-01-08 09:40:00", "2026-01-08 16:10:00"]}
    for market, tss in off.items():
        for scale in ("30m", "1h"):
            out["offgrid"].append({"market": market, "scale": scale, "timestamps": tss,
                                   **gen_calendar(tss, market, scale)})
    return out


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "tiny")
    args = ap.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sizes = {}

    real_stats = load_stats(ROOT / "references" / "data" / "stats.json")
    stats = {k: v for k, v in real_stats.items() if k.startswith(f"{MARKET}|{SCALE}|")}
    assert len(stats) == 5, stats
    sizes["stats.json"] = write_json(out_dir / "stats.json", stats, indent=1)

    # model
    model = build_model()
    sd = {k: v.to(torch.float32) for k, v in model.state_dict().items()}
    sizes["model.safetensors"] = write_safetensors(out_dir / "model.safetensors", sd)
    sizes["config.json"] = write_json(out_dir / "config.json", TINY_MODEL, indent=2)

    # bars + windows
    bars = make_bars(N_BARS)
    bars_json = [{"datetime": fmt_ts(r.datetime), "open": r.open, "high": r.high, "low": r.low,
                  "close": r.close, "volume": r.volume} for r in bars.itertuples()]
    sizes["bars.json"] = write_json(out_dir / "bars.json", bars_json)

    common = dict(Lc=LC, Lh=LH, stats=stats, soft_clip_c=SOFT_CLIP_C, half_life_days=HALF_LIFE_DAYS,
                  sector_id=SECTOR_ID, inst_id=INST_ID, trading_days=None)
    wins = {"main": prepare_inference_window(bars.copy(), MARKET, SCALE, **common),
            "pad": prepare_inference_window(bars.iloc[:N_BARS_PAD].copy(), MARKET, SCALE, **common)}
    assert wins["main"]["n_pad"] == 0 and wins["pad"]["n_pad"] == LC - N_BARS_PAD

    hl = wins["main"]["half_life_bars"]
    raw = compute_features(*(bars[c].to_numpy() for c in ("open", "high", "low", "close", "volume")),
                           half_life_bars=hl)
    rt = invert_features(raw["r_gap"], raw["r_body"], raw["r_up"], raw["r_dn"], raw["v"],
                         prev_close=bars["open"].iloc[0], prev_ema=bars["volume"].iloc[0], half_life_bars=hl)
    window_json = {
        "params": {"market": MARKET, "scale": SCALE, "Lc": LC, "Lh": LH, "sector_id": SECTOR_ID,
                   "inst_id": INST_ID, "soft_clip_c": SOFT_CLIP_C, "half_life_days": HALF_LIFE_DAYS,
                   "trading_days": None, "market_id": MARKET_IDS[MARKET], "scale_id": TIMESCALE_IDS[SCALE],
                   "n_bars": {"main": N_BARS, "pad": N_BARS_PAD}},
        "main": window_to_dict(wins["main"]),
        "pad": window_to_dict(wins["pad"]),
        "features_raw_all_bars": raw,
        "features_roundtrip_ohlcv": rt,
    }
    sizes["window.json"] = write_json(out_dir / "window.json", window_json)

    # reference stage outputs
    ref: dict = {}
    forward_refs(model, "main", wins["main"], 99, ref)
    forward_refs(model, "pad", wins["pad"], 100, ref)
    variants_main = {
        "euler": dict(solver="euler"), "heun": dict(solver="heun"),
        "euler_cfg": dict(solver="euler", cfg_w=1.5, hist_cfg_w=1.3),
        "heun_cfg": dict(solver="heun", cfg_w=1.5, hist_cfg_w=1.3),
        "euler_idcfg": dict(solver="euler", cfg_w=1.5),
        "euler_histcfg": dict(solver="euler", hist_cfg_w=1.3),
    }
    variants_pad = {"euler": dict(solver="euler"), "euler_cfg": dict(solver="euler", cfg_w=1.5, hist_cfg_w=1.3)}
    paths_main = sampler_refs(model, "main", wins["main"], variants_main, ref)
    paths_pad = sampler_refs(model, "pad", wins["pad"], variants_pad, ref)
    sizes["reference.safetensors"] = write_safetensors(out_dir / "reference.safetensors", ref)
    sizes["rng.safetensors"] = write_safetensors(out_dir / "rng.safetensors", rng_refs())

    post = {"params": {"limit_rates": [0.1, 0.002], "steps": 4, "n_paths": 3, "sampler_seed": SAMPLER_SEED},
            "main_euler": postprocess_goldens(paths_main["euler"], wins["main"], stats),
            "pad_euler": postprocess_goldens(paths_pad["euler"], wins["pad"], stats),
            "normalize": normalize_goldens(stats)}
    sizes["postprocess.json"] = write_json(out_dir / "postprocess.json", post)
    sizes["calendar.json"] = write_json(out_dir / "calendar.json", calendar_goldens())

    for k, v in sizes.items():
        print(f"{k:24s}{v:>10,d} bytes")
    print(f"{'TOTAL':24s}{sum(sizes.values()):>10,d} bytes")
    ranges = {k: (float(v.abs().max())) for k, v in ref.items() if k.endswith("v_pred")}
    print("max |v_pred|:", {k: round(v, 3) for k, v in ranges.items()})
    print("limit projection changed values:", post["main_euler"]["limit_changed_values"])


if __name__ == "__main__":
    main()
