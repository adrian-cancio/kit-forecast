#!/usr/bin/env python
"""Generate real-checkpoint reference data (git-ignored) into tools/cache/reference/<case_tag>/.

Mirrors references/KiT/eval/infer_for_pred.py for window preparation and sampling (fp32, CPU),
using the upstream helpers directly (parse_case / load_case_frame / DEFAULT_CASES).

Run from the repo root:
    references/.venv/Scripts/python.exe tools/parity/make_reference.py --cases 1d,2h,1h
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import time
from pathlib import Path

from common import KIT_ROOT, ROOT, fmt_ts, fmt_ts_list, write_json, write_safetensors

import numpy as np
import pandas as pd
import torch

MARKET = "ashare"
SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")


def load_upstream_script():
    """Import eval/infer_for_pred.py as a module (no side effects besides sys.path)."""
    spec = importlib.util.spec_from_file_location("kit_infer_for_pred", KIT_ROOT / "eval" / "infer_for_pred.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StepRecorder:
    """Captures, for every model call, the target part of the input z and the output velocity."""

    def __init__(self, model: torch.nn.Module, Lc: int) -> None:
        self.Lc, self.z_in, self.v = Lc, [], []
        self.h1 = model.register_forward_pre_hook(
            lambda m, args: self.z_in.append(args[0][:, self.Lc:].detach().clone()))
        self.h2 = model.register_forward_hook(lambda m, a, o: self.v.append(o.detach().clone()))

    def close(self) -> None:
        self.h1.remove()
        self.h2.remove()


def csv_frame(df: pd.DataFrame, path: Path) -> None:
    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    out["datetime"] = pd.DatetimeIndex(out["datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False, float_format=None, lineterminator="\n")   # floats use shortest repr


def run_case(up, idx, case_text, args, ctx, out_root: Path) -> dict:
    from kit_v1.infer import (apply_limit_projection, denormalize_paths, flow_sample,
                              paths_to_ohlcv, prepare_inference_window, summarize_paths)
    from kit_v1.data.calendar import bars_per_day

    model, cfg, stats, tdays, instruments = ctx
    code, scale, end, date_only = up.parse_case(case_text)
    Lc, Lh = cfg.data.window_for(scale)
    t0 = time.time()
    keep = Lc + int(math.ceil(12.0 * cfg.data.vol_ema_half_life_days * bars_per_day(MARKET, scale))) + 8
    hist, fut = up.load_case_frame(args.data_root, code, scale, end, Lc, Lh, keep)
    inst = instruments[code]
    win = prepare_inference_window(
        hist, MARKET, scale, Lc=Lc, Lh=Lh, stats=stats, soft_clip_c=cfg.data.soft_clip_c,
        half_life_days=cfg.data.vol_ema_half_life_days, end_ts=end,
        sector_id=int(inst.get("sector", 0)), inst_id=int(inst.get("vocab_index", 0)), trading_days=tdays)
    if win["n_pad"] != 0:
        raise SystemExit(f"{code} {scale}: padding not allowed")
    fut_matches = pd.DatetimeIndex(fut["datetime"]).equals(pd.DatetimeIndex(win["future_timestamps"]))
    t_prep = time.time() - t0

    seed = args.seed + idx
    tag = f"{code}_{scale}_{end.strftime('%Y%m%d')}" + ("" if end.hour == 23 else end.strftime("_%H%M"))
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    rec = StepRecorder(model, Lc)
    paths = flow_sample(model, win["x_ctx"], win["cond"], Lh=Lh, steps=args.steps, solver=args.solver,
                        n_paths=args.paths, cfg_w=1.0, hist_cfg_w=1.0,
                        generator=torch.Generator().manual_seed(seed), device=torch.device("cpu"),
                        chunk_size=16, precision="fp32")
    rec.close()
    t_sample = time.time() - t1

    K = args.paths
    noise = torch.randn(K, Lh, 5, generator=torch.Generator().manual_seed(seed))
    z_in = torch.stack(rec.z_in)          # [n_calls, K, Lh, 5]; one call per step for euler
    v = torch.stack(rec.v)
    n_calls = z_in.shape[0]
    assert args.solver != "euler" or n_calls == args.steps, n_calls
    assert torch.equal(z_in[0], noise), "recorded step-0 input differs from replayed noise"
    replay_ok = None
    if args.solver == "euler":
        grid = torch.linspace(1.0, 0.0, args.steps + 1)
        z = noise
        for i in range(args.steps):
            z = z - (float(grid[i]) - float(grid[i + 1])) * v[i]
            if i + 1 < args.steps:
                assert torch.equal(z, z_in[i + 1])
        replay_ok = bool(torch.equal(z.view(1, K, Lh, 5), paths))

    feats = denormalize_paths(paths, MARKET, scale, stats, c=cfg.data.soft_clip_c)[0]
    ohlcv = paths_to_ohlcv(feats, win["prev_close"], win["prev_ema"], win["half_life_bars"])
    limited = apply_limit_projection(ohlcv, win["prev_close"], 0.1)

    # files
    wt = {"x_ctx": win["x_ctx"], **{f"cond.{k}": v_ for k, v_ in win["cond"].items()}}
    write_safetensors(out / "window.safetensors", wt)
    write_safetensors(out / "noise.safetensors", {"noise": noise})
    t_grid = torch.linspace(1.0, 0.0, args.steps + 1)
    vel = {"v_step0": v[0], "v_last": v[-1], "z_in_last": z_in[-1], "z_final": paths[0],
           "t_step0": t_grid[0].reshape(()), "t_last": t_grid[args.steps - 1].reshape(())}
    for n in (1, 16):               # z after n Euler steps = model input of step n
        if n < n_calls:
            vel[f"z_after_step{n}"] = z_in[n]
    write_safetensors(out / "velocity_first.safetensors", vel)
    write_safetensors(out / "steps.safetensors", {"z_in": z_in, "v": v})
    write_safetensors(out / "paths.safetensors", {"paths": paths})
    write_safetensors(out / "feats.safetensors", {"feats": torch.from_numpy(feats)})
    flat = {k: torch.from_numpy(a) for k, a in ohlcv.items()}
    flat.update({f"limit0.1.{k}": torch.from_numpy(a) for k, a in limited.items()})
    write_safetensors(out / "ohlcv.safetensors", flat)
    write_json(out / "summary.json", {
        "prev_close": win["prev_close"], "ohlcv_prev_close": summarize_paths(ohlcv, win["prev_close"]),
        "feats": summarize_paths(feats), "limit0.1_prev_close": summarize_paths(limited, win["prev_close"])})
    csv_frame(hist, out / "bars.csv")
    csv_frame(fut, out / "future.csv")
    write_json(out / "inputs.json", {
        "case": case_text, "code": code, "scale": scale, "market": MARKET, "anchor_end": fmt_ts(end),
        "date_only": date_only, "Lc": Lc, "Lh": Lh, "keep": keep, "n_hist_rows": len(hist), "n_future_rows": len(fut),
        "sector_id": int(inst.get("sector", 0)), "inst_id": int(inst.get("vocab_index", 0)),
        "soft_clip_c": cfg.data.soft_clip_c, "half_life_days": cfg.data.vol_ema_half_life_days,
        "prev_close": win["prev_close"], "prev_ema": win["prev_ema"], "half_life_bars": win["half_life_bars"],
        "n_pad": win["n_pad"], "ctx_timestamps": fmt_ts_list(win["ctx_timestamps"]),
        "future_timestamps": fmt_ts_list(win["future_timestamps"]),
        "real_future_matches_calendar": bool(fut_matches),
        "paths": K, "steps": args.steps, "solver": args.solver, "cfg_w": 1.0, "hist_cfg_w": 1.0,
        "seed": seed, "case_index": idx, "precision": "fp32", "device": "cpu", "chunk": 16,
        "n_model_calls": n_calls, "euler_replay_exact": replay_ok,
        "seconds": {"prepare": round(t_prep, 2), "sample": round(t_sample, 2)},
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "tensors": {"window.safetensors": "x_ctx f32 [1,Lc,5]; cond.* as produced by prepare_inference_window",
                    "noise.safetensors": "noise f32 [K,Lh,5]",
                    "steps.safetensors": "z_in, v f32 [steps,K,Lh,5]",
                    "paths.safetensors": "paths f32 [1,K,Lh,5]",
                    "feats.safetensors": "feats f64 [K,Lh,5]",
                    "ohlcv.safetensors": "open/high/low/close/volume (+ limit0.1.*) f64 [K,Lh]"},
    }, indent=1)
    return {"tag": tag, "scale": scale, "Lc": Lc, "Lh": Lh, "tokens": Lc + Lh + cfg.model.n_registers,
            "prepare_s": t_prep, "sample_s": t_sample, "match": fut_matches, "replay_ok": replay_ok,
            "finite": bool(torch.isfinite(paths).all())}


def select_cases(up, spec: str) -> list:
    """Map a comma list of scales (or case indices) to [(index, case_text)] keeping upstream indices."""
    if spec == "all":
        return list(enumerate(up.DEFAULT_CASES))
    picked = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok.isdigit():
            picked.append((int(tok), up.DEFAULT_CASES[int(tok)]))
        elif tok in SCALES:
            i = next(i for i, c in enumerate(up.DEFAULT_CASES) if c.split(",")[1] == tok)
            picked.append((i, up.DEFAULT_CASES[i]))
        else:
            raise SystemExit(f"unknown case selector {tok!r}; use scales {SCALES}, indices 0-6, or 'all'")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="all", help="comma list of scales/indices of the 7 default cases, or 'all'")
    ap.add_argument("--paths", type=int, default=2)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--solver", choices=("euler", "heun"), default="euler")
    ap.add_argument("--seed", type=int, default=0, help="case i uses seed+i (i = index in the upstream default list)")
    ap.add_argument("--out", type=Path, default=ROOT / "tools" / "cache" / "reference")
    ap.add_argument("--data-root", type=Path, default=ROOT / "references" / "data")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "references" / "ckpt" / "step_market_284M.pt")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    up = load_upstream_script()
    from kit_v1.data.calendar import load_trading_days
    from kit_v1.data.normalize import load_stats
    from kit_v1.infer import load_model_and_config

    cases = select_cases(up, args.cases)
    instruments = json.loads((args.data_root / "instruments.json").read_text(encoding="utf-8"))
    stats = load_stats(args.data_root / "stats.json")
    tdays = load_trading_days(args.data_root / "trade_dates.csv")
    if tdays is None:
        raise SystemExit("missing trade_dates.csv")
    t0 = time.time()
    model, cfg = load_model_and_config(str(args.ckpt), "tiny", args.seed)
    model.to("cpu").eval()
    print(f"model loaded in {time.time() - t0:.1f}s; torch {torch.__version__} threads={torch.get_num_threads()} "
          f"K={args.paths} steps={args.steps} solver={args.solver} ({platform.processor()})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, text in cases:
        r = run_case(up, idx, text, args, (model, cfg, stats, tdays, instruments), args.out)
        rows.append(r)
        print(f"{r['tag']:28s} scale={r['scale']:3s} tokens={r['tokens']:5d} prepare={r['prepare_s']:6.1f}s "
              f"sample={r['sample_s']:8.1f}s fut_calendar_match={r['match']} euler_replay={r['replay_ok']} "
              f"finite={r['finite']}", flush=True)
    print("total sample seconds:", round(sum(r["sample_s"] for r in rows), 1))


if __name__ == "__main__":
    main()
