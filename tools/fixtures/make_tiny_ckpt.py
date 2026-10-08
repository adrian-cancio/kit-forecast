#!/usr/bin/env python
"""Generate the tiny PyTorch checkpoint fixture tests/fixtures/tiny_ckpt.pt (dev-only, not needed at runtime).

Run from the repo root:
    references/.venv/Scripts/python.exe tools/fixtures/make_tiny_ckpt.py
    references/.venv/Scripts/python.exe tools/fixtures/make_tiny_ckpt.py --print-golden

The file is written with torch.save and has the same structure as the official checkpoint
(step / config{model,data,train,infer} / ema{decay,shadow} / world_size / accum / global_batch / slim),
with the real upstream KlineDiT state_dict names and shapes. The model is smaller than the official
one (d_model 16, 2 layers, 4 heads) so that the file stays below 100 KB; weights are random but fixed
by the seed. The optional --variant flag writes deliberately broken files (to --out) for manual tests:
  f16        the weights are float16
  no_ema     weights under `model` instead of `ema`
  no_yday    config.model lacks yday_dim
--print-golden prints the spot values asserted in src/checkpoint/tests.rs.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "references" / "KiT"))

import torch  # noqa: E402

from kit_v1.config import Config, DataConfig, InferConfig, ModelConfig  # noqa: E402
from kit_v1.model import KlineDiT  # noqa: E402

SEED = 20240607
STEP = 1234
EMA_DECAY = 0.999


def build():
    model_cfg = ModelConfig(
        d_model=16, n_layers=2, n_heads=4, ffn_hidden=24, inst_vocab=16, inst_dim=8,
        sess_hidden=16, t_emb_dim=16, market_slots=4, sector_slots=8, scale_slots=8,
        attn_backend="math",
    )
    data_cfg = DataConfig(
        Lc=48, Lh=8, trunc_min=16, stride=0,
        scale_windows="1m:32/8/16,5m:24/6,1d:20/4/10",
    )
    infer_cfg = InferConfig(steps=8, solver="heun", n_paths=4, cfg_w=1.5, hist_cfg_w=1.25)
    cfg = Config(model=model_cfg, data=data_cfg, infer=infer_cfg)
    gen = torch.Generator().manual_seed(SEED)
    model = KlineDiT(model_cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    return cfg, model


def checkpoint(cfg, state, variant):
    config = dataclasses.asdict(cfg)
    config["train"] = {}
    if variant == "no_yday":
        del config["model"]["yday_dim"]
    ck = {
        "step": STEP,
        "config": config,
        "world_size": 1,
        "accum": 1,
        "global_batch": 2,
        "slim": {"kind": "inference_only_ema", "dropped": ["optimizer"]},
    }
    if variant == "no_ema":
        ck["model"] = state
    else:
        ck["ema"] = {"decay": EMA_DECAY, "shadow": state}
    return ck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "tiny_ckpt.pt"))
    parser.add_argument("--variant", choices=["f16", "no_ema", "no_yday"])
    parser.add_argument("--print-golden", action="store_true")
    args = parser.parse_args()

    cfg, model = build()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if args.print_golden:
        for name in ("in_proj.weight", "cond_emb.inst.weight", "blocks.1.mlp.w2.weight", "out_proj.bias"):
            t = state[name]
            print(name, tuple(t.shape), t.flatten()[:4].tolist(), t.double().sum().item())
        print("tensors", len(state), "params", sum(v.numel() for v in state.values()))
        return
    if args.variant == "f16":
        state = {k: v.half() for k, v in state.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(cfg, state, args.variant), out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
