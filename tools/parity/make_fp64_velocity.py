"""Compute the first-step velocity of one real-checkpoint case in float64 with upstream code.

Used to decide whether a float32 difference between PyTorch and the Rust port is ordinary
rounding: if both float32 implementations sit about equally far from the float64 result, the
difference is rounding and not a logic error.

    references/.venv/Scripts/python.exe tools/parity/make_fp64_velocity.py 5m
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "references" / "KiT"))

import torch
from safetensors.torch import load_file, save_file

from kit_v1.infer import load_model_and_config

scale = sys.argv[1]
case = next(p for p in (ROOT / "tools/cache/reference").iterdir() if p.name.split("_")[1] == scale)
torch.set_num_threads(os.cpu_count())

model, cfg = load_model_and_config(str(ROOT / "references/ckpt/step_market_284M.pt"), "tiny", 0)
model.eval()
win = load_file(case / "window.safetensors")
noise = load_file(case / "noise.safetensors")["noise"]
first = load_file(case / "velocity_first.safetensors")

K, Lh, F = noise.shape
x_ctx = win["x_ctx"].repeat_interleave(K, dim=0)
cond = {k[5:]: v.repeat_interleave(K, dim=0) if v.shape[0] == 1 else v for k, v in win.items() if k.startswith("cond.")}
z = torch.cat([x_ctx, noise], dim=1)
t = torch.ones(K)

with torch.no_grad():
    v32 = model(z, t, cond, Lh)
    model.double()
    cond64 = {k: (v.double() if v.is_floating_point() else v) for k, v in cond.items()}
    v64 = model(z.double(), t.double(), cond64, Lh)

print("torch32 vs recorded v_step0:", float((v32 - first["v_step0"]).abs().max()))
print("torch32 vs fp64:", float((v32.double() - v64).abs().max()))
save_file({"v64": v64.contiguous(), "v32": v32.contiguous()}, str(case / "velocity_fp64.safetensors"))
