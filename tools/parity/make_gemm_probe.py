"""One large linear layer of the real checkpoint: torch fp32 vs fp64 on identical inputs.

    references/.venv/Scripts/python.exe tools/parity/make_gemm_probe.py 5m

Writes tools/cache/reference/<case>/gemm.safetensors with `x`, `y32`, `y64` for
`blocks.0.attn.qkv` (input = torch's own fp32 `norm1` output, taken from ops.safetensors).
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
scale = sys.argv[1]
case = next(p for p in (ROOT / "tools/cache/reference").iterdir() if p.name.split("_")[1] == scale)
ops = load_file(case / "ops.safetensors")
x = ops["fp32.norm1"]
sd = torch.load(ROOT / "references/ckpt/step_market_284M.pt", map_location="cpu", weights_only=False)["ema"]["shadow"]
w = sd["blocks.0.attn.qkv.weight"]
y32 = torch.nn.functional.linear(x, w)
y64 = torch.nn.functional.linear(x.double(), w.double())
print("torch32 vs fp64 max abs:", float((y32.double() - y64).abs().max()), " max|y|:", float(y64.abs().max()))
save_file({"x": x.contiguous(), "y32": y32.contiguous(), "y64": y64.contiguous()}, str(case / "gemm.safetensors"))
