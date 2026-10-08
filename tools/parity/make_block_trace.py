"""Record the token stream after every DiT block for one real-checkpoint case (fp32 and fp64).

    references/.venv/Scripts/python.exe tools/parity/make_block_trace.py 5m

Writes tools/cache/reference/<case>/blocks.safetensors with `fp32.<i>` and `fp64.<i>`
(`i` = 0 is the input token stream, `i` = n is after block n), each [K, n_reg + L, d].
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
model, _ = load_model_and_config(str(ROOT / "references/ckpt/step_market_284M.pt"), "tiny", 0)
model.eval()
win = load_file(case / "window.safetensors")
noise = load_file(case / "noise.safetensors")["noise"]
K, Lh, _ = noise.shape
cond = {k[5:]: v.repeat_interleave(K, dim=0) if v.shape[0] == 1 else v for k, v in win.items() if k.startswith("cond.")}
z = torch.cat([win["x_ctx"].repeat_interleave(K, dim=0), noise], dim=1)

out = {}


def trace(tag, m, z, t, cond):
    taps = []
    hooks = [b.register_forward_hook(lambda mod, inp, res: taps.append(res.detach().clone())) for b in m.blocks]
    first = m.blocks[0].register_forward_pre_hook(lambda mod, inp: taps.append(inp[0].detach().clone()))
    with torch.no_grad():
        m(z, t, cond, Lh)
    for h in hooks + [first]:
        h.remove()
    taps = [taps[-0 - len(m.blocks) - 1]] + taps[: len(m.blocks)] if False else taps
    for i, x in enumerate(taps):
        out[f"{tag}.{i}"] = x.double().contiguous() if tag == "fp64" else x.contiguous()


trace("fp32", model, z, torch.ones(K), cond)
model.double()
cond64 = {k: (v.double() if v.is_floating_point() else v) for k, v in cond.items()}
trace("fp64", model, z.double(), torch.ones(K, dtype=torch.float64), cond64)
save_file(out, str(case / "blocks.safetensors"))
print("saved", len(out), "tensors")
