"""Intermediate tensors inside block 0 for one real-checkpoint case (torch fp32 and fp64).

    references/.venv/Scripts/python.exe tools/parity/make_op_trace.py 5m

Writes tools/cache/reference/<case>/ops.safetensors with `<dtype>.<op>` entries.
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


def run(tag, m, z, t, cond):
    blk = m.blocks[0]
    taps = {}

    def grab(name):
        def hook(mod, inp, res):
            taps[name] = res.detach().clone()
        return hook

    hooks = [
        blk.norm1.register_forward_hook(grab("norm1")),
        blk.attn.qkv.register_forward_hook(grab("qkv")),
        blk.attn.q_norm.register_forward_hook(grab("q_norm")),
        blk.attn.out_proj.register_forward_hook(grab("attn_out_proj")),
        blk.attn.register_forward_hook(grab("attn")),
        blk.norm2.register_forward_hook(grab("norm2")),
        blk.mlp.w1.register_forward_hook(grab("mlp_w1")),
        blk.mlp.register_forward_hook(grab("mlp")),
        blk.register_forward_hook(grab("block")),
    ]
    with torch.no_grad():
        m(z, t, cond, Lh)
    for h in hooks:
        h.remove()
    for name, x in taps.items():
        out[f"{tag}.{name}"] = x.double().contiguous() if tag == "fp64" else x.contiguous()


run("fp32", model, z, torch.ones(K), cond)
model.double()
cond64 = {k: (v.double() if v.is_floating_point() else v) for k, v in cond.items()}
run("fp64", model, z.double(), torch.ones(K, dtype=torch.float64), cond64)
save_file(out, str(case / "ops.safetensors"))
print({k: tuple(v.shape) for k, v in out.items() if k.startswith("fp32")})
