"""Shared helpers for the parity tools (upstream path setup, JSON / safetensors writers)."""
from __future__ import annotations

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid duplicate libiomp abort on Windows
sys.dont_write_bytecode = True

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = ROOT / "references" / "KiT"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

TS_FMT = "%Y-%m-%d %H:%M:%S"


def fmt_ts(ts) -> str:
    """Timestamp -> 'YYYY-MM-DD HH:MM:SS'."""
    return pd.Timestamp(ts).strftime(TS_FMT)


def fmt_ts_list(idx) -> list:
    return [fmt_ts(t) for t in pd.DatetimeIndex(idx)]


def to_py(obj: Any) -> Any:
    """Recursively convert tensors / arrays / numpy scalars to JSON-ready Python values.

    Floating arrays go through float64 (exact for float32), so repr round-trips.
    """
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f":
            obj = obj.astype(np.float64)
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return fmt_ts(obj) if isinstance(obj, pd.Timestamp) else fmt_ts_list(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any, indent=None) -> int:
    """Write JSON (NaN/Inf rejected); returns the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seps = None if indent else (",", ":")
    path.write_text(json.dumps(to_py(obj), indent=indent, separators=seps,
                               allow_nan=False, ensure_ascii=True) + "\n", encoding="utf-8")
    return path.stat().st_size


def write_safetensors(path: Path, tensors: Dict[str, torch.Tensor], metadata=None) -> int:
    """Write tensors (sorted by key for determinism); returns the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: tensors[k].detach().cpu().contiguous().clone() for k in sorted(tensors)}
    save_file(out, str(path), metadata=metadata or {})
    return path.stat().st_size
