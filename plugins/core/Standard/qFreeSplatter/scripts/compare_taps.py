#!/usr/bin/env python3
"""Compare engine activation taps against PyTorch reference dumps, in topological
order, stopping at the first diverging tap (everything downstream of a real
divergence is noise).

Usage: compare_taps.py <reference_dir> <engine_dir> [--row-err 1e-2] [--cos 0.99999]
                       [--scale 1.0] [--keep-going]

Each dir holds a meta.json ({"taps": {name: {"shape":[r,c], "dtype":"f32"|"i32"}}})
plus one <name>.f32 / <name>.i32 raw little-endian file per tap.

Pass rule per f32 tap (BOTH must hold):
  1. per-row cosine >= --cos (primary gate; a real graph bug -- wrong transpose,
     swapped tensor, wrong mask -- craters this).
  2. per-row normalized error  max|a-b| / max(1, max|b|)  <= --row-err. Absolute
     tolerances do not fit: the residual-stream magnitude grows with depth, so a
     mid-layer absolute error tracks |row| while cosine and the final head stay
     clean. Normalizing by the row inf-norm bounds the error where it is
     meaningful and keeps discrimination for real bugs.

--scale multiplies the tolerances (e.g. 4 for Vulkan/fp16).
The PyTorch reference computes numerically-sensitive ops in float64 (hf_dump.py),
so the engine targets the exact value, not a second noisy float path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Topological order of the free-splatter graph (pieces 1-3). Per-block sub-order:
_BLOCK_SUB = [
    "attn_norm", "q", "k", "v", "attn_out", "attn_resid",
    "ffn_norm", "ffn_up", "gelu", "ffn_down", "l_out",
]
_PRE  = ["images", "patch_embed", "tokens_pos", "tokens_in"]
_POST = ["result_norm", "head_logits", "gaussians_raw", "gaussians"]


def load(d: Path, name: str, meta) -> np.ndarray:
    t = meta["taps"][name]
    a = np.fromfile(d / f"{name}.{t['dtype']}", dtype=f"<{t['dtype'][0]}4")
    return a.reshape(t["shape"])


def tap_order(names):
    def key(n):
        if n in _PRE:
            return (-1, _PRE.index(n), 0)
        m = re.match(r"l(\d+)\.(.+)", n)
        if m:
            sub = m.group(2)
            return (int(m.group(1)), 1, _BLOCK_SUB.index(sub) if sub in _BLOCK_SUB else 99)
        if n in _POST:
            return (1 << 30, _POST.index(n), 0)
        return (1 << 30, 99, 0)
    return sorted(names, key=key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("engine_dir", type=Path)
    ap.add_argument("--row-err", type=float, default=1e-2)
    ap.add_argument("--cos", type=float, default=0.99999)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    ref_meta = json.loads((args.ref_dir / "meta.json").read_text())
    en_meta = json.loads((args.engine_dir / "meta.json").read_text())
    common = [n for n in tap_order(ref_meta["taps"]) if n in en_meta["taps"]]
    missing = [n for n in ref_meta["taps"] if n not in en_meta["taps"]]
    if missing:
        print(f"note: engine did not dump: {missing}")

    print(f"{'tap':24s} {'max_abs':>10s} {'row_err':>10s} {'min_cos':>9s}  verdict")
    failed = None
    for name in common:
        ref = load(args.ref_dir, name, ref_meta).astype(np.float64)
        en = load(args.engine_dir, name, en_meta).astype(np.float64)
        if ref.shape != en.shape:
            print(f"{name:24s} SHAPE MISMATCH ref{ref.shape} engine{en.shape}")
            failed = failed or name
            if not args.keep_going:
                break
            continue

        if ref.ndim == 1:
            ref = ref[None, :]
            en = en[None, :]
        diff = np.abs(ref - en)
        row_scale = np.maximum(np.abs(ref).max(axis=1), 1.0)
        row_err = (diff.max(axis=1) / row_scale).max()
        num = (ref * en).sum(1)
        den = np.linalg.norm(ref, axis=1) * np.linalg.norm(en, axis=1) + 1e-12
        cos = (num / den).min()
        ok = row_err <= args.row_err * args.scale and cos >= args.cos
        print(f"{name:24s} {diff.max():10.3e} {row_err:10.3e} {cos:9.6f}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failed = failed or name
            if not args.keep_going:
                break

    if failed:
        print(f"FIRST DIVERGENCE: {failed}")
        return 1
    print("ALL TAPS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
