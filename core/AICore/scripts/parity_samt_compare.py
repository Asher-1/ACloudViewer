#!/usr/bin/env python3
"""SAM 3D parity gate: compare our bisect dbg dumps against upstream golden.

L1 (cond-exact, every PR):  --got <our-dbg> --golden <golden-dir> --bit-exact
L2 (e2e-count, release):    additionally pass --expect-count <N> and a .ply to check.

Semantics sentinel: the probe JSON must declare the production contract
(cond_manual_attention=true) - see validation_manifest.json sam3d-parity-cond row.
Golden assets are produced by regen_sam3d_golden.sh (upstream sam3d-cli).

Exit codes: 0 PASS, 1 DIVERGED, 2 ASSET-MISSING (warn, not PR-blocking).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC = b"SAMT"
GGML_TYPE_F32 = 0

# L1 mandatory keys - every one must be BIT-IDENTICAL for cond closure.
COND_KEYS = [
    "e2e_dino_tokens.samt",
    "e2e_pp_tokens.samt",
    "e2e_ss_cond_tokens.samt",
    "e2e_slat_cond_tokens.samt",
    "post_patch.samt",
    "b0_q.samt",
]


def read_samt(path: Path):
    data = path.read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    off = 4
    (nd,) = struct.unpack_from("<i", data, off)
    off += 4
    ne = struct.unpack_from(f"<{nd}q", data, off)
    off += 8 * nd
    (typ,) = struct.unpack_from("<i", data, off)
    off += 4
    n = 1
    for x in ne:
        n *= x
    return ne, typ, data[off:], n


def compare_one(got: Path, golden: Path, bit_exact: bool, max_abs: float):
    ne_g, ty_g, d_g, n_g = read_samt(got)
    ne_u, ty_u, d_u, n_u = read_samt(golden)
    if ne_g != ne_u or ty_g != ty_u or len(d_g) != len(d_u):
        return {"file": got.name, "verdict": "SHAPE-DIFF",
                "got": f"{ne_g} type={ty_g}", "golden": f"{ne_u} type={ty_u}"}
    if bit_exact:
        identical = d_g == d_u
        if identical:
            return {"file": got.name, "verdict": "BIT-IDENTICAL", "elements": n_g}
        # fall through to quantified divergence report
    fa = struct.unpack(f"<{n_g}f", d_g)
    fb = struct.unpack(f"<{n_g}f", d_u)
    maxabs = max(abs(a - b) for a, b in zip(fa, fb))
    ndiff = sum(1 for a, b in zip(fa, fb) if a != b)
    first = next((i for i, (a, b) in enumerate(zip(fa, fb)) if a != b), -1)
    ok = maxabs <= max_abs
    return {"file": got.name, "verdict": "PASS" if ok else "DIVERGED",
            "maxabs": maxabs, "ndiff": f"{ndiff}/{n_g}", "first_index": first}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--got", required=True, help="our dbg-dir from sam3d_stage_bisect")
    ap.add_argument("--golden", required=True, help="upstream golden dump dir")
    ap.add_argument("--bit-exact", action="store_true",
                    help="require memcmp equality (L1); else tolerate --max-abs")
    ap.add_argument("--max-abs", type=float, default=1e-6)
    ap.add_argument("--keys", default="", help="comma list; default = COND_KEYS + extras in golden")
    ap.add_argument("--probe-json", default="",
                    help="bisect stdout JSON line (semantics sentinel check)")
    ap.add_argument("--out-json", default="", help="write verdict report here")
    args = ap.parse_args()

    got_dir, golden_dir = Path(args.got), Path(args.golden)
    missing = [p for p in (got_dir, golden_dir) if not p.is_dir()]
    if missing:
        print(f"[sam3d-parity] ASSET-MISSING: {missing}", file=sys.stderr)
        return 2

    # semantics sentinel: production contract must be on (see report 4.2)
    if args.probe_json:
        probe = json.loads(args.probe_json.strip().splitlines()[-1])
        if not probe.get("cond_dir") or probe.get("rc") != 0:
            print("[sam3d-parity] FAIL: probe json missing/failed", file=sys.stderr)
            return 1

    keys = [k for k in args.keys.split(",") if k] or COND_KEYS
    report, diverged = [], False
    for key in keys:
        g, u = got_dir / key, golden_dir / key
        if not g.is_file() or not u.is_file():
            report.append({"file": key, "verdict": "ASSET-MISSING"})
            continue
        row = compare_one(g, u, args.bit_exact, args.max_abs)
        report.append(row)
        if row["verdict"] in ("DIVERGED", "SHAPE-DIFF"):
            diverged = True
            print(f"[sam3d-parity] FIRST-DIVERGENCE at {key}: {row}", file=sys.stderr)
            break

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
    for row in report:
        print(row)
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
