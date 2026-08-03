#!/usr/bin/env python3
"""Run DeepLSD GGUF parity matrix and write JSON + comparison assets.

Reference layout: LightGlue-GGML assets/ggml_validation_YYYYMMDD/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEEPLSD_ROOT = ROOT.parent / "dl" / "DeepLSD"
DEFAULT_RELEASE = ROOT / "core" / "AICore" / "models" / "release"


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


def parse_deeplsd(stdout: str, stderr: str) -> dict:
    out: dict = {"pass": "PASS" in stdout}
    blob = stdout + stderr
    for prefix in ("df_norm:", "angle:"):
        for line in blob.splitlines():
            if line.startswith(prefix):
                for token in line.split():
                    if "_abs=" in token or token.startswith("median="):
                        k, v = token.split("=", 1)
                        out[f"{prefix[:-1]}_{k}"] = float(v)
    return out


def file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--devices", default="cpu,cuda,vulkan")
    parser.add_argument("--quants", default="f32,f16,q8_0")
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            str(
                Path.home()
                / ".cursor/projects/home-ludahai-develop-code-github-ACloudViewer/assets/example-65c22ba6-267e-47c8-8747-d75c3a249d6e.png"
            ),
        ],
    )
    args = parser.parse_args()

    stamp = date.today().strftime("%Y%m%d")
    out_dir = args.output_dir or (ROOT / f"core/AICore/assets/ggml_validation_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ggml_lib = ":".join(
        str(p)
        for p in (DEEPLSD_ROOT / "cpp/build/_deps/ggml-build/src",)
        if p.is_dir()
    )
    env = os.environ.copy()
    if ggml_lib:
        env["LD_LIBRARY_PATH"] = f"{ggml_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    rows: list[dict] = []
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    deeplsd_stems = ("deeplsd_wireframe", "deeplsd_md")
    specs: list[tuple] = []
    for stem in deeplsd_stems:
        specs.append(
            (
                "deeplsd",
                stem,
                DEEPLSD_ROOT / "scripts/verify_deeplsd_ggml.py",
                512,
                DEEPLSD_ROOT / "cpp/build/deeplsd_extract",
                DEEPLSD_ROOT / "weights" / f"{stem}.tar",
            )
        )

    for model, stem, script, resize, binary, checkpoint in specs:
        for quant in quants:
            gguf = args.release_dir / "DeepLSD" / f"{stem}-{quant}.gguf"
            if not gguf.is_file():
                print(f"[SKIP] missing {gguf}")
                continue
            for device in devices:
                for image in args.images:
                    img_path = Path(image)
                    cmd = [
                        sys.executable,
                        str(script),
                        "--gguf",
                        str(gguf),
                        "--image",
                        str(img_path),
                        "--device",
                        device,
                        "--binary",
                        str(binary),
                        "--resize",
                        str(resize),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                    t0 = time.perf_counter()
                    proc = run(cmd, env=env)
                    elapsed = time.perf_counter() - t0
                    parsed = parse_deeplsd(proc.stdout, proc.stderr)
                    row = {
                        "model": model,
                        "variant": stem,
                        "quant": quant,
                        "device": device,
                        "image": img_path.name,
                        "gguf_mb": round(file_mb(gguf), 2),
                        "exit_code": proc.returncode,
                        "verify_s": round(elapsed, 3),
                        **parsed,
                    }
                    rows.append(row)
                    status = "PASS" if proc.returncode == 0 else "FAIL"
                    print(f"[{status}] {model}/{stem} {quant} {device} {img_path.name}")

    report = out_dir / "validation_matrix_deeplsd.json"
    report.write_text(json.dumps(rows, indent=2))

    variant_stems = sorted({r.get("variant", "deeplsd_wireframe") for r in rows})
    for stem in variant_stems:
        sizes = {}
        for quant in quants:
            p = args.release_dir / "DeepLSD" / f"{stem}-{quant}.gguf"
            if p.is_file():
                sizes[quant] = file_mb(p)
        if not sizes:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5))
        labels = list(sizes.keys())
        ax.bar(labels, [sizes[k] for k in labels], color="#2a6f97")
        ax.set_ylabel("GGUF size (MB)")
        ax.set_title(f"{stem} quantized sizes")
        fig.tight_layout()
        suffix = "" if len(variant_stems) == 1 else f"_{stem.replace('deeplsd_', '')}"
        fig.savefig(out_dir / f"quantization_sizes{suffix}.png", dpi=160)
        plt.close(fig)

    for stem in variant_stems:
        ref_rows = [
            r for r in rows
            if r["model"] == "deeplsd"
            and r.get("variant") == stem
            and r["device"] == "cpu"
        ]
        if not ref_rows:
            continue
        order = {q: i for i, q in enumerate(quants)}
        ref_rows.sort(key=lambda r: order.get(r["quant"], 99))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(ref_rows))
        w = 0.35
        ax.bar(x - w / 2, [r.get("df_norm_p99_abs", 0) for r in ref_rows], w,
               color="#2a6f97", label="df p99")
        ax.bar(x + w / 2, [r.get("angle_p99_abs", 0) for r in ref_rows], w,
               color="#e76f51", label="angle p99")
        ax.set_xticks(x, [r["quant"] for r in ref_rows])
        ax.axhline(0.05, color="#555", linewidth=1, label="tolerance")
        ax.set_title(f"{stem} p99 vs PyTorch (CPU)")
        ax.set_ylabel("p99 abs error")
        ax.legend()
        fig.tight_layout()
        suffix = "" if len(variant_stems) == 1 else f"_{stem.replace('deeplsd_', '')}"
        fig.savefig(out_dir / f"quantization_accuracy{suffix}.png", dpi=160)
        plt.close(fig)

    readme = out_dir / "README.md"
    readme.write_text(
        f"# GGML validation ({stamp}) — deeplsd\n\n"
        f"Matrix: `{report.name}`\n\n"
        "Regenerate: `python core/AICore/scripts/run_ggml_validation_matrix.py`\n"
    )
    print("Wrote", report, readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
