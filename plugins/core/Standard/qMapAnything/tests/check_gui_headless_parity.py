#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""GUI/headless closure check for qMapAnything.

The GUI worker and -MAP_ANYTHING_RECONSTRUCT share one driver
(MapAnythingReconstruction), so on the same images, model, device and
frames-per-pass budget their merged clouds must be byte-identical. This
script runs both on the same view set:

  GUI      map_anything_worker_parity_harness (the real MapAnythingWorker,
           Settings seeded like MapAnythingDialog)
  headless ACloudViewer -SILENT -MAP_ANYTHING_RECONSTRUCT

for every (model, budget) pair. Budget 64 is the dialog default (headless
default); budget 16 is the dialog "responsive" checkbox (headless
MAX_FRAMES_PER_PASS 16). A row passes when both exit 0, the PLYs are
byte-identical with > 0 points, retains at least --min-retention-ratio of the
pre-confidence cloud, and every streamed chunk was aligned from a non-empty
finite Sim3. Per-process peak VRAM is sampled on CUDA so the single-pass vs
streamed memory behavior is measured, not assumed.

  python3 check_gui_headless_parity.py --build build_app --device cuda \
      --images-dir ~/cloudViewer_data/extract/map_anything_test_data/courtyard/images \
      --models vggt-omega-1b-512-q8_0.gguf mapanything-q8_0.gguf \
      --out build_app/Testing/map_anything_gui_headless

Exit 0 = all rows pass, 1 = a row failed, 77 = missing assets.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import threading
import time

# Dialog default single-pass budget and the "responsive" checkbox budget
# (MapAnythingWorker: responsiveStreaming ? 16 : maxFramesPerPass=64).
GUI_DEFAULT_BUDGET = 64
GUI_RESPONSIVE_BUDGET = 16


def qt_lib_dir(build: pathlib.Path):
    """Qt runtime dir the build links against, from its CMakeCache."""
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        for key in ("Qt5Core_DIR:", "Qt6Core_DIR:"):
            if line.startswith(key):
                # <prefix>/lib/cmake/Qt5Core -> <prefix>/lib
                return str(pathlib.Path(line.split("=", 1)[1]).parents[1])
    return None


class VramSampler:
    """Peak per-PID GPU memory (MiB) via nvidia-smi, polled every 100 ms."""

    def __init__(self, pid: int):
        self.pid = pid
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                for row in out.splitlines():
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) == 2 and parts[0] == str(self.pid):
                        self.peak = max(self.peak, int(parts[1]))
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self._stop.wait(0.1)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def run(cmd, env, log_path: pathlib.Path, sample_vram: bool):
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd,
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=log,
                                text=True)
        peak = None
        if sample_vram:
            with VramSampler(proc.pid) as s:
                stdout, _ = proc.communicate()
            peak = s.peak
        else:
            stdout, _ = proc.communicate()
        log.write(stdout)
    return proc.returncode, stdout, (time.perf_counter() - t0) * 1e3, peak


def sha256(path: pathlib.Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ply_vertex_count(path: pathlib.Path):
    with open(path, "rb") as f:
        for _ in range(16):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
    return -1


def scene_meta(images):
    """MapAnythingSceneMeta::discover (the dialog default): the WAI
    <scene>/scene_meta.json when every image sits in one <scene>/images."""
    dirs = {pathlib.Path(os.path.abspath(p)).parent for p in images}
    if len(dirs) != 1:
        return None
    d = dirs.pop()
    meta = d.parent / "scene_meta.json"
    return meta if d.name == "images" and meta.is_file() else None


def check_row(model_path, budget, images, args, env, out_dir):
    tag = f"{model_path.stem}-k{budget}"
    row_dir = out_dir / tag
    row_dir.mkdir(parents=True, exist_ok=True)
    cuda = args.device.lower() == "cuda"
    row = {"model": model_path.name, "budget": budget, "views": len(images)}
    failures = []

    gui_ply = row_dir / "gui.ply"
    gui_cmd = [
        str(args.harness), str(model_path), args.device,
        "1" if budget == GUI_RESPONSIVE_BUDGET else "0",
        str(gui_ply)
    ] + [str(p) for p in images]
    rc, stdout, ms, vram = run(gui_cmd, env, row_dir / "gui.log", cuda)
    row["gui"] = {"rc": rc, "wall_ms": round(ms, 1), "peak_vram_mib": vram}
    if rc == 77:
        return None
    if rc != 0:
        failures.append(f"GUI worker rc={rc}")
    else:
        try:
            last = json.loads(stdout.strip().splitlines()[-1])
            row["gui"]["points"] = last["points"]
            row["gui"]["frames_per_pass"] = last["result"].get(
                "frames_per_pass")
        except (IndexError, ValueError, KeyError):
            failures.append("GUI worker printed no result JSON")

    head_dir = row_dir / "headless"
    head_cmd = [
        str(args.acloudviewer), "-SILENT", "-NO_TIMESTAMP",
        "-MAP_ANYTHING_RECONSTRUCT"
    ]
    for p in images:
        head_cmd += ["IMAGE", str(p)]
    head_cmd += [
        "OUT_DIR",
        str(head_dir), "MODEL",
        str(model_path), "DEVICE", args.device, "CLOUD_PLY", "headless.ply",
        "RESULT_JSON", "result.json"
    ]
    if budget != GUI_DEFAULT_BUDGET:
        head_cmd += ["MAX_FRAMES_PER_PASS", str(budget)]
    # The GUI anchors to a discovered benchmark frame by default; the
    # headless equivalent is the explicit GT_META.
    meta = scene_meta(images)
    if meta is not None:
        head_cmd += ["GT_META", str(meta)]
        row["gt_meta"] = str(meta)
    rc, _, ms, vram = run(head_cmd, env, row_dir / "headless.log", cuda)
    row["headless"] = {"rc": rc, "wall_ms": round(ms, 1), "peak_vram_mib": vram}
    head_ply = head_dir / "headless.ply"
    result_json = head_dir / "result.json"
    if rc != 0 or not result_json.is_file():
        failures.append(f"headless rc={rc}")
    else:
        rep = json.loads(result_json.read_text())
        row["headless"]["points"] = rep.get("points")
        row["headless"]["filter"] = rep.get("filter")
        filt = row["headless"]["filter"] or {}
        input_points = filt.get("input", 0)
        kept_points = filt.get("kept", 0)
        retention = (kept_points / input_points) if input_points > 0 else 0.0
        row["headless"]["retention_ratio"] = retention
        if retention < args.min_retention_ratio:
            failures.append(
                f"implausibly sparse confidence result: kept/input "
                f"{kept_points}/{input_points}={retention:.6f} < "
                f"{args.min_retention_ratio:.6f}")
        if meta is not None:
            row["gt_sim3"] = rep.get("gt_sim3")
            if not rep.get("gt_sim3"):
                failures.append("GT_META given but no gt_sim3 in RESULT_JSON")
        row["chunks"] = rep.get("chunks", [])
        row["headless"]["frames_per_pass"] = rep.get("frames_per_pass")
        if rep.get("memory_downshifts"):
            row["headless"]["memory_downshifts"] = rep["memory_downshifts"]
        gui_k = row["gui"].get("frames_per_pass")
        if gui_k is not None and gui_k != rep.get("frames_per_pass"):
            failures.append(
                f"pass size differs: GUI {gui_k} vs headless "
                f"{rep.get('frames_per_pass')} (device memory admission "
                f"diverged between the two runs)")
        for i, c in enumerate(row["chunks"][1:], start=1):
            if not (c.get("correspondences", 0) > 0 and
                    math.isfinite(c.get("scale") or float("nan"))):
                failures.append(f"chunk {i} has no finite Sim3 alignment")

    if gui_ply.is_file() and head_ply.is_file():
        a, b = sha256(gui_ply), sha256(head_ply)
        row["gui_sha256"], row["headless_sha256"] = a, b
        n = ply_vertex_count(head_ply)
        row["points"] = n
        if a != b:
            failures.append("GUI and headless PLYs differ")
        if n <= 0:
            failures.append("empty cloud")
    elif not failures:
        failures.append("missing PLY output")

    row["failures"] = failures
    row["verdict"] = "PASS" if not failures else "FAIL"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", type=pathlib.Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--images-dir", type=pathlib.Path, required=True)
    ap.add_argument("--views",
                    type=int,
                    default=0,
                    help="first N sorted images (0 = all)")
    ap.add_argument("--models-dir",
                    type=pathlib.Path,
                    default=pathlib.Path.home() /
                    "cloudViewer_data/extract/map_anything_models")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["vggt-omega-1b-512-q8_0.gguf", "mapanything-q8_0.gguf"])
    ap.add_argument("--budgets",
                    type=int,
                    nargs="+",
                    default=[GUI_DEFAULT_BUDGET, GUI_RESPONSIVE_BUDGET])
    ap.add_argument(
        "--min-retention-ratio",
        type=float,
        default=0.01,
        help="minimum confidence-kept/input cloud ratio (default: 0.01)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    if not 0.0 < args.min_retention_ratio <= 1.0:
        ap.error("--min-retention-ratio must be in (0, 1]")
    for b in args.budgets:
        if b not in (GUI_DEFAULT_BUDGET, GUI_RESPONSIVE_BUDGET):
            ap.error(f"budget {b} has no GUI equivalent "
                     f"({GUI_DEFAULT_BUDGET} or {GUI_RESPONSIVE_BUDGET})")
    bin_dir = args.build / "bin"
    args.harness = bin_dir / "plugin_test" / "map_anything_worker_parity_harness"
    args.acloudviewer = bin_dir / "ACloudViewer"
    for exe in (args.harness, args.acloudviewer):
        if not exe.is_file():
            print(f"missing executable: {exe}", file=sys.stderr)
            return 77
    images = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.views > 0:
        images = images[:args.views]
    if len(images) < 2:
        print(f"need >= 2 images in {args.images_dir}", file=sys.stderr)
        return 77

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt = qt_lib_dir(args.build.resolve())
    libs = [str((bin_dir).resolve())] + ([qt] if qt else [])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        libs + [env.get("LD_LIBRARY_PATH", "")])

    args.out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for name in args.models:
        model = args.models_dir / name
        if not model.is_file():
            missing.append(name)
            continue
        for budget in args.budgets:
            row = check_row(model, budget, images, args, env, args.out)
            if row is None:
                missing.append(name)
                break
            rows.append(row)
            g, h = row["gui"], row["headless"]
            print(f"[{row['verdict']}] {name} K={budget} views={len(images)} "
                  f"points={row.get('points')} chunks={len(row.get('chunks', []))} "
                  f"K_eff={row['headless'].get('frames_per_pass')} "
                  f"gui={g['wall_ms']:.0f}ms/{g['peak_vram_mib']}MiB "
                  f"headless={h['wall_ms']:.0f}ms/{h['peak_vram_mib']}MiB "
                  + ("; ".join(row["failures"]) if row["failures"] else ""),
                  flush=True)

    report = {
        "device": args.device,
        "images_dir": str(args.images_dir),
        "views": len(images),
        "min_retention_ratio": args.min_retention_ratio,
        "rows": rows,
        "missing_models": missing,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    if missing:
        print(f"missing models: {missing}", file=sys.stderr)
    if any(r["verdict"] != "PASS" for r in rows):
        return 1
    return 77 if missing or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
