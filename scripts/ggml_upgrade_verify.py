#!/usr/bin/env python3
"""ggml upgrade one-click verification: baseline build vs candidate build.

Design (first principles): a "regression" is nothing more than
    probe -> metric -> threshold
evaluated on two build trees. Everything else (build system, ctest labels,
plugins) is ceremony around that comparison. So this script:

  1. Runs the same probe list against BASELINE and CANDIDATE build dirs.
  2. Parses each probe's output into {metric: value} (probe families:
     yolo-json / sam3-json / rfdetr-text / parity-text / exit-code).
  3. Compares candidate vs baseline with a per-metric regression threshold
     (default +5%: candidate slower by more than 5% => REGRESSION).
  4. Emits a Markdown report and exits non-zero on any regression/failure.

Probes whose binary or model asset is missing are reported as SKIPPED and
counted against coverage, never silently dropped.

Usage:
    python3 scripts/ggml_upgrade_verify.py \
        --baseline /path/to/build_app \
        --candidate /path/to/build-ggml021 \
        [--report /tmp/ggml_upgrade_report.md] [--threshold 5.0] \
        [--only yolo,rfdetr] [--device vulkan]

Environment (auto-derived unless overridden):
    AICORE_TEST_YOLO_MODELS_DIR   dir with yolo *.gguf
    AICORE_TEST_YOLO_IMAGE        test image
    AICORE_TEST_RMBG_GGUF         rmbg gguf
    AICORE_TEST_RFDETR_GGUF       rfdetr gguf (default: seg-nano-f16)
    AICORE_TEST_DEPTH_GGUF / _IMAGE / AICORE_TEST_DEVICE
    AICORE_TEST_ALIKED_GGUF / _IMAGE
    AICORE_TEST_LIGHTGLUE_GGUF / _IMAGE0 / _IMAGE1
    QT_LIB_DIR                    (default /opt/qt515/lib)
    CLOUDVIEWER_DATA_ROOT         (default ~/cloudViewer_data)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Probe manifest. Each probe: (name, family, rel_binary, extra_argv)
# family decides parsing. rel_binary is relative to <build>/bin or
# <build>/bin/aicore_tests (both are searched).
PROBES = [
    # AICore C-API performance benches (JSON-line emitters)
    ("yolo_vulkan", "yolo-json", "test_yolo_capi_performance", []),
    ("yolo_cpu", "yolo-json", "test_yolo_capi_performance", []),
    # whitebox manual bench (text emitter)
    ("rfdetr_vulkan", "rfdetr-text", "bench_rfdetr_perf", []),
    # CPU-vs-GPU numerical parity gates (must PASS on both trees)
    ("depth_parity", "parity-text", "test_depth_capi_backend_parity", []),
    ("aliked_parity", "exit-code", "test_aliked_capi_parity", []),
    ("aliked_parity_q8", "exit-code", "test_aliked_capi_parity_q8", []),
    ("lightglue_e2e", "exit-code", "test_lightglue_aliked_e2e", []),
    ("aliked_smoke", "exit-code", "test_aliked_e2e_smoke", []),
    # SAM3 end-to-end acceptance (JSON single line; present only when the
    # bench tool has been added to the tree)
    ("sam3_acceptance", "sam3-json", "bench_sam3_backend_acceptance", []),
]

# Coverage universe: (task, backend) pairs that a full upgrade verification
# must say something about. Derived from core/AICore/src/tasks/* and the
# GPU-capable subset; CPU-only tasks count once.
COVERAGE_UNIVERSE = [
    ("depth", "cpu"), ("depth", "vulkan"),
    ("gaussian", "cpu"), ("gaussian", "vulkan"),
    ("aliked", "cpu"), ("aliked", "vulkan"),
    ("lightglue", "cpu"), ("lightglue", "vulkan"),
    ("deeplsd", "cpu"), ("deeplsd", "vulkan"),
    ("facedetect", "cpu"),
    ("rfdetr", "cpu"), ("rfdetr", "vulkan"),
    ("rmbg", "cpu"), ("rmbg", "vulkan"),
    ("yolo", "cpu"), ("yolo", "vulkan"),
    ("sam3", "cpu"), ("sam3", "vulkan"),
    ("trellis", "cpu"), ("trellis", "vulkan"),
]


@dataclass
class ProbeResult:
    name: str
    status: str  # ok | regression | fail | skipped
    metrics: dict = field(default_factory=dict)
    note: str = ""
    duration_s: float = 0.0


def find_binary(build_dir: Path, rel: str) -> Path | None:
    for sub in ("bin/aicore_tests", "bin"):
        p = build_dir / sub / rel
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def run(cmd: list[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=timeout, check=False)


def parse_yolo_json(stdout: str) -> dict:
    """yolo perf emits one JSON per (model, backend); key by file+task."""
    out = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith('{"suite"'):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = f"{row['file']}:{row['task']}"
        out[f"{key}/graph_ms"] = row["graph_ms"]["mean"]
        out[f"{key}/e2e_ms"] = row["e2e_ms"]["mean"]
    return out


def parse_sam3_json(stdout: str) -> dict:
    """sam3 acceptance emits a final single-line JSON summary."""
    out = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"runs"'):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            for be in ("cpu", "vulkan"):
                blk = row.get(be) or {}
                for stage in ("load", "encode", "pvs", "track", "total"):
                    m = (blk.get(stage) or {}).get("median_ms")
                    if m:
                        out[f"{be}/{stage}_ms"] = m
                if blk.get("peak_vram_mib", -1) > 0:
                    out[f"{be}/peak_vram_mib"] = blk["peak_vram_mib"]
            out["mask_iou"] = row.get("mask_iou", 0.0)
            out["gates_passed"] = 1.0 if row.get("gates_passed") else 0.0
            break
    return out


def parse_rfdetr_text(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        m = re.match(r"\s+(forward \(A\+topK\+B\)|detect-total|detect-plugin-path)"
                     r"\s*:\s*([\d.]+)", line)
        if m:
            key = m.group(1).replace(" ", "_").replace("(", "").replace(")", "")
            out[key + "_ms"] = float(m.group(2))
    return out


def parse_parity_text(stdout: str) -> dict:
    """depth backend parity prints 'relative MAE x (limit y)' per output."""
    out, worst = {}, 0.0
    for m in re.finditer(r"relative MAE ([\d.eE+-]+) \(limit ([\d.eE+-]+)\)",
                         stdout):
        worst = max(worst, float(m.group(1)))
    if worst:
        out["worst_rel_mae"] = worst
    return out


PARSERS = {
    "yolo-json": parse_yolo_json,
    "sam3-json": parse_sam3_json,
    "rfdetr-text": parse_rfdetr_text,
    "parity-text": parse_parity_text,
}

def median_metrics(runs: list[dict]) -> dict:
    """Per-key median across repeated runs (single-run dict passes through)."""
    if len(runs) == 1:
        return runs[0]
    keys = set().union(*[r.keys() for r in runs])
    return {k: statistics.median(r[k] for r in runs if k in r)
            for k in sorted(keys)}


# Metrics where "higher is better" (IoU / gate flags) -> regression means drop.
HIGHER_IS_BETTER = {"mask_iou", "gates_passed"}
# Metrics that are absolute gates, not A/B comparisons.
ABSOLUTE_GATES = {"gates_passed": 1.0, "mask_iou": 0.995}
# Noise floor: ignore relative deltas below this (machine jitter).
NOISE_FLOOR_PCT = 3.0


def probe_env(args, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    data_root = Path(os.environ.get("CLOUDVIEWER_DATA_ROOT",
                                    Path.home() / "cloudViewer_data"))
    ext = data_root / "extract"
    env.setdefault("AICORE_TEST_YOLO_MODELS_DIR", str(ext / "yolo_models"))
    env.setdefault("AICORE_TEST_YOLO_IMAGE",
                   str(args.repo / "examples/test_data/image/00000.png"))
    env.setdefault("AICORE_TEST_RMBG_GGUF",
                   str(ext / "rmbg_models/rmbg_f16.gguf"))
    env.setdefault("AICORE_TEST_DEPTH_GGUF",
                   str(ext / "da3_models/depth-anything-base-f16.gguf"))
    env.setdefault("AICORE_TEST_DEPTH_IMAGE",
                   str(args.repo / "examples/test_data/image/00000.png"))
    env.setdefault("AICORE_TEST_ALIKED_GGUF",
                   str(ext / "lightglue_models/aliked-n16rot-f16.gguf"))
    env.setdefault("AICORE_TEST_ALIKED_IMAGE",
                   str(ext / "lightglue_test_images/sacre_coeur1.jpg"))
    env.setdefault("AICORE_TEST_LIGHTGLUE_GGUF",
                   str(ext / "lightglue_models/aliked-lightglue-f16.gguf"))
    env.setdefault("AICORE_TEST_LIGHTGLUE_IMAGE0",
                   str(ext / "lightglue_test_images/sacre_coeur1.jpg"))
    env.setdefault("AICORE_TEST_LIGHTGLUE_IMAGE1",
                   str(ext / "lightglue_test_images/sacre_coeur2.jpg"))
    env.setdefault("AICORE_TEST_DEVICE", args.device)
    env.setdefault("CLOUDVIEWER_DATA_ROOT", str(data_root))
    if extra:
        env.update(extra)
    return env


def build_ld_path(build_dir: Path, args) -> str:
    parts = [str(build_dir / "bin"), args.qt_lib]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def run_probe(name: str, family: str, rel: str, argv_extra: list[str],
              build_dir: Path, args) -> ProbeResult:
    t0 = time.time()
    binary = find_binary(build_dir, rel)
    if binary is None:
        return ProbeResult(name, "skipped",
                           note=f"binary not found: {rel}")

    env = probe_env(args, {"LD_LIBRARY_PATH": build_ld_path(build_dir, args)})
    cmd = [str(binary)] + argv_extra
    if name == "yolo_cpu":
        env["AICORE_TEST_YOLO_DEVICE"] = "cpu"
    elif name == "yolo_vulkan":
        env["AICORE_TEST_YOLO_DEVICE"] = args.device
        cmd = [str(binary)]  # argv[1] would be treated as a GGUF path
    if name == "rfdetr_vulkan":
        gguf = os.environ.get(
            "AICORE_TEST_RFDETR_GGUF",
            str(Path(env["CLOUDVIEWER_DATA_ROOT"]) /
                "extract/rfdetr_models/rfdetr-seg-nano-f16.gguf"))
        cmd = [str(binary), gguf, args.device, "4", "640", "640",
               str(args.iters)]
    if name == "sam3_acceptance":
        gguf = os.environ.get("AICORE_TEST_SAM3_GGUF", "")
        img = os.environ.get("AICORE_TEST_SAM3_IMAGE", "")
        if not gguf or not img:
            return ProbeResult(name, "skipped",
                               note="AICORE_TEST_SAM3_GGUF/IMAGE unset")
        cmd = [str(binary), gguf, img, "3", "all"]

    try:
        proc = run(cmd, env, args.timeout)
    except subprocess.TimeoutExpired:
        return ProbeResult(name, "fail", note=f"timeout {args.timeout}s",
                           duration_s=time.time() - t0)
    dur = time.time() - t0

    if proc.returncode == 77:
        return ProbeResult(name, "skipped", note="exit 77 (asset/backend)",
                           duration_s=dur)
    # ggml aborts and uncaught vk::*Error kill the process non-zero; capture
    # the last meaningful line as the note.
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        note = tail[-1][:160] if tail else f"exit {proc.returncode}"
        # yolo perf keeps printing rows until a crash; salvage them.
        if family == "yolo-json":
            m = PARSERS[family](proc.stdout + "\n" + proc.stderr)
            if m:
                return ProbeResult(name, "fail", metrics=m,
                                   note="crashed mid-suite: " + note,
                                   duration_s=dur)
        return ProbeResult(name, "fail", note=note, duration_s=dur)

    # Some tests report metrics on stderr (depth parity); merge both streams.
    combined = proc.stdout + "\n" + proc.stderr
    metrics = PARSERS[family](combined) if family in PARSERS else {}
    if family == "exit-code" and not metrics:
        metrics["exit_ok"] = 1.0
    if not metrics:
        return ProbeResult(name, "skipped",
                           note="ran but produced no parsable metrics",
                           duration_s=dur)
    return ProbeResult(name, "ok", metrics=metrics, duration_s=dur)


def merge_results(name: str, runs: list[ProbeResult]) -> ProbeResult:
    """Merge repeated rounds of one probe: per-metric median over all runs
    that produced metrics; status prefers ok > fail > skipped."""
    with_metrics = [r for r in runs if r.metrics]
    metrics = median_metrics([r.metrics for r in with_metrics]) \
        if with_metrics else {}
    if any(r.status == "ok" for r in runs):
        status = "ok"
    elif any(r.status == "fail" for r in runs):
        status = "fail"
    else:
        status = "skipped"
    first_notable = next((r for r in runs if r.status != "ok"), runs[0])
    return ProbeResult(name, status, metrics=metrics,
                       note=first_notable.note,
                       duration_s=sum(r.duration_s for r in runs))


def compare(baseline: ProbeResult, candidate: ProbeResult,
            threshold_pct: float) -> tuple[str, list[str]]:
    """Return (status, rows) for one probe pair."""
    rows = []
    if candidate.status == "skipped":
        return "skipped", [f"candidate skipped: {candidate.note}"]
    if candidate.status == "fail":
        return "fail", [f"candidate failed: {candidate.note}"]
    if baseline.status != "ok":
        rows.append(f"baseline {baseline.status}: {baseline.note} "
                    "(candidate measured, no A/B)")
        return "no-baseline", rows

    status = "ok"
    for key, cand in sorted(candidate.metrics.items()):
        base = baseline.metrics.get(key)
        if base is None:
            rows.append(f"{key}: candidate={cand:.4g} (no baseline)")
            continue
        if key in ABSOLUTE_GATES:
            floor = ABSOLUTE_GATES[key]
            okv = cand >= floor if key in HIGHER_IS_BETTER else cand <= floor
            rows.append(f"{key}: candidate={cand:.6g} gate={'PASS' if okv else 'FAIL'}")
            if not okv:
                status = "regression"
            continue
        if base == 0:
            rows.append(f"{key}: baseline 0, candidate={cand:.4g}")
            continue
        delta_pct = (cand - base) / abs(base) * 100.0
        if key in HIGHER_IS_BETTER:
            delta_pct = -delta_pct  # a drop is the regression direction
        flag = ""
        if delta_pct > threshold_pct and delta_pct > NOISE_FLOOR_PCT:
            flag = "  **REGRESSION**"
            status = "regression"
        elif delta_pct < -threshold_pct:
            flag = "  (improved)"
        rows.append(f"{key}: base={base:.4g} cand={cand:.4g} "
                    f"Δ={delta_pct:+.1f}%{flag}")
    return status, rows


def coverage(results: dict) -> tuple[float, list[str]]:
    """Map probe outcomes onto the (task, backend) universe."""
    covered, notes = set(), []
    probe_coverage = {
        "yolo_vulkan": [("yolo", "vulkan")],
        "yolo_cpu": [("yolo", "cpu")],
        "rfdetr_vulkan": [("rfdetr", "vulkan")],
        "depth_parity": [("depth", "cpu"), ("depth", "vulkan")],
        "aliked_parity": [("aliked", "cpu"), ("aliked", "vulkan")],
        "aliked_parity_q8": [("aliked", "vulkan")],
        "lightglue_e2e": [("lightglue", "vulkan")],
        "aliked_smoke": [("lightglue", "cpu")],
        "sam3_acceptance": [("sam3", "cpu"), ("sam3", "vulkan")],
    }
    for probe, pairs in probe_coverage.items():
        res = results.get(probe)
        if res and res.status == "ok":
            covered.update(pairs)
    # contract/parity ctest tiers cover the remaining tasks' CPU paths via
    # the capi label suite; count them if the caller ran ctest separately.
    pct = 100.0 * len(covered) / len(COVERAGE_UNIVERSE)
    missing = sorted(set(COVERAGE_UNIVERSE) - covered)
    for task, be in missing:
        notes.append(f"{task}/{be}")
    return pct, notes

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[1])
    ap.add_argument("--report", type=Path,
                    default=Path("/tmp/ggml_upgrade_report.md"))
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="regression threshold percent (default 5)")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--qt-lib", default="/opt/qt515/lib")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=2,
                    help="interleaved rounds per probe (baseline and "
                         "candidate measured back-to-back each round, "
                         "per-metric median across rounds; default 2, "
                         "rejects machine-load drift)")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--only", default="",
                    help="comma-separated probe names to run")
    args = ap.parse_args()

    # Echo resolved asset env so inherited-shell pollution is visible
    # (e.g. a stale AICORE_TEST_YOLO_MODELS_DIR pointing at a subset dir).
    resolved = probe_env(args)
    print("[env] asset selection (inherited values override defaults):",
          flush=True)
    for k in sorted(resolved):
        if k.startswith("AICORE_TEST_") or k == "CLOUDVIEWER_DATA_ROOT":
            src = "inherited" if k in os.environ else "default"
            print(f"[env]   {k}={resolved[k]}  ({src})", flush=True)

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    base_results: dict[str, ProbeResult] = {}
    cand_results: dict[str, ProbeResult] = {}

    for name, family, rel, extra in PROBES:
        if only and name not in only:
            continue
        # Interleaved rounds: baseline and candidate are measured
        # back-to-back each round so both share the same machine-load
        # window (background CPU load drifts by the minute).
        base_runs, cand_runs = [], []
        for rnd in range(1, args.repeats + 1):
            tag = f"round {rnd}/{args.repeats}" if args.repeats > 1 else ""
            print(f"[probe] {name} @ baseline {tag} ...".rstrip(), flush=True)
            br = run_probe(name, family, rel, extra, args.baseline, args)
            print(f"        -> {br.status} ({br.duration_s:.0f}s)")
            print(f"[probe] {name} @ candidate {tag} ...".rstrip(),
                  flush=True)
            cr = run_probe(name, family, rel, extra, args.candidate, args)
            print(f"        -> {cr.status} ({cr.duration_s:.0f}s)")
            base_runs.append(br)
            cand_runs.append(cr)
        base_results[name] = merge_results(name, base_runs)
        cand_results[name] = merge_results(name, cand_runs)

    report = ["# ggml upgrade verification report",
              "",
              f"- baseline: `{args.baseline}`",
              f"- candidate: `{args.candidate}`",
              f"- threshold: +{args.threshold}% (noise floor "
              f"{NOISE_FLOOR_PCT}%)",
              f"- device: {args.device}",
              f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
              ""]
    overall = "ok"
    for name, family, rel, extra in PROBES:
        if only and name not in only:
            continue
        status, rows = compare(base_results[name], cand_results[name],
                               args.threshold)
        if status == "regression" or status == "fail":
            overall = "regression"
        icon = {"ok": "PASS", "regression": "**REGRESSION**",
                "fail": "FAIL", "skipped": "skipped",
                "no-baseline": "NO-BASELINE"}[status]
        report.append(f"## {name} — {icon}")
        report.extend(f"- {r}" for r in rows)
        report.append("")

    pct, missing = coverage(cand_results)
    report += ["## Coverage",
               f"- (task, backend) pairs verified: {pct:.1f}% "
               f"({len(COVERAGE_UNIVERSE) - len(missing)}/{len(COVERAGE_UNIVERSE)})",
               f"- uncovered: {', '.join(missing) if missing else 'none'}",
               ""]
    report.append(f"## Verdict: {'REGRESSION/FAIL — do not release' if overall != 'ok' else 'NO REGRESSION'}")

    text = "\n".join(report)
    args.report.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nreport written to {args.report}")
    return 0 if overall == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
