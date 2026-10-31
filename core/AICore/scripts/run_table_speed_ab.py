#!/usr/bin/env python3
"""Same-host speed A/B: AICore `table` vs the official PaddleX pipeline.

Both sides see the same images, the same warmups and the same number of
timed passes, and both time from decoded pixels (Paddle receives the cv2
ndarray; bench_table decodes before its timer), so file decode is outside
every number. Two scopes are measured:

  structure   SLANet_plus alone (PaddleX ``create_model("SLANet_plus")`` vs
              ``bench_table --table``): the table-structure model itself
  pipeline    det + rec + structure + cell matching (PaddleX
              ``table_recognition`` with the GT-matrix config vs
              ``bench_table --det --rec --table``, det max/960)

Per AICore row (backend x quant) the report has p50/p95 per scope, the
speedup against Paddle GPU p50, and the per-image structure agreement with
Paddle's own prediction on the same image (TEDS-S mean and exact rate),
next to the host state (CPU governor, GPU clocks/processes) that decides
whether the numbers are release evidence.

    run_table_speed_ab.py --images DIR --list val_list.txt --bench bench_table \
        --models-dir ~/cloudViewer_data/extract/ocr_models --output /tmp/ab

Requires paddlepaddle-gpu 3.0.0 + paddlex 3.0.1 for the Paddle rows.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_teds as TEDS  # noqa: E402

QUANTS = ("f32", "f16", "q8_0")


def pct(values: list[float], p: float) -> float:
    v = sorted(values)
    return v[int(p * (len(v) - 1))] if v else 0.0


def host_state() -> dict:
    state = {}
    try:
        state["cpu_governor"] = Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        state["cpu_governor"] = "unknown"
    for key, query in (("gpu", "name,clocks.sm,clocks.max.sm,utilization.gpu,memory.used"),):
        try:
            state[key] = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True, text=True, check=True).stdout.strip()
            state["gpu_processes"] = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            state[key] = "unavailable"
    return state


PADDLE_CHILD = r'''
import json, sys, time
from pathlib import Path
import cv2
sys.path.insert(0, sys.argv[1])
from run_paddle_table_baseline import pipeline_config, source_dirs
cfg = json.loads(sys.argv[2])
images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in cfg["images"]]
names = [Path(p).name for p in cfg["images"]]
if cfg["scope"] == "structure":
    from paddlex import create_model
    model = create_model(model_name="SLANet_plus", device=cfg["device"])
    def run(img):
        r = next(iter(model.predict(img)))
        return "".join(r["structure"])
else:
    from paddlex import create_pipeline
    pipe = create_pipeline(config=pipeline_config(*source_dirs(Path(cfg["det_gguf"]),
                                                               Path(cfg["rec_gguf"]))),
                           device=cfg["device"])
    def run(img):
        r = next(iter(pipe.predict(img, use_doc_orientation_classify=False,
                                   use_doc_unwarping=False, use_layout_detection=False,
                                   use_ocr_model=True)))
        t = r["table_res_list"]
        return t[0]["pred_html"] if t else ""
for _ in range(cfg["warmups"]):
    run(images[0])
out = {"ms": [], "html": {}}
for p in range(cfg["passes"]):
    for name, img in zip(names, images):
        t0 = time.perf_counter()
        html = run(img)
        out["ms"].append((time.perf_counter() - t0) * 1000.0)
        if p == 0:
            out["html"][name] = html
print("@@RESULT@@" + json.dumps(out, ensure_ascii=False))
'''


def run_paddle(scope: str, images: list[str], args) -> dict:
    cfg = {"scope": scope, "images": images, "device": args.paddle_device,
           "warmups": args.warmups, "passes": args.passes,
           # Same-weight baseline: the F32 GGUFs' digest-verified sources.
           "det_gguf": str(args.models_dir / "ppocr-det-ch-f32.gguf"),
           "rec_gguf": str(args.models_dir / "ppocr-rec-ch-f32.gguf")}
    proc = subprocess.run([sys.executable, "-c", PADDLE_CHILD,
                           str(Path(__file__).resolve().parent), json.dumps(cfg)],
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    raise SystemExit(f"paddle {scope} failed:\n{proc.stderr[-3000:]}")


def run_aicore(scope: str, backend: str, quant: str, images: list[str], args) -> dict:
    models = args.models_dir
    cmd = [str(args.bench), "--table", str(models / f"slanet-plus-table-{quant}.gguf"),
           "--device", backend, "--warmups", str(args.warmups),
           "--threads", str(args.threads)]
    if scope == "pipeline":
        # Uniform tier, the qOCR configuration (det/rec/table share the quant).
        cmd += ["--det", str(models / f"ppocr-det-ch-{quant}.gguf"),
                "--rec", str(models / f"ppocr-rec-ch-{quant}.gguf"),
                "--det-limit-type", "1", "--det-max-side", "960"]
    listing = args.output / f"list_{scope}_{backend}_{quant}.txt"
    listing.write_text("\n".join(images * args.passes) + "\n", encoding="utf-8")
    cmd += ["--list", str(listing)]
    env = os.environ.copy()
    bin_dir = args.bench.resolve().parent
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(bin_dir), str(bin_dir.parent / "external" / "lib"), env.get("LD_LIBRARY_PATH", "")])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"bench_table {scope}/{backend}/{quant} failed:\n{proc.stderr[-3000:]}")
    out = {"ms": [], "html": {}}
    for line in proc.stdout.splitlines():
        record = json.loads(line)
        if record.get("summary"):
            continue
        key = "table_ms" if scope == "structure" else None
        out["ms"].append(record[key] if key else record["ocr_ms"] + record["table_ms"])
        name = Path(record["image"]).name
        if name not in out["html"]:
            tables = record.get("tables") or []
            out["html"][name] = tables[0]["html"] if tables else ""
    return out


def agreement(aicore: dict, paddle: dict) -> dict:
    scores = [TEDS.teds(aicore["html"].get(n, ""), h, structure_only=True)
              for n, h in paddle["html"].items()]
    return {"teds_s_vs_paddle": round(statistics.mean(scores), 6),
            "exact_vs_paddle": round(sum(s == 1.0 for s in scores) / len(scores), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True,
                        help="image names relative to --images, one per line")
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=("cuda", "vulkan"))
    parser.add_argument("--quants", nargs="+", default=QUANTS, choices=QUANTS)
    parser.add_argument("--scopes", nargs="+", default=("structure", "pipeline"),
                        choices=("structure", "pipeline"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--paddle-device", default="gpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    images = [str(args.images / line.strip())
              for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {"host_before": host_state(), "images": len(images),
              "warmups": args.warmups, "passes": args.passes, "scopes": {}}
    for scope in args.scopes:
        paddle = run_paddle(scope, images, args)
        p50 = statistics.median(paddle["ms"])
        rows = [{"side": "paddle", "backend": args.paddle_device, "quant": "fp32",
                 "p50_ms": round(p50, 2), "p95_ms": round(pct(paddle["ms"], 0.95), 2),
                 "samples": len(paddle["ms"])}]
        print(json.dumps({"scope": scope, **rows[0]}), flush=True)
        for backend in args.backends:
            for quant in args.quants:
                ours = run_aicore(scope, backend, quant, images, args)
                ours_p50 = statistics.median(ours["ms"])
                row = {"side": "aicore", "backend": backend, "quant": quant,
                       "p50_ms": round(ours_p50, 2), "p95_ms": round(pct(ours["ms"], 0.95), 2),
                       "samples": len(ours["ms"]),
                       "speedup_vs_paddle_p50": round(p50 / ours_p50, 3),
                       **agreement(ours, paddle)}
                rows.append(row)
                print(json.dumps({"scope": scope, **row}), flush=True)
        report["scopes"][scope] = rows
    report["host_after"] = host_state()
    (args.output / "table_speed_ab.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output / "table_speed_ab.json"),
                      "host": report["host_before"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
