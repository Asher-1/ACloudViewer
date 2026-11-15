#!/usr/bin/env python3
"""Run the per-model YOLO matrix and compare with the pinned upstream baseline.

Device/thread policy and separate processes match the historical Bash runner;
artifact discovery and native loader paths use AICore's shared runtime module.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from runtime_environment import (build_configurations, configured_qt_library_dir,
                                 embedded_qt_library_dir, find_build_tool,
                                 prepend_runtime_paths, runtime_directories)


def parser():
    repo = Path(__file__).resolve().parents[4]
    upstream = Path(os.environ.get("UPSTREAM_REPO", Path.home() /
                                  "develop/code/github/dl/ultralytics-ggml"))
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--build", type=Path,
                        default=Path(os.environ.get("BUILD_DIR", repo / "build_app")))
    result.add_argument("--upstream-bench", type=Path,
                        default=upstream / "cpp_ggml/benchmarks/bench.jsonl")
    result.add_argument("--models-dir", type=Path, default=upstream / "cpp_ggml/models/gguf")
    result.add_argument("--image", type=Path, default=upstream / "ultralytics/assets/bus.jpg")
    result.add_argument("--devices", default="cuda,vulkan,cpu")
    result.add_argument("--out-dir", type=Path, default=repo / "build_app/yolo_parity")
    result.add_argument("--warmup", type=int, default=20)
    result.add_argument("--iters", type=int, default=50)
    result.add_argument("--limit", type=float, default=5)
    result.add_argument("--no-build", action="store_true")
    return result


def run(args):
    if not args.upstream_bench.is_file():
        raise FileNotFoundError(f"upstream bench not found: {args.upstream_bench}")
    models = sorted(args.models_dir.glob("*.gguf"))
    if not models:
        raise FileNotFoundError(f"no GGUF models found: {args.models_dir}")
    if not args.no_build:
        subprocess.run(["cmake", "--build", str(args.build), "--config",
                        build_configurations(args.build)[0], "--target",
                        "test_yolo_capi_performance", "--parallel",
                        os.environ.get("BUILD_JOBS", "8")], check=True)
    binary = find_build_tool(args.build, "aicore_tests/test_yolo_capi_performance")
    environment = os.environ.copy()
    directories = runtime_directories(args.build)
    qt = configured_qt_library_dir(args.build) or embedded_qt_library_dir(args.build)
    if qt:
        directories.insert(0, qt)
    prepend_runtime_paths(environment, directories)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    outputs = []
    for device in args.devices.split(","):
        device = device.strip()
        if not device:
            continue
        output = args.out_dir / f"aicore-{device}.jsonl"
        outputs.append(output)
        with output.open("wb") as stdout, output.with_suffix(".log").open("wb") as stderr:
            for model in models:
                env = environment.copy()
                env.update(AICORE_TEST_YOLO_GGUF=str(model), AICORE_TEST_YOLO_DEVICE=device,
                           AICORE_TEST_YOLO_IMAGE=str(args.image),
                           AICORE_TEST_YOLO_WARMUP=str(args.warmup),
                           AICORE_TEST_YOLO_ITERS=str(args.iters))
                env.pop("AICORE_TEST_YOLO_THREADS", None)
                if device == "cpu":
                    env["AICORE_TEST_YOLO_THREADS"] = os.environ.get("YOLO_CPU_THREADS", "8")
                elif device in ("cuda", "vulkan"):
                    env["AICORE_TEST_YOLO_THREADS"] = os.environ.get("YOLO_GPU_THREADS", "32")
                result = subprocess.run([str(binary)], env=env, stdout=stdout, stderr=stderr)
                if result.returncode:
                    failed = True
                    print(f"FAILED: {model.name} on {device}; see {output.with_suffix('.log')}",
                          file=sys.stderr)
    # Combine this invocation's outputs once; never include a stale combined file.
    combined = args.out_dir / "aicore-all.jsonl"
    with combined.open("wb") as stream:
        for output in outputs:
            stream.write(output.read_bytes())
    gate = subprocess.run([sys.executable, str(Path(__file__).with_name("bench_compare.py")),
                           "--upstream", str(args.upstream_bench), "--aicore", str(combined),
                           "--limit", str(args.limit)])
    return int(failed or gate.returncode != 0)


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (OSError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
