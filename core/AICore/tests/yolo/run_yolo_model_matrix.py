#!/usr/bin/env python3
"""Run and audit the YOLO task/quant/backend benchmark and parity matrix."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


TASKS = ("detect", "segment", "depth", "pose", "obb", "semantic", "classify")
QUANTS = ("f32", "f16", "q8_0")


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def benchmark_rows(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("suite") == "aicore":
            rows.append(row)
    return rows


def command_output(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def text_role(filename: str) -> str:
    if filename.startswith("yoloe") and "-pf-" not in filename:
        return "yoloe"
    if "world" in filename:
        return "world"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--models-dir", default=os.getenv("AICORE_TEST_YOLO_MODELS_DIR"))
    parser.add_argument("--image", default=os.getenv("AICORE_TEST_YOLO_IMAGE"))
    parser.add_argument(
        "--classes",
        default=os.getenv("AICORE_TEST_YOLO_CLASSES"),
        help="comma-separated classes required by open-vocabulary models",
    )
    parser.add_argument(
        "--text-model",
        default=os.getenv("AICORE_TEST_YOLO_TEXT_MODEL"),
        help="legacy fallback text model for both open-vocabulary roles",
    )
    parser.add_argument(
        "--world-text-model",
        default=os.getenv("AICORE_TEST_YOLO_WORLD_TEXT_MODEL"),
        help="CLIP-compatible text GGUF for YOLO-World",
    )
    parser.add_argument(
        "--yoloe-text-model",
        default=os.getenv("AICORE_TEST_YOLOE_TEXT_MODEL"),
        help="MobileCLIP-compatible text GGUF for YOLOE",
    )
    parser.add_argument(
        "--semantic-reference",
        default=os.getenv("AICORE_TEST_YOLO_SEMANTIC_REFERENCE"),
        help="raw uint8 PyTorch semantic class map for the same image/input size",
    )
    parser.add_argument("--backends", default="cpu,cuda,vulkan")
    parser.add_argument(
        "--model-globs",
        default=os.getenv("AICORE_TEST_YOLO_MODEL_GLOBS", ""),
        help="comma-separated fnmatch patterns against model file names; "
             "empty means every YOLO inference model in --models-dir")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--report")
    args = parser.parse_args()

    if not args.models_dir or not args.image:
        print("[yolo-matrix] SKIP: set AICORE_TEST_YOLO_MODELS_DIR and AICORE_TEST_YOLO_IMAGE")
        return 77
    models_dir = Path(args.models_dir)
    image = Path(args.image)
    if not models_dir.is_dir() or not image.is_file():
        print(f"[yolo-matrix] SKIP: missing assets: models={models_dir} image={image}")
        return 77
    expected_files = {
        path.name
        for path in models_dir.glob("*.gguf")
        if not path.name.startswith(("clip-", "mobileclip", "mclip-"))
    }
    model_globs = [
        item.strip() for item in args.model_globs.split(",") if item.strip()
    ]
    if model_globs:
        expected_files = {
            name
            for name in expected_files
            if any(fnmatch.fnmatchcase(name, pattern)
                   for pattern in model_globs)
        }
    if not expected_files:
        print(
            "[yolo-matrix] SKIP: no YOLO inference model assets"
            + (f" matching {model_globs}" if model_globs else "")
        )
        return 77
    world_text_model = args.world_text_model or args.text_model
    yoloe_text_model = args.yoloe_text_model or args.text_model
    required_roles = {text_role(name) for name in expected_files} - {""}
    text_models = {"world": world_text_model, "yoloe": yoloe_text_model}
    if required_roles and not args.classes:
        print("[yolo-matrix] --classes is required by open-vocabulary models")
        return 1
    for role in sorted(required_roles):
        model = text_models[role]
        if not model or not Path(model).is_file():
            print(f"[yolo-matrix] missing {role} text model: {model or '<unset>'}")
            return 1

    backends = tuple(item.strip() for item in args.backends.split(",") if item.strip())
    env_base = os.environ.copy()
    env_base["AICORE_TEST_YOLO_MODELS_DIR"] = str(models_dir)
    env_base["AICORE_TEST_YOLO_IMAGE"] = str(image)
    env_base["AICORE_TEST_YOLO_WARMUP"] = str(args.warmup)
    env_base["AICORE_TEST_YOLO_ITERS"] = str(args.iters)
    # Install only the role-correct prompt payload for each row below.
    env_base.pop("AICORE_TEST_YOLO_CLASSES", None)
    env_base.pop("AICORE_TEST_YOLO_TEXT_MODEL", None)
    if args.semantic_reference:
        semantic_reference = Path(args.semantic_reference)
        if not semantic_reference.is_file():
            print(f"[yolo-matrix] invalid semantic reference: {semantic_reference}")
            return 1
        env_base["AICORE_TEST_YOLO_SEMANTIC_REFERENCE"] = str(semantic_reference)

    rows_by_backend: dict[str, list[dict[str, object]]] = {}
    failures: list[str] = []
    for backend in backends:
        rows_by_backend[backend] = []
        for filename in sorted(expected_files):
            env = env_base.copy()
            env["AICORE_TEST_YOLO_DEVICE"] = backend
            role = text_role(filename)
            if role:
                env["AICORE_TEST_YOLO_CLASSES"] = args.classes
                env["AICORE_TEST_YOLO_TEXT_MODEL"] = str(text_models[role])
            completed = run([args.benchmark, str(models_dir / filename)], env)
            sys.stdout.write(completed.stdout)
            if completed.returncode != 0:
                failures.append(
                    f"benchmark:{backend}:{filename}:exit={completed.returncode}"
                )
            rows_by_backend[backend].extend(benchmark_rows(completed.stdout))

    gpu_backends = tuple(backend for backend in backends if backend != "cpu")
    parity_results: dict[str, dict[str, object]] = {}
    for backend in gpu_backends:
        exit_codes: dict[str, int] = {}
        summary: list[str] = []
        for filename in sorted(expected_files):
            env = env_base.copy()
            env["AICORE_TEST_YOLO_PARITY_DEVICE"] = backend
            env["AICORE_TEST_YOLO_GGUF"] = str(models_dir / filename)
            role = text_role(filename)
            if role:
                env["AICORE_TEST_YOLO_CLASSES"] = args.classes
                env["AICORE_TEST_YOLO_TEXT_MODEL"] = str(text_models[role])
            completed = run([args.parity], env)
            sys.stdout.write(completed.stdout)
            exit_codes[filename] = completed.returncode
            summary.extend(
                line
                for line in completed.stdout.splitlines()
                if "FAIL:" in line or line.startswith("[yolo-parity] done:")
            )
            if completed.returncode != 0:
                failures.append(
                    f"parity:cpu-vs-{backend}:{filename}:"
                    f"exit={completed.returncode}"
                )
        parity_results[backend] = {
            "exit_codes": exit_codes,
            "summary": summary,
        }

    files_by_backend = {
        backend: {str(row.get("file")) for row in rows}
        for backend, rows in rows_by_backend.items()
    }
    missing_model_backend = [
        {"file": filename, "backend": backend}
        for filename in sorted(expected_files)
        for backend in backends
        if filename not in files_by_backend[backend]
    ]

    covered_cells = {
        (str(row.get("task")), str(row.get("dtype")), backend)
        for backend, rows in rows_by_backend.items()
        for row in rows
    }
    missing_task_quant_backend = [
        {"task": task, "quant": quant, "backend": backend}
        for task in TASKS
        for quant in QUANTS
        for backend in backends
        if (task, quant, backend) not in covered_cells
    ]
    repo_root = Path(__file__).resolve().parents[4]
    revision = command_output(["git", "rev-parse", "HEAD"], repo_root)
    dirty = bool(command_output(["git", "status", "--porcelain"], repo_root))
    gpu_info = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    model_assets = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "resolved_path": str(path.resolve()),
        }
        for path in sorted(models_dir.glob("*.gguf"))
        if path.name in expected_files
    ]
    report = {
        "schema": 3,
        "revision": revision,
        "worktree_dirty": dirty,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "nvidia": gpu_info,
        },
        "models_dir": str(models_dir),
        "model_assets": model_assets,
        "classes": args.classes or "",
        "text_models": {
            "world": world_text_model or "",
            "yoloe": yoloe_text_model or "",
        },
        "image": str(image),
        "semantic_reference": args.semantic_reference or "",
        "warmup": args.warmup,
        "iterations": args.iters,
        "backends": list(backends),
        "rows": {backend: len(rows) for backend, rows in rows_by_backend.items()},
        "measurements": rows_by_backend,
        "parity": parity_results,
        "missing_model_backend": missing_model_backend,
        "missing_task_quant_backend": missing_task_quant_backend,
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")

    if failures or missing_model_backend or missing_task_quant_backend:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
