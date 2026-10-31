#!/usr/bin/env python3
"""Public-ABI dataset probe, using the upstream evaluators as independent truth.

This is a probe consumed by validate_all.py, not a separate matrix runner.
The checkout is supplied through AICORE_TEST_RAM_UPSTREAM (test-only).
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time


def gpu_overlap_pids(samples, allowed_pids, child_pid):
    """Return external CUDA compute PIDs; desktop graphics are observational."""
    return sorted({pid for sample in samples for pid in sample["pids"]
                   if pid not in allowed_pids and pid != child_pid})


def aicore_latency_ms(row):
    """Use the typed decoded-image pipeline timing, excluding caller decode."""
    return row.get("e2e_ms", row["latency_ms"])


def active_graphics_processes(output, threshold=5):
    """NVIDIA compute-app queries omit GPU work submitted by the desktop."""
    active = set()
    for line in output.splitlines():
        columns = line.split()
        if len(columns) < 4 or columns[2] not in ("G", "C+G"):
            continue
        try:
            if int(columns[3]) > threshold:
                active.add(int(columns[1]))
        except ValueError:
            continue
    return active


def include_graphics_audit(evaluation, idle_timeout=60):
    compute_processes = evaluation.gpu_process_ids
    reservation_wait = None

    def sample():
        monitor = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                                 capture_output=True, text=True, check=True)
        return {"pids": sorted(compute_processes()),
                "active_graphics_pids": sorted(active_graphics_processes(monitor.stdout)),
                "graphics_pmon": monitor.stdout}

    def gpu_processes():
        nonlocal reservation_wait
        activity = sample()
        busy = set(activity["pids"])
        if busy:
            now = time.monotonic()
            if reservation_wait is None:
                reservation_wait = now
            elif now - reservation_wait >= idle_timeout:
                raise RuntimeError(f"GPU did not become idle within {idle_timeout}s; "
                                   f"active processes: {sorted(busy)}")
        else:
            reservation_wait = None
        return busy

    def run_command_monitored(command, log_path, allowed_gpu_pids):
        idle_start = time.monotonic()
        waiting = []
        while True:
            activity = sample()
            busy = gpu_overlap_pids([activity], allowed_gpu_pids, None)
            if not busy:
                break
            waiting.append(activity)
            if time.monotonic() - idle_start >= idle_timeout:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps({"gpu_idle_timeout_seconds": idle_timeout,
                    "active_gpu_pids": busy, "waiting_samples": waiting}, indent=2) + "\n")
                raise RuntimeError(f"GPU did not become idle within {idle_timeout}s; "
                                   f"audit: {log_path}")
            print(f"waiting for transient GPU activity: {busy}", flush=True)
            time.sleep(5)
        begin = time.perf_counter()
        samples, errors = [], []
        stop = threading.Event()
        process = subprocess.Popen(command, cwd=evaluation.ROOT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def monitor():
            while not stop.is_set():
                try:
                    activity = sample()
                    activity["seconds"] = time.perf_counter() - begin
                    samples.append(activity)
                except (OSError, subprocess.SubprocessError) as error:
                    errors.append(str(error))
                stop.wait(0.5)

        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise
        finally:
            stop.set()
            worker.join()
        transient = gpu_overlap_pids(samples, allowed_gpu_pids, process.pid)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(stdout + "\n--- stderr ---\n" + stderr +
                            "\n--- gpu timing audit ---\n" + json.dumps({
                                "allowed_gpu_pids": sorted(allowed_gpu_pids),
                                "command_pid": process.pid,
                                "desktop_graphics_observational": True,
                                "graphics_sm_threshold_pct": 5,
                                "transient_gpu_pids": transient, "samples": samples},
                                indent=2) + "\n", encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"command failed ({process.returncode}): {command[0]}\n{stderr[-2000:]}")
        if errors or not samples:
            raise RuntimeError(f"GPU timing audit failed: {errors}")
        return {"wall_seconds": time.perf_counter() - begin,
                "transient_gpu_pids": transient, "stdout": stdout[-1000:]}

    # Preserve CUDA compute isolation. Desktop graphics are retained in raw
    # samples for diagnosis but do not invalidate a desktop-hosted benchmark.
    evaluation.gpu_process_ids = gpu_processes
    evaluation.run_command_monitored = run_command_monitored


def execute(command, log, env=None):
    result = subprocess.run([str(item) for item in command], capture_output=True,
                            text=True, check=False, env=env)
    log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"probe command failed ({result.returncode}): "
                           f"{command[0]}: {result.stderr[-2000:]}")


def execute_audited(command, log, evaluation, allowed_pids, env=None, retries=3):
    """Reuse the upstream overlap audit, including its retained failed samples."""
    original_env = os.environ.copy()
    try:
        if env is not None:
            os.environ.clear()
            os.environ.update(env)
        for attempt in range(retries + 1):
            result = evaluation.run_command_monitored(
                [str(item) for item in command], log, allowed_pids)
            if not result["transient_gpu_pids"]:
                return
            discarded = log.with_name(f"{log.stem}.overlap-{attempt + 1}.log")
            log.rename(discarded)
            print(f"Discarding GPU-overlapped sample {result['transient_gpu_pids']}; "
                  f"audit: {discarded}", flush=True)
        raise RuntimeError(f"GPU overlap persisted after {retries + 1} attempts; "
                           f"last rejected sample: {discarded.name}; audit: {discarded}")
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def upstream_backend_env(upstream, backend, build=None):
    """Keep the upstream process on its own GGML build during controlled A/B.

    validate_all adds the candidate build's libraries to the environment.
    Since both builds use the same GGML soname, that path overrides upstream
    RUNPATH unless its own library directories come first.
    """
    build = build or upstream / "cpp_ggml" / f"build-{backend}"
    lib_dir = build / "third_party" / "ggml" / "src"
    env = os.environ.copy()
    variable = ("PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH"
                if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    library_paths = [str(lib_dir), str(lib_dir / f"ggml-{backend}"), str(build / "bin")]
    if env.get(variable):
        library_paths.append(env[variable])
    env[variable] = os.pathsep.join(library_paths)
    env.pop("GGML_BACKEND_PATH", None)
    return env


def truth_cache(root, revision, family, task, manifest, records, extra, checkpoint,
                reference_device="cuda"):
    digest = hashlib.sha256((revision + family + task + manifest.read_text()).encode())
    if reference_device != "cuda":
        # Keep existing CUDA truth reusable; CPU references own a distinct key.
        digest.update(("reference_device=" + reference_device).encode())
    for path in [checkpoint, *(Path(record["image"]) for record in records), *extra[1::2]]:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    cache = root / "ram-pytorch-reference" / digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def build_settings(build):
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    settings = {}
    for line in cache.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key.startswith(("GGML_", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
                           "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS")) or key in (
                               "CMAKE_BUILD_TYPE", "CMAKE_HOME_DIRECTORY"):
            settings[key] = value
    return settings


def candidate_build_settings(build):
    """Read the selected source owner's flags, retaining old baseline support."""
    selected = build_settings(build).get("GGML_BUILD_DIR")
    dependency = Path(selected) if selected else build / "ggml/src/ext_ggml-build"
    settings = build_settings(dependency)
    if not settings:
        raise RuntimeError(f"candidate ggml build settings unavailable: {dependency}")
    return settings


def probability_alignment(scores, baseline_scores, truth_scores):
    """Bound independent error and detect added error beyond GGUF quantization."""
    error = abs(truth_scores - scores)
    baseline_error = abs(truth_scores - baseline_scores)
    return {
        "pytorch_probability_mae": float(error.mean()),
        "pytorch_probability_max_error": float(error.max()),
        "upstream_pytorch_probability_mae": float(baseline_error.mean()),
        "upstream_pytorch_probability_max_error": float(baseline_error.max()),
        "pytorch_probability_error_excess": float(
            (error - baseline_error).clip(min=0).max()),
    }


def metrics_for_backend(metrics, backend):
    """Expose compatibility claims only when the CPU path actually runs one."""
    if backend == "cpu":
        return dict(metrics)
    return {key: value for key, value in metrics.items()
            if not key.startswith("compatibility_")}


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, default=(
        Path(tempfile.gettempdir()) / "aicore-tagging-reference-cache"))
    parser.add_argument("--upstream-build", type=Path,
                        default=os.environ.get("AICORE_TEST_RAM_UPSTREAM_BUILD"),
                        help="Alternate upstream build with matching compiler/CPU flags; source remains upstream")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--task", choices=("tags", "caption"), default="tags")
    parser.add_argument("--reference-device", choices=("cuda", "cpu"),
                        default=os.environ.get("AICORE_TEST_RAM_REFERENCE_DEVICE", "cuda"),
                        help="Official PyTorch reference device; CPU supports hosts without CUDA")
    parser.add_argument("--allow-gpu-pid", type=int, action="append",
                        default=[int(pid) for pid in os.environ.get(
                            "AICORE_TEST_RAM_ALLOWED_GPU_PIDS", "").split(",") if pid],
                        help="Explicitly allow an idle long-lived GPU context in the timing audit")
    parser.add_argument("--gpu-retries", type=int, default=3)
    parser.add_argument("--gpu-idle-timeout", type=float, default=60,
                        help="Maximum seconds waiting for a clean GPU sample; busy rows fail")
    parser.add_argument("--wait-process-pid", type=int, action="append",
                        default=[int(pid) for pid in os.environ.get(
                            "AICORE_TEST_RAM_WAIT_PROCESS_PIDS", "").split(",") if pid],
                        help="Wait for another benchmark coordinator, including its GPU-idle gaps")
    args = parser.parse_args()
    for key in ("probe", "model", "assets", "output", "reference_cache", "upstream_build"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.gpu_retries < 0:
        parser.error("--gpu-retries must be nonnegative")
    if args.gpu_idle_timeout <= 0:
        parser.error("--gpu-idle-timeout must be positive")
    if args.threads <= 0:
        args.threads = 8
    reference_root = args.reference_cache or args.output.parent
    upstream = Path(os.environ.get("AICORE_TEST_RAM_UPSTREAM", ""))
    evaluator = upstream / "cpp_ggml/benchmarks/full_eval.py"
    if not evaluator.is_file():
        raise RuntimeError("Set AICORE_TEST_RAM_UPSTREAM to a recognize-anything-ggml "
                           "checkout with official checkpoints and prepared label_spaces")
    upstream = upstream.resolve()
    upstream_build = args.upstream_build or upstream / "cpp_ggml" / f"build-{args.backend}"
    sys.path.insert(0, str(upstream / "cpp_ggml/scripts"))
    spec = importlib.util.spec_from_file_location("ram_upstream_eval", evaluator)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    audited = args.backend in ("cuda", "vulkan") and shutil.which("nvidia-smi") is not None
    if audited:
        include_graphics_audit(evaluation, args.gpu_idle_timeout)
    allowed_pids = set(args.allow_gpu_pid)
    gpu_slot = evaluation.reserve_gpu_slot(allowed_pids, args.wait_process_pid) if audited else None
    def run(command, log, env=None):
        if audited:
            execute_audited(command, log, evaluation, allowed_pids, env, args.gpu_retries)
        else:
            execute(command, log, env)
    family = ("ram_plus" if args.model.name.startswith("ram_plus_") else
              "tag2text" if args.model.name.startswith("tag2text_") else "ram")
    dtype = args.model.stem.rsplit("-", 1)[1]
    if args.task == "caption" and family != "tag2text":
        raise RuntimeError("caption probe requires Tag2Text")
    for info in evaluation.DATASET_INFO.values():
        for key, path in tuple(info.items()):
            if isinstance(path, Path) and path.is_relative_to(upstream / "datasets"):
                info[key] = args.assets / "ram_datasets" / path.relative_to(upstream / "datasets")
    datasets = evaluation.make_records("datasets", list(evaluation.DATASET_INFO))
    space_args = argparse.Namespace(label_space_dir=upstream / "cpp_ggml/benchmarks/label_spaces")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = upstream / "cpp_ggml/models/pytorch" / evaluation.MODEL_CHECKPOINTS[family]
    work = Path(tempfile.mkdtemp(prefix=f"ram-{family}-{args.backend}-{dtype}-",
                                 dir=args.output.parent))
    backend_env = upstream_backend_env(upstream, args.backend, upstream_build)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    from demo_reference_probe import REVISION
    if revision != REVISION:
        raise RuntimeError(f"upstream reference must be revision {REVISION}; got {revision}")
    results = []
    candidate_latencies, upstream_latencies = [], []
    for dataset, records in datasets.items():
        manifest = work / f"{dataset}.txt"
        evaluation.write_manifest(manifest, records)
        extra = []
        if args.task == "tags":
            space = evaluation.label_space(space_args, dataset, family)
            for key, flag in (("tags", "--tag-list"), ("thresholds", "--thresholds"),
                              ("embedding", "--label-embedding")):
                if space[key].is_file():
                    extra.extend((flag, space[key]))
            if family != "tag2text" and not space["embedding"].is_file():
                raise RuntimeError(f"missing upstream label-space embedding: {space['embedding']}")
        candidate = work / f"{dataset}-aicore.jsonl"
        logits = work / f"{dataset}-aicore.f32"
        command = [args.probe, "--model", args.model, "--backend", args.backend,
                   "--task", args.task, "--manifest", manifest, "--jsonl", candidate,
                   "--logits-matrix", logits, "--threads", args.threads,
                   "--warmup", args.warmup, "--repeat", args.repeat]
        run(command + extra, work / f"{dataset}-aicore.log")
        baseline = work / f"{dataset}-upstream.jsonl"
        baseline_logits = work / f"{dataset}-upstream.f32"
        baseline_command = list(command)
        baseline_command[0] = upstream_build / "ram-ggml"
        baseline_command[baseline_command.index("--jsonl") + 1] = baseline
        baseline_command[baseline_command.index("--logits-matrix") + 1] = baseline_logits
        run(baseline_command + extra, work / f"{dataset}-upstream.log", backend_env)
        # ABBA balances GPU clock/temperature drift. Retain both samples and
        # aggregate both per-image samples while preserving first-run numerics.
        second_baseline = work / f"{dataset}-upstream-second.jsonl"
        second_candidate = work / f"{dataset}-aicore-second.jsonl"
        second_baseline_command = list(baseline_command)
        second_baseline_command[second_baseline_command.index("--jsonl") + 1] = second_baseline
        second_baseline_command[second_baseline_command.index("--logits-matrix") + 1] = work / f"{dataset}-upstream-second.f32"
        run(second_baseline_command + extra, work / f"{dataset}-upstream-second.log", backend_env)
        second_candidate_command = list(command)
        second_candidate_command[second_candidate_command.index("--jsonl") + 1] = second_candidate
        second_candidate_command[second_candidate_command.index("--logits-matrix") + 1] = work / f"{dataset}-aicore-second.f32"
        run(second_candidate_command + extra, work / f"{dataset}-aicore-second.log")
        compatibility = candidate
        compatibility_logits = logits
        automatic_cpu = (args.backend == "cpu" and family == "ram" and dtype == "q8_0")
        if automatic_cpu:
            compatibility = work / f"{dataset}-aicore-upstream-arithmetic.jsonl"
            compatibility_logits = compatibility.with_suffix(".f32")
            compatibility_command = list(command)
            compatibility_command[compatibility_command.index("--jsonl") + 1] = compatibility
            compatibility_command[compatibility_command.index("--logits-matrix") + 1] = compatibility_logits
            # Numeric compatibility is tested separately from AUTO latency.
            compatibility_command[compatibility_command.index("--warmup") + 1] = 0
            compatibility_command[compatibility_command.index("--repeat") + 1] = 1
            run(compatibility_command + ["--cpu-precision", 1] + extra,
                work / f"{dataset}-aicore-upstream-arithmetic.log")
        rows = evaluation.load_rows(candidate)
        baseline_rows = evaluation.load_rows(baseline)
        candidate_second_rows = evaluation.load_rows(second_candidate)
        baseline_second_rows = evaluation.load_rows(second_baseline)
        if any(len(sample) != len(records) for sample in
               (rows, baseline_rows, candidate_second_rows, baseline_second_rows)):
            raise RuntimeError("dataset row count differs")
        for first, second in ((candidate, second_candidate),
                              (baseline, second_baseline)):
            left, right = evaluation.load_rows(first), evaluation.load_rows(second)
            if any(a["image"] != b["image"] or a["tags"] != b["tags"] or
                   a.get("caption") != b.get("caption") for a, b in zip(left, right)):
                raise RuntimeError(f"unstable results between processes: {first}")
            if first.with_suffix(".f32").read_bytes() != second.with_suffix(".f32").read_bytes():
                raise RuntimeError(f"unstable all-class logits between processes: {first}")
        classes = np.fromfile(logits, np.float32).size // len(records)
        raw_difference = np.abs(np.fromfile(logits, np.float32) -
                                np.fromfile(baseline_logits, np.float32))
        scores = evaluation.load_matrix(logits, len(records), classes, True)
        baseline_scores = evaluation.load_matrix(baseline_logits, len(records), classes, True)
        compatibility_rows = evaluation.load_rows(compatibility)
        if len(compatibility_rows) != len(records):
            raise RuntimeError("CPU compatibility row count differs")
        compatibility_scores = evaluation.load_matrix(compatibility_logits, len(records), classes, True)
        compatibility_raw_error = abs(np.fromfile(compatibility_logits, np.float32) -
                                      np.fromfile(baseline_logits, np.float32))
        candidate_samples = rows + candidate_second_rows
        upstream_samples = baseline_rows + baseline_second_rows
        metrics = {
            "dataset": dataset, "images": len(rows),
            "pipeline_ms": float(np.mean(
                [aicore_latency_ms(row) for row in candidate_samples])),
            "upstream_ms": float(np.mean(
                [row["latency_ms"] for row in upstream_samples])),
            "upstream_probability_mae": float(np.abs(scores - baseline_scores).mean()),
            "upstream_probability_max_error": float(np.abs(scores - baseline_scores).max()),
            "upstream_logit_mae": float(raw_difference.mean()),
            "upstream_logit_max_error": float(raw_difference.max()),
            "compatibility_upstream_logit_max_error": float(compatibility_raw_error.max()),
            "compatibility_upstream_probability_max_error": float(
                abs(compatibility_scores - baseline_scores).max()),
        }
        metrics["pipeline_images_per_second"] = 1000 / metrics["pipeline_ms"]
        metrics["upstream_images_per_second"] = 1000 / metrics["upstream_ms"]
        for prefix, samples, latency in (
                ("pipeline", candidate_samples, aicore_latency_ms),
                ("upstream", upstream_samples, lambda row: row["latency_ms"])):
            for percentile in (50, 95):
                metrics[f"{prefix}_p{percentile}_ms"] = float(np.percentile(
                    [latency(row) for row in samples], percentile))
        metrics["latency_over_upstream"] = metrics["pipeline_ms"] / metrics["upstream_ms"]
        candidate_latencies.extend(aicore_latency_ms(row)
                                   for row in candidate_samples)
        upstream_latencies.extend(row["latency_ms"] for row in upstream_samples)
        cross = evaluation.set_metrics([{tag["name"] for tag in row["tags"]} for row in baseline_rows],
                                       [{tag["name"] for tag in row["tags"]} for row in rows])
        metrics["upstream_cross_f1"] = cross["f1"]
        metrics["compatibility_upstream_cross_f1"] = evaluation.set_metrics(
            [{tag["name"] for tag in row["tags"]} for row in baseline_rows],
            [{tag["name"] for tag in row["tags"]} for row in compatibility_rows])["f1"]
        if args.task == "caption":
            from collections import Counter
            def word_f1(left, right):
                left, right = Counter(left.lower().split()), Counter(right.lower().split())
                return 2 * sum((left & right).values()) / max(1, sum(left.values()) + sum(right.values()))
            metrics["caption_exact"] = float(np.mean([a["caption"] == b["caption"]
                                                       for a, b in zip(rows, baseline_rows)]))
            metrics["caption_word_f1"] = float(np.mean([word_f1(a["caption"], b["caption"])
                                                         for a, b in zip(rows, baseline_rows)]))
            cache = truth_cache(reference_root, revision, family, args.task,
                                manifest, records, extra, checkpoint, args.reference_device)
            truth = cache / "truth.jsonl"
            if not (cache / "complete").is_file():
                execute([sys.executable, Path(__file__).with_name("caption_reference.py"),
                         "--upstream", upstream, "--manifest", manifest, "--output", truth,
                         "--warmup", args.warmup, "--repeat", args.repeat,
                         "--device", args.reference_device], cache / "reference.log")
                (cache / "complete").write_text(revision + "\n")
            truth_rows = evaluation.load_rows(truth)
            if len(truth_rows) != len(records):
                raise RuntimeError("PyTorch caption reference row count differs")
            metrics["pytorch_caption_exact"] = float(np.mean([a["caption"] == b["caption"]
                                                               for a, b in zip(rows, truth_rows)]))
            metrics["pytorch_caption_word_f1"] = float(np.mean([word_f1(a["caption"], b["caption"])
                                                                 for a, b in zip(rows, truth_rows)]))
        else:
            # Cache independent PyTorch scores only after a complete successful
            # run. Input/sidecar/revision fingerprints prevent stale truth reuse.
            cache = truth_cache(reference_root, revision, family, args.task,
                                manifest, records, extra, checkpoint, args.reference_device)
            truth = cache / "truth.jsonl"
            truth_matrix = cache / "truth.f32"
            if not (cache / "complete").is_file():
                execute([sys.executable, Path(__file__).with_name("tagging_reference.py"),
                         "--upstream", upstream,
                         "--model", family, "--manifest", manifest, "--output-jsonl", truth,
                         "--output-matrix", truth_matrix, "--batch-size", 1,
                         "--device", args.reference_device, "--warmup", args.warmup,
                         "--repeat", args.repeat] + extra,
                        cache / "reference.log")
                (cache / "complete").write_text(revision + "\n")
            truth_rows = evaluation.load_rows(truth)
            if len(truth_rows) != len(records):
                raise RuntimeError("PyTorch reference row count differs")
            truth_scores = evaluation.load_matrix(truth_matrix, len(records), classes, False)
            cross = evaluation.set_metrics([{tag["name"] for tag in row["tags"]} for row in truth_rows],
                                           [{tag["name"] for tag in row["tags"]} for row in rows])
            upstream_cross = evaluation.set_metrics(
                [{tag["name"] for tag in row["tags"]} for row in truth_rows],
                [{tag["name"] for tag in row["tags"]} for row in baseline_rows])
            metrics.update({"cross_f1": cross["f1"], "cross_precision": cross["precision"],
                            "cross_recall": cross["recall"],
                            "upstream_pytorch_cross_f1": upstream_cross["f1"],
                            "pytorch_cross_f1_delta": abs(cross["f1"] - upstream_cross["f1"]),
                            "pytorch_cross_f1_drop": max(0., upstream_cross["f1"] - cross["f1"])})
            metrics.update(probability_alignment(scores, baseline_scores, truth_scores))
            metrics["compatibility_pytorch_probability_error_excess"] = probability_alignment(
                compatibility_scores, baseline_scores, truth_scores)["pytorch_probability_error_excess"]
            compatibility_cross = evaluation.set_metrics(
                [{tag["name"] for tag in row["tags"]} for row in truth_rows],
                [{tag["name"] for tag in row["tags"]} for row in compatibility_rows])
            metrics["compatibility_pytorch_cross_f1_drop"] = max(
                0., upstream_cross["f1"] - compatibility_cross["f1"])
            if automatic_cpu:
                arithmetic_cache = truth_cache(reference_root, revision, family,
                    "cpu-q8-auto-pytorch-v1-" + hashlib.sha256(
                        Path(__file__).with_name("cpu_precision_reference.py").read_bytes()).hexdigest(),
                    manifest, records, extra, args.model, "cpu")
                arithmetic_scores_file = arithmetic_cache / "truth.f32"
                if not (arithmetic_cache / "complete").is_file():
                    execute([sys.executable, Path(__file__).with_name("cpu_precision_reference.py"),
                             "--upstream", upstream, "--gguf", args.model,
                             "--threads", args.threads, "--model", family,
                             "--manifest", manifest, "--output-jsonl", arithmetic_cache / "truth.jsonl",
                             "--output-matrix", arithmetic_scores_file, "--device", "cpu",
                             "--batch-size", 1, "--warmup", 0, "--repeat", 1] + extra,
                            arithmetic_cache / "reference.log")
                    (arithmetic_cache / "complete").write_text(revision + "\n")
                arithmetic_scores = evaluation.load_matrix(arithmetic_scores_file, len(records), classes, False)
                metrics["arithmetic_reference_sha256"] = arithmetic_cache.name
                metrics["arithmetic_reference"] = "independent_pytorch_cpu_q8_auto"
            else:
                arithmetic_scores = baseline_scores
                metrics["arithmetic_reference"] = "original_upstream_ggml"
            metrics["arithmetic_probability_max_error"] = float(abs(scores - arithmetic_scores).max())
            metrics["arithmetic_probability_mae"] = float(abs(scores - arithmetic_scores).mean())
            labels_key = "labels_tag2text" if family == "tag2text" else "labels_ram"
            metrics["annotation_targets"] = sum(len(set(record.get(labels_key, [])))
                                                 for record in records)
            metrics["annotation_role"] = ("ground_truth" if metrics["annotation_targets"]
                                           else "zero_target_diagnostic")
            metrics.update(evaluation.annotation_metrics(records, family, scores, rows, dataset, space_args))
            if metrics["annotation_status"] != "ok":
                raise RuntimeError(f"unscored label space: {dataset}: {metrics['annotation_status']}")
            for prefix, reference_scores, reference_rows in (
                    ("upstream", baseline_scores, baseline_rows),
                    ("pytorch", truth_scores, truth_rows)):
                annotation = evaluation.annotation_metrics(
                    records, family, reference_scores, reference_rows, dataset, space_args)
                if annotation["annotation_status"] != "ok":
                    raise RuntimeError(f"unscored {prefix} annotation: {dataset}")
                for key in ("mAP", "CP", "CR", "precision", "recall", "f1",
                            "macro_precision", "macro_recall"):
                    metrics[f"{prefix}_{key}"] = annotation[key]
                    metrics[f"{prefix}_{key}_delta"] = abs(metrics[key] - annotation[key])
            metrics["pytorch_annotation_error_excess"] = max(
                max(0., metrics[f"pytorch_{key}_delta"] -
                    abs(metrics[f"upstream_{key}"] - metrics[f"pytorch_{key}"]))
                for key in ("mAP", "CP", "CR", "precision", "recall", "f1",
                            "macro_precision", "macro_recall"))
            compatibility_annotation = evaluation.annotation_metrics(
                records, family, compatibility_scores, compatibility_rows, dataset, space_args)
            if compatibility_annotation["annotation_status"] != "ok":
                raise RuntimeError(f"unscored CPU compatibility annotation: {dataset}")
            metrics["compatibility_pytorch_annotation_error_excess"] = max(
                max(0., abs(compatibility_annotation[key] - metrics[f"pytorch_{key}"]) -
                    abs(metrics[f"upstream_{key}"] - metrics[f"pytorch_{key}"]))
                for key in ("mAP", "CP", "CR", "precision", "recall", "f1",
                            "macro_precision", "macro_recall"))
        metrics["pytorch_reference_sha256"] = cache.name
        results.append(metrics_for_backend(metrics, args.backend))
    upstream_settings = build_settings(upstream_build)
    compiled_main = Path(upstream_settings.get("CMAKE_HOME_DIRECTORY",
                                               upstream / "cpp_ggml")) / "src/main.cpp"
    report = {"upstream_revision": revision, "raw_outputs": str(work),
              "model": args.model.name, "backend": args.backend, "threads": args.threads,
              "cpu_precision": "auto",
              "compatibility_cpu_precision": "upstream",
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "warmup": args.warmup, "repeat": args.repeat, "task": args.task,
              "pytorch_reference_device": args.reference_device,
              "timing_protocol": {"order": "ABBA", "resident_model": True,
                  "scope": "decode, preprocess, graph, readback and label selection; excludes load and JSON",
                  "percentiles": "per-image means of repeat forwards from both processes"},
              "gpu_overlap_audit": {"enabled": audited,
                  "graphics_sm_threshold_pct": 5,
                  "idle_timeout_seconds": args.gpu_idle_timeout,
                  "allowed_gpu_pids": sorted(allowed_pids),
                  "wait_process_pids": args.wait_process_pid,
                  "samples": "per-command raw logs include the upstream GPU process audit"},
              "candidate_probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "candidate_runtime_sha256": hashlib.sha256(
                  (args.probe.parents[1] / "libAICore.so").read_bytes()).hexdigest()
                  if (args.probe.parents[1] / "libAICore.so").is_file() else None,
              "candidate_ggml_build_settings": candidate_build_settings(args.probe.parents[2]),
              "upstream_binary_sha256": hashlib.sha256((upstream_build / "ram-ggml").read_bytes()).hexdigest(),
              "upstream_source_sha256": hashlib.sha256((upstream / "cpp_ggml/src/main.cpp").read_bytes()).hexdigest(),
              "upstream_compiled_source": str(compiled_main),
              "upstream_compiled_source_sha256": hashlib.sha256(compiled_main.read_bytes()).hexdigest(),
              "upstream_build_settings": upstream_settings,
              "upstream_dirty_state": subprocess.check_output([
                  "git", "-C", str(upstream), "status", "--porcelain"], text=True).strip(),
              "upstream_library_root": str(upstream_build),
              "datasets": results,
              "images": sum(row["images"] for row in results),
              "pipeline_ms": float(np.mean(candidate_latencies)),
              "upstream_ms": float(np.mean(upstream_latencies)),
              "aggregation": {
                  "latency_means": "image-count weighted across datasets",
                  "percentiles": "all per-image repeat means from both ABBA samples",
                  "score_summaries": "worst dataset; see datasets for each label space"},
              "latency_over_upstream": max(row["latency_over_upstream"] for row in results),
              "upstream_cross_f1": min(row["upstream_cross_f1"] for row in results)}
    report["weighted_latency_over_upstream"] = report["pipeline_ms"] / report["upstream_ms"]
    report["pipeline_images_per_second"] = 1000 / report["pipeline_ms"]
    report["upstream_images_per_second"] = 1000 / report["upstream_ms"]
    for prefix, samples in (("pipeline", candidate_latencies), ("upstream", upstream_latencies)):
        for percentile in (50, 95):
            report[f"{prefix}_p{percentile}_ms"] = float(np.percentile(samples, percentile))
    for key in ("cross_f1", "cross_precision", "cross_recall", "caption_exact",
                "upstream_pytorch_cross_f1", "pytorch_cross_f1_delta", "pytorch_cross_f1_drop",
                "caption_word_f1", "pytorch_caption_exact", "pytorch_caption_word_f1",
                "mAP", "CP", "CR", "pytorch_probability_mae",
                "pytorch_probability_max_error", "upstream_logit_mae", "upstream_logit_max_error",
                "upstream_pytorch_probability_mae", "upstream_pytorch_probability_max_error",
                "pytorch_probability_error_excess", "pytorch_annotation_error_excess",
                "upstream_probability_mae", "upstream_probability_max_error", "upstream_ms", "pipeline_p50_ms",
                "compatibility_upstream_cross_f1", "compatibility_upstream_logit_max_error",
                "compatibility_upstream_probability_max_error",
                "compatibility_pytorch_probability_error_excess", "compatibility_pytorch_cross_f1_drop",
                "compatibility_pytorch_annotation_error_excess",
                "arithmetic_probability_max_error", "arithmetic_probability_mae",
                "pipeline_p95_ms", "upstream_p50_ms", "upstream_p95_ms",
                "precision", "recall", "f1", "macro_precision", "macro_recall",
                "upstream_mAP_delta", "upstream_CP_delta", "upstream_CR_delta",
                "pytorch_mAP_delta", "pytorch_CP_delta", "pytorch_CR_delta"):
        if key in results[0]:
            if key in report:
                continue
            if key.endswith(("_mae", "_delta", "_max_error", "_excess", "_drop")):
                report[key] = max(row[key] for row in results)
            elif key.endswith("_ms"):
                report[key] = float(np.mean([row[key] for row in results]))
            else:
                report[key] = min(row[key] for row in results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RAM {family} {dtype}: {report['images']} images; report: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
