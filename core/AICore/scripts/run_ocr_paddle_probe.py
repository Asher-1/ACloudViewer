#!/usr/bin/env python3
"""One published OCR graph against its original Paddle program and weights.

validate_all.py owns the catalog, downloads, repeats and baseline comparison.
This probe owns a single verified tensor, original-source comparison and the
optional independently audited source CUDA timing. It makes no page-level claim.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from gpu_audit import run_audited
from ocr_paddle_environment import source_environment
from run_ocr_onnx_probe import (native_exit_is_diagnostic, probability_command,
                                source_performance_pass)
from run_ocr_paddle_pipeline import verified_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(actual, expected, *, argmax=False):
    import numpy as np
    if (actual.shape != expected.shape or not actual.size or
            not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("invalid probability shape or non-finite values")
    delta = np.abs(actual.astype(np.float64) - expected)
    result = dict(max_diff=float(delta.max()), mean_diff=float(delta.mean()))
    if argmax:
        result["argmax_match"] = float(np.mean(actual.argmax(-1) == expected.argmax(-1)))
    return result


def passes(metrics, maximum, mean=None, *, argmax=False, quantized=False):
    values = [metrics["max_diff"], metrics["mean_diff"]]
    if argmax and "argmax_match" in metrics:
        values.append(metrics["argmax_match"])
    if not all(math.isfinite(value) for value in values):
        return False
    if quantized:
        return True
    return (metrics["max_diff"] < maximum and
            (mean is None or metrics["mean_diff"] < mean) and
            (not argmax or metrics.get("argmax_match") == 1))


def execute(command, report, *, environment=None, gpu=False, strict=False, timeout=60,
            require_success=True):
    audit = None
    if gpu:
        completed, audit = run_audited(command, report.with_suffix(".gpu.json"),
            strict=strict, idle_timeout=timeout, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    report.with_suffix(".log").write_text(completed.stdout + completed.stderr)
    if completed.returncode and require_success:
        raise RuntimeError(f"probe process failed ({completed.returncode}): {report.with_suffix('.log')}")
    if gpu and strict and not audit["passed"]:
        raise RuntimeError(f"GPU measurement audit rejected: {report.with_suffix('.gpu.json')}")
    return completed, audit


def source_command(args, files, python, report, device):
    tool = Path(__file__).resolve().parents[1] / "tools/bench_paddle_ocr_graph.py"
    command = [str(python), str(tool), "--program", str(files["program"]),
        "--parameters", str(files["parameters"]), "--input", str(args.input),
        "--program-sha256", digest(files["program"]),
        "--parameters-sha256", digest(files["parameters"]),
        "--input-sha256", args.input_sha256, "--device", device,
        "--warmups", str(args.warmups if device == "cuda" else 0),
        "--runs", str(args.runs if device == "cuda" else 1),
        "--threads", str(args.threads), "--report", str(report)]
    if device == "cuda":
        command += ["--audit-hold-ms", "1500"]
    return command


def run(args):
    import numpy as np
    for name in ("input", "reference"):
        if digest(getattr(args, name)) != getattr(args, name + "_sha256"):
            raise ValueError(f"{name} SHA-256 mismatch")
    quantized = args.model.name.endswith(("-f16.gguf", "-q8_0.gguf"))
    pins = json.loads(args.source_pins.read_text())
    files = verified_source(args.source_root, pins, args.source_role, args.model)
    report = dict(model_sha256=digest(args.model), input_sha256=args.input_sha256,
        reference_sha256=args.reference_sha256, source_pins_sha256=digest(args.source_pins),
        source_files={key: digest(files[key]) for key in ("program", "parameters")},
        backend=args.backend, role=args.role, exit_code=0,
        scope="one original Paddle graph tensor; page accuracy is a separate gate",
        sampling=dict(warmups=args.warmups, runs=args.runs, threads=args.threads))
    benchmark = Path(__file__).resolve().parents[1] / "tools/bench_paddle_ocr_graph.py"
    report["source_benchmark_sha256"] = digest(benchmark)
    identity_module = Path(__file__).resolve().with_name("ocr_paddle_metadata.py")
    report["distribution_identity_sha256"] = digest(identity_module)
    gpu = args.backend != "cpu"
    controlled = gpu and args.measurement_profile == "release"
    python = args.source_gpu_python if gpu else args.source_python
    if not python:
        raise ValueError("requested backend requires its original source interpreter")
    device = "cuda" if gpu else "cpu"
    source_report = args.report.with_suffix(".source-cuda.json" if gpu else ".source-cpu.json")
    _, source_audit = execute(source_command(args, files, python, source_report, device),
        source_report, environment=source_environment(python), gpu=gpu,
        strict=controlled, timeout=args.gpu_idle_timeout)
    truth = np.load(args.reference)
    live_source = np.load(source_report.with_suffix(".npy"))
    fixture_metrics = comparison(live_source, truth, argmax=args.role == "rec")
    report["accuracy_reference"] = "original Paddle CUDA" if gpu else "original Paddle CPU diagnostic"
    report["source_cuda_fixture_comparison" if gpu else "source_cpu_fixture"] = fixture_metrics
    # The pinned CPU fixture remains an identity/shape anchor. GPU numerical
    # qualification uses the original CUDA graph, including near-tie argmax.
    if not gpu and not passes(fixture_metrics, args.source_max_diff,
                              argmax=args.role == "rec", quantized=quantized):
        raise ValueError("original Paddle CPU output differs from pinned fixture")
    source_identity = json.loads(source_report.read_text())
    if source_identity["distribution_identity_sha256"] != report["distribution_identity_sha256"]:
        raise ValueError("source package identity helper mismatch")
    report["source_cuda_report_sha256" if gpu else "source_cpu_report_sha256"] = digest(source_report)
    report["source_gpu_environment" if gpu else "source_cpu_environment"] = source_identity["environment"]
    report["source_gpu_distribution_candidates" if gpu else "source_cpu_distribution_candidates"] = source_identity["distribution_candidates"]
    with tempfile.TemporaryDirectory(prefix="paddle-graph-", dir=args.report.parent) as directory:
        work = Path(directory)
        shutil.copyfile(args.input, work / f"{args.source_role}_input.npy")
        # Supply the selected original CUDA output to the C++ probe. The Python
        # runner applies the dtype-specific output policy below.
        np.save(work / f"{args.source_role}_reference.npy", live_source)
        completed, audit = execute(probability_command(args, work), args.report.with_suffix(".native.json"),
            gpu=controlled, strict=controlled, timeout=args.gpu_idle_timeout, require_success=False)
        report["exit_code"] = int(completed.returncode != 0 and not quantized)
        for line in completed.stdout.splitlines():
            if line.startswith("{"):
                report.update(json.loads(line))
        native_path = work / "native.bin"
        native = np.fromfile(native_path, np.float32)
        if native.size != live_source.size:
            raise ValueError("native probability element count mismatch")
        native = native.reshape(live_source.shape)
        outputs_complete = bool(np.isfinite(native).all())
        report["native_outputs_complete"] = int(outputs_complete)
        if not native_exit_is_diagnostic(completed.returncode,
                quantized=quantized, outputs_complete=outputs_complete):
            report["exit_code"] = 1
            report["native_output_error"] = "invalid native probability values"
        else:
            report["native_raw_threshold_exit_diagnostic"] = int(
                completed.returncode != 0 and quantized)
        metrics = comparison(native, live_source, argmax=args.role == "rec")
        report.update(metrics)
        if not passes(metrics, args.max_diff, args.mean_diff,
                      argmax=args.role == "rec", quantized=quantized):
            report["exit_code"] = 1
        output = args.report.with_suffix(".probabilities.npy")
        np.save(output, native)
        report.update(output_sha256=digest(output), probability_path=str(output))
        report["native_cpu_comparison"] = "separate diagnostic; not a GPU release gate"
        if audit:
            report.update(gpu_audit_pass=int(audit["passed"]),
                gpu_audit_archive=audit["archive_path"],
                gpu_sampled_process_peak_mib=audit["sampled_process_peak_mib"])
    if gpu:
        report.update({"source_gpu_" + key: value for key, value in metrics.items()})
        report.update(source_gpu_audit_pass=int(source_audit["passed"]),
            source_gpu_audit_archive=source_audit["archive_path"],
            source_gpu_sampled_process_peak_mib=source_audit["sampled_process_peak_mib"],
            source_p50_ms=source_identity["p50_ms"], source_p95_ms=source_identity["p95_ms"],
            source_performance_pass=int(source_performance_pass(report, source_identity)),
            source_performance_qualified=int(controlled and source_audit["passed"]),
            source_performance_gate=dict(threshold_pct=5, absolute_noise_floor_ms=3),
            graph_throughput_hz=1000 / report["p50_ms"])
        if controlled and not report["source_performance_pass"]:
            report["exit_code"] = 1
    if digest(benchmark) != report["source_benchmark_sha256"]:
        raise ValueError("original Paddle benchmark changed during execution")
    if digest(identity_module) != report["distribution_identity_sha256"]:
        raise ValueError("source package identity helper changed during execution")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-pins", "source-python", "model", "input",
                 "reference", "probe", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("input-sha256", "reference-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source-gpu-python", type=Path)
    parser.add_argument("--source-role", choices=("det_ch", "rec_ch", "rec_en"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--measurement-profile", choices=("developer", "release"), default="developer")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-diff", type=float, required=True)
    parser.add_argument("--mean-diff", type=float)
    parser.add_argument("--source-max-diff", type=float, default=.001)
    parser.add_argument("--gpu-idle-timeout", type=float, default=60)
    args = parser.parse_args()
    args.role = "det" if args.source_role == "det_ch" else "rec"
    if args.threads == 0:
        args.threads = 8
    if args.threads < 1 or args.warmups < 0 or args.runs < 1:
        parser.error("invalid sampling counts")
    if args.measurement_profile == "release":
        if args.warmups < 2 or args.runs < 10:
            parser.error("release measurement requires >=2 warmups and >=10 forwards")
        if args.backend != "cpu" and not args.source_gpu_python:
            parser.error("release GPU measurement requires the original Paddle GPU interpreter")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report = dict(exit_code=1, error=str(error))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Paddle graph report: {args.report}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
