#!/usr/bin/env python3
"""All eight published PicoDet heads against the original Paddle graph.

The common runner owns asset provisioning, process repeats and same-host A/B.
This consumer requires all heads, normal source exits and strict argmax
agreement. Native CPU comparisons are separate diagnostics, not GPU gates.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from run_ocr_onnx_probe import probability_command, source_performance_pass
from run_ocr_paddle_pipeline import verified_source
from run_ocr_paddle_probe import comparison, digest, execute, source_environment


def canonical_heads(values, references):
    if len(values) != 8 or len(references) != 8:
        raise ValueError("all eight layout heads are required")
    shapes = [reference.shape for reference in references]
    if (len(set(shapes)) != 8 or any(len(s) != 3 or s[0] != 1 or
            s[2] not in (10, 32) for s in shapes)):
        raise ValueError("invalid or duplicate layout head shape")
    by_shape = {}
    for value in values:
        if (value.dtype != np.float32 or not value.size or
                not np.isfinite(value).all() or value.shape in by_shape):
            raise ValueError("invalid or duplicate layout output")
        by_shape[value.shape] = value
    if set(by_shape) != set(shapes):
        raise ValueError("layout source/native head shapes differ")
    return [by_shape[shape] for shape in shapes]


def head_comparison(actual, expected):
    actual = canonical_heads(actual, expected)
    metrics = []
    for a, e in zip(actual, expected):
        metric = comparison(a, e, argmax=True)
        if a.shape[-1] == 32:
            # DFL has four independent eight-bin edge distributions. A
            # single argmax over 32 channels can hide a changed edge bin.
            a_bins = a.reshape(*a.shape[:-1], 4, 8)
            e_bins = e.reshape(*e.shape[:-1], 4, 8)
            bin_match = float(np.mean(
                np.argmax(a_bins, axis=-1) == np.argmax(e_bins, axis=-1)))
            metric['global_channel_argmax_match'] = metric['argmax_match']
            metric['dfl_bin_argmax_match'] = bin_match
            metric['argmax_match'] = min(metric['argmax_match'], bin_match)
            def probabilities(logits):
                values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                return values / np.sum(values, axis=-1, keepdims=True)
            probability = comparison(probabilities(a_bins), probabilities(e_bins))
            metric['dfl_probability_max_diff'] = probability['max_diff']
            metric['dfl_probability_mean_diff'] = probability['mean_diff']
            metric['argmax_axis'] = 'four independent eight-bin edges'
        else:
            metric['argmax_axis'] = 'ten classes'
        metrics.append(metric)
    total = sum(a.size for a in actual)
    return dict(max_diff=max(m["max_diff"] for m in metrics),
        mean_diff=sum(m["mean_diff"] * a.size for m, a in zip(metrics, actual)) / total,
        argmax_match=min(m["argmax_match"] for m in metrics), heads=metrics)


def accuracy_pass(metrics, maximum, mean=None):
    return (metrics["max_diff"] < maximum and metrics["argmax_match"] == 1 and
            (mean is None or metrics["mean_diff"] < mean))


def source_run(args, files, python, target, device):
    benchmark = Path(__file__).resolve().parents[1] / "tools/bench_paddle_ocr_graph.py"
    command = [str(python), str(benchmark), "--program", str(files["program"]),
        "--parameters", str(files["parameters"]), "--input", str(args.input),
        "--program-sha256", digest(files["program"]),
        "--parameters-sha256", digest(files["parameters"]),
        "--input-sha256", args.input_sha256, "--device", device,
        "--expected-outputs", "8", "--threads", str(args.threads),
        "--warmups", str(args.warmups if device == "cuda" else 0),
        "--runs", str(args.runs if device == "cuda" else 1),
        "--report", str(target)]
    strict = args.measurement_profile == "release" and device == "cuda"
    if strict:
        command += ["--audit-hold-ms", "1500"]
    _, audit = execute(command, target, environment=source_environment(python),
                       gpu=strict, strict=strict, timeout=args.gpu_idle_timeout)
    with np.load(target.with_suffix(".npz")) as archive:
        if set(archive.files) != {f"head{i}" for i in range(8)}:
            raise ValueError("source did not export every layout head")
        values = [archive[f"head{i}"] for i in range(8)]
    return values, json.loads(target.read_text()), audit


def native_run(args, work, references, *, cpu=False):
    target = args.report.with_suffix(".cpu.json" if cpu else ".native.json")
    strict = not cpu and args.backend != "cpu" and args.measurement_profile == "release"
    completed, audit = execute(probability_command(args, work, cpu=cpu), target,
        gpu=strict, strict=strict, timeout=args.gpu_idle_timeout, require_success=False)
    values, prefix = [], "cpu.bin" if cpu else "native.bin"
    for i, reference in enumerate(references):
        value = np.fromfile(work / f"{prefix}.head{i}.bin", np.float32)
        if value.size != reference.size:
            raise ValueError("native layout head element count mismatch")
        values.append(value.reshape(reference.shape))
    metrics = {}
    for line in completed.stdout.splitlines():
        if line.startswith("{"):
            metrics.update(json.loads(line))
    return values, metrics, audit, completed.returncode


def run(args):
    if digest(args.input) != args.input_sha256:
        raise ValueError("input SHA-256 mismatch")
    references = []
    for path, expected in args.reference_head:
        path = Path(path)
        if digest(path) != expected:
            raise ValueError("layout reference SHA-256 mismatch")
        references.append(np.load(path))
    canonical_heads(references, references)
    pins = json.loads(args.source_pins.read_text())
    files = verified_source(args.source_root, pins, "layout", args.model)
    scripts = Path(__file__).resolve().parent
    dependencies = [Path(__file__), scripts / "run_ocr_paddle_probe.py",
        scripts / "ocr_paddle_environment.py",
        scripts / "ocr_paddle_metadata.py",
        scripts / "run_ocr_paddle_pipeline.py", scripts / "run_ocr_onnx_probe.py",
        scripts / "gpu_audit.py", scripts.parent / "tools/bench_paddle_ocr_graph.py",
        args.source_pins, args.probe]
    before = {str(p): digest(p) for p in dependencies}
    report = dict(exit_code=0, role="layout", backend=args.backend,
        model_sha256=digest(args.model), input_sha256=args.input_sha256,
        reference_sha256=[sha for _, sha in args.reference_head], dependencies=before,
        source_files={key: digest(value) for key, value in files.items()},
        scope="all eight original graph heads; no page-accuracy or full-release claim")
    gpu = args.backend != "cpu"
    controlled = gpu and args.measurement_profile == "release"
    python = args.source_gpu_python if gpu else args.source_python
    if not python:
        raise ValueError("requested backend requires its original source interpreter")
    source, source_metadata, source_audit = source_run(args, files, python,
        args.report.with_suffix(".source-cuda.json" if gpu else ".source-cpu.json"),
        "cuda" if gpu else "cpu")
    source = canonical_heads(source, references)
    report["accuracy_reference"] = "original Paddle CUDA" if gpu else "original Paddle CPU diagnostic"
    report["source_cuda_fixture_comparison" if gpu else "source_cpu_fixture"] = head_comparison(source, references)
    report["source_gpu_environment" if gpu else "source_cpu_environment"] = source_metadata
    if not gpu and not accuracy_pass(report["source_cpu_fixture"], args.source_max_diff):
        report["exit_code"] = 1
    with tempfile.TemporaryDirectory(prefix="paddle-layout-", dir=args.report.parent) as directory:
        work = Path(directory)
        np.save(work / "layout_input.npy", np.load(args.input))
        for i, value in enumerate(source):
            np.save(work / f"layout_head{i}.npy", value)
        native, metrics, audit, code = native_run(args, work, source)
        report.update(metrics)
        report.update(head_comparison(native, source))
        if code or not accuracy_pass(report, args.max_diff, args.mean_diff):
            report["exit_code"] = 1
        report["native_cpu_comparison"] = "separate diagnostic; not a GPU release gate"
        output = args.report.with_suffix(".probabilities.npz")
        np.savez(output, **{f"head{i}": value for i, value in enumerate(native)})
        report["output_sha256"] = digest(output)
        if audit:
            report.update(gpu_audit_pass=int(audit["passed"]),
                gpu_audit_archive=audit["archive_path"],
                gpu_sampled_process_peak_mib=audit["sampled_process_peak_mib"])
    if gpu:
        metrics = head_comparison(native, source)
        report["source_gpu_comparison"] = metrics
        report.update(source_gpu_audit_pass=int(bool(source_audit and source_audit["passed"])),
            source_gpu_audit_archive=source_audit["archive_path"] if source_audit else None,
            source_gpu_sampled_process_peak_mib=source_audit["sampled_process_peak_mib"] if source_audit else None,
            source_performance_pass=int(source_performance_pass(report, source_metadata)),
            source_performance_qualified=int(controlled and bool(source_audit and source_audit["passed"])),
            graph_throughput_hz=1000 / report["p50_ms"])
        if controlled and not report["source_performance_pass"]:
            report["exit_code"] = 1
    if before != {str(p): digest(p) for p in dependencies}:
        raise ValueError("layout probe dependencies changed during execution")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-pins", "source-python", "model", "input", "probe", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--reference-head", nargs=2, action="append", required=True)
    parser.add_argument("--source-gpu-python", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--measurement-profile", choices=("developer", "release"), default="developer")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-diff", type=float, required=True)
    parser.add_argument("--mean-diff", type=float, required=True)
    parser.add_argument("--source-max-diff", type=float, default=.001)
    parser.add_argument("--gpu-idle-timeout", type=float, default=60)
    args = parser.parse_args()
    args.role = "layout"
    if args.threads == 0:
        args.threads = 8
    if len(args.reference_head) != 8 or args.threads < 1 or args.warmups < 0 or args.runs < 1:
        parser.error("require eight heads and valid sampling counts")
    if args.measurement_profile == "release":
        if args.warmups < 2 or args.runs < 10:
            parser.error("release requires >=2 warmups and >=10 forwards")
        if args.backend != "cpu" and not args.source_gpu_python:
            parser.error("release GPU measurement requires original Paddle GPU")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report = dict(exit_code=1, error=str(error))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Paddle layout graph report: {args.report}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
