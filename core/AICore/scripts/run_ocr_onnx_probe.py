#!/usr/bin/env python3
"""One independent-source OCR probe, scheduled by validate_all.py.

No download, catalog or matrix ownership here. The common runner supplies
verified assets; generated truth stays in the report's build directory.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from gpu_audit import run_audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--extra-image", type=Path, action="append", default=[],
                        help="additional accuracy inputs; timing remains the primary input's timing")
    parser.add_argument("--extra-crop", nargs=9, action="append", default=[],
                        metavar="PAGE_OR_COORDINATE",
                        help="additional recognizer input: page path followed by eight quad coordinates")
    parser.add_argument("--include-rotated", action="store_true",
                        help="also verify every classifier image upside down")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--role", choices=("det", "rec", "cls", "orientation"), required=True)
    parser.add_argument("--det", type=Path)
    parser.add_argument("--cls", type=Path)
    parser.add_argument("--rec-source", type=Path)
    parser.add_argument("--rec-source-sha256")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--dictionary-sha256")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-benchmark-python", type=Path,
                        help="isolated source CUDA environment; required for release GPU throughput")
    parser.add_argument("--source-benchmark-root", type=Path,
                        help="isolated source CUDA virtual environment directory")
    parser.add_argument("--measurement-profile", choices=("developer", "release"),
                        default="developer")
    parser.add_argument("--gpu-idle-timeout", type=float, default=60)
    parser.add_argument("--max-side", type=int,
                        help="detector limit: developer default 320; release requires source default 960")
    parser.add_argument("--max-diff", type=float, default=.001,
                        help="source probability tolerance, owned by the manifest")
    parser.add_argument("--stored-max-diff", type=float, default=.001,
                        help="native versus independent identical-weight ONNX tolerance")
    args = parser.parse_args()
    if args.source_benchmark_root:
        args.source_benchmark_python = args.source_benchmark_root / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.threads < 0:
        parser.error("thread count must be nonnegative")
    args.threads = args.threads or 8
    try:
        args.max_side = detector_input_limit(args.measurement_profile, args.role, args.max_side)
    except ValueError as error:
        parser.error(str(error))
    if args.measurement_profile == "release" and not (args.warmups >= 2 and args.runs >= 10):
        parser.error("OCR release measurement requires >=2 warmups and >=10 timed forwards")
    if args.measurement_profile == "release" and args.backend in ("cuda", "vulkan"):
        if not args.source_benchmark_python or not args.source_benchmark_python.is_file():
            parser.error("release GPU throughput requires --source-benchmark-python with a working source CUDA provider")
    inputs = [(path, rotation, None) for path, rotation in accuracy_inputs(args)]
    try:
        inputs += accuracy_crops(args)
    except ValueError as error:
        parser.error(str(error))
    if len(inputs) == 1:
        return run_probe(args)
    if args.role == "orientation":
        parser.error("orientation already owns its upright/rotated fixture set")
    fixtures, codes = [], []
    for index, (path, rotate180, quad) in enumerate(inputs):
        fixture = copy.copy(args)
        fixture.image = path
        fixture.rotate180 = rotate180
        fixture.quad = quad
        fixture.report = args.report.with_name(f"{args.report.stem}-fixture-{index}.json")
        if index:
            fixture.warmups, fixture.runs = 0, 1
            fixture.measurement_profile = "developer"
            # Extra inputs remain independent-source accuracy fixtures; the
            # provider benchmark and telemetry belong to the primary tensor.
            # Keep CUDA accuracy truth for every extra fixture. Only the
            # primary fixture owns controlled performance qualification.
        code = run_probe(fixture)
        codes.append(code)
        if fixture.report.is_file():
            fixtures.append(json.loads(fixture.report.read_text()))
        else:
            fixtures.append({"exit_code": code or 1, "error": "accuracy fixture produced no report"})
    report = aggregate_fixtures(fixtures, codes)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"OCR multi-input report: {args.report}")
    return report["exit_code"]


def detector_input_limit(profile, role, requested=None):
    limit = requested if requested is not None else (960 if profile == "release" and role == "det" else 320)
    if limit < 32:
        raise ValueError("detector input limit must be at least 32")
    if profile == "release" and role == "det" and limit != 960:
        raise ValueError("detector release measurement requires the source default limit 960")
    return limit


def accuracy_inputs(args):
    if args.include_rotated and args.role != "cls":
        raise ValueError("rotated probability fixtures require the classifier role")
    paths = [args.image, *args.extra_image]
    return [(path, False) for path in paths] + (
        [(path, True) for path in paths] if args.include_rotated else [])


def accuracy_crops(args):
    crops = getattr(args, "extra_crop", [])
    if crops and args.role != "rec":
        raise ValueError("quadrilateral fixtures require the recognizer role")
    result = []
    for entry in crops:
        if len(entry) != 9:
            raise ValueError("crop requires a page and eight coordinates")
        quad = [float(value) for value in entry[1:]]
        if not all(math.isfinite(value) for value in quad):
            raise ValueError("crop coordinates must be finite")
        result.append((Path(entry[0]), False, quad))
    return result


def aggregate_fixtures(fixtures, codes):
    """Keep primary timing and require every accuracy input to pass."""
    if not fixtures or len(fixtures) != len(codes):
        raise ValueError("incomplete OCR accuracy fixture results")
    report = dict(fixtures[0])
    report["fixtures"] = [{k: v for k, v in f.items() if not k.endswith("_ms")}
                          for f in fixtures]
    report["accuracy_fixture_count"] = len(fixtures)
    report["timing_scope"] = "primary fixture; additional fixtures have zero warmups and one accuracy forward"
    report["exit_code"] = int(any(codes) or any(f.get("exit_code", 1) for f in fixtures))
    for key in ("max_diff", "mean_diff", "stored_max_diff", "stored_mean_diff"):
        if all(key in f and math.isfinite(f[key]) for f in fixtures):
            report[key] = max(f[key] for f in fixtures)
        else:
            report.pop(key, None)
            report["exit_code"] = 1
    if report.get("role") != "det":
        for key in ("argmax_match", "stored_argmax_match"):
            if all(key in f and math.isfinite(f[key]) for f in fixtures):
                report[key] = min(f[key] for f in fixtures)
            else:
                report.pop(key, None)
                report["exit_code"] = 1
    hashes = [f.get("output_sha256") for f in fixtures]
    if all(hashes):
        report["output_sha256"] = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    else:
        report.pop("output_sha256", None)
        report["exit_code"] = 1
    return report


def probability_command(args, work, *, cpu=False):
    backend = "cpu" if cpu else args.backend
    hold_ms = 1500 if (backend in ("cuda", "vulkan") and
                       args.measurement_profile == "release") else 0
    return [str(args.probe), str(work), str(args.model), args.role, backend, "-1",
            f"dump={work / ('cpu.bin' if cpu else 'native.bin')}",
            str(args.max_diff), str(0 if cpu else args.warmups),
            str(1 if cpu else args.runs), str(args.threads), str(hold_ms)]


def source_performance_pass(native, source, threshold_pct=5., noise_floor_ms=3.):
    """Use the common runner's relative AND absolute regression rule."""
    for key in ("p50_ms", "p95_ms"):
        current, baseline = native.get(key), source.get(key)
        if (current is None or baseline is None or
                not math.isfinite(current) or not math.isfinite(baseline) or
                current <= 0 or baseline <= 0):
            return False
        if (current > baseline * (1 + threshold_pct / 100) and
                current - baseline > noise_floor_ms):
            return False
    return True


def source_environment(python):
    environment = os.environ.copy()
    for key in ("LD_PRELOAD", "PYTHONPATH"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    root = python.absolute().parent.parent
    candidates = list(root.glob("lib/python*/site-packages/nvidia/*/lib"))
    candidates += list(root.glob("Lib/site-packages/nvidia/*/bin"))
    candidates += list(root.glob("Lib/site-packages/nvidia/*/lib"))
    variable = "PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    environment[variable] = os.pathsep.join([str(p) for p in candidates] + [environment.get(variable, "")])
    return environment


def run_probe(args):
    import numpy as np

    tools = Path(__file__).resolve().parents[1] / "tools"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ocr-onnx-", dir=args.report.parent) as directory:
        work = Path(directory)
        orientation = args.role == "orientation"
        exporter = "export_ocr_orientation_fixture.py" if orientation else "export_onnx_ocr_fixture.py"
        gpu_reference = args.backend in ("cuda", "vulkan")
        if gpu_reference and (not args.source_benchmark_python or
                              not args.source_benchmark_python.is_file()):
            raise ValueError("GPU accuracy requires a working original-source CUDA interpreter")
        source_python = args.source_benchmark_python if gpu_reference else Path(sys.executable)
        command = [str(source_python), str(tools / exporter), str(args.source), str(args.image),
                   str(work), "--source-sha256", args.source_sha256]
        if not orientation:
            command += ["--role", args.role, "--max-side", str(args.max_side),
                        "--weights", str(args.model),
                        "--provider", "cuda" if gpu_reference else "cpu",
                        "--threads", str(args.threads)]
            if getattr(args, "rotate180", False): command += ["--rotate180"]
            if getattr(args, "quad", None) is not None:
                command += ["--quad", *map(str, args.quad)]
        elif args.rec_source:
            if not all((args.rec_source, args.rec_source_sha256,
                        args.dictionary, args.dictionary_sha256)):
                raise ValueError("orientation requires pinned recognizer source and dictionary")
            command += ["--rec-source", str(args.rec_source),
                        "--rec-source-sha256", args.rec_source_sha256,
                        "--dictionary", str(args.dictionary),
                        "--dictionary-sha256", args.dictionary_sha256]
        if orientation:
            command += ["--provider", "cuda" if gpu_reference else "cpu",
                        "--threads", str(args.threads)]
        exported = subprocess.run(command, capture_output=True, text=True,
                                  env=source_environment(source_python) if gpu_reference else None)
        if exported.returncode:
            print(exported.stdout + exported.stderr, file=sys.stderr)
            return exported.returncode
        if orientation:
            if not args.det or not args.cls:
                raise ValueError("orientation requires --det and --cls")
            command = [str(args.probe), str(args.det), str(args.model), str(args.cls),
                       str(work / "orientation.png"), str(work / "orientation.json"), args.backend]
        else:
            command = probability_command(args, work)
        audit = None
        if not orientation and args.backend in ("cuda", "vulkan") and args.measurement_profile == "release":
            audit_path = args.report.with_suffix(".gpu.json")
            try:
                completed, audit = run_audited(
                    command, audit_path, strict=True,
                    idle_timeout=args.gpu_idle_timeout,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except RuntimeError as error:
                rejected_audit = json.loads(audit_path.read_text())
                args.report.write_text(json.dumps({"exit_code": 1, "gpu_audit_pass": 0,
                    "error": str(error), "audit_path": str(audit_path),
                    "gpu_audit_id": rejected_audit["audit_id"],
                    "gpu_audit_archive": rejected_audit["archive_path"]}, indent=2) + "\n")
                print(str(error), file=sys.stderr)
                return 1
        else:
            completed = subprocess.run(command, capture_output=True, text=True)
        args.report.with_suffix(".log").write_text(completed.stdout + completed.stderr)
        metrics = {}
        for line in completed.stdout.splitlines():
            if line.startswith("{"):
                try: metrics.update(json.loads(line))
                except ValueError: pass
        # The common generic parser names top-level JSON scalars jsonN/<key>.
        # Keep native accuracy values at that level for its suffix resolver.
        report = {**metrics, "role": args.role, "backend": args.backend,
                  "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  "source_sha256": args.source_sha256, "exit_code": completed.returncode,
                  "scope": "real word/page graph fixture; not dataset accuracy or release evidence"}
        report["input_profile"] = {"detector_max_side": args.max_side if args.role == "det" else None,
                                   "shape_source": "reference/input_shape"}
        report["sampling"] = ({"classifier_threshold_count": 2, "calls_per_context": 2}
                              if orientation else {"warmups": args.warmups, "runs": args.runs})
        if audit is not None:
            report.update(gpu_audit_pass=int(audit["passed"]),
                          gpu_audit_id=audit["audit_id"],
                          gpu_audit_archive=audit["archive_path"],
                          gpu_sampled_process_peak_mib=audit["sampled_process_peak_mib"],
                          gpu_sampled_device_peak_mib=audit["sampled_device_peak_mib"],
                          gpu_audit_path=str(args.report.with_suffix(".gpu.json")))
            if not audit["passed"]:
                report["exit_code"] = 1
        if not orientation:
            report["reference"] = json.loads((work / "reference.json").read_text())
            report["accuracy_reference"] = "original and stored-weight ONNX CUDA" if gpu_reference else "ONNX CPU diagnostic"
            stem = {"det": "det_ch", "rec": "rec_ch", "cls": "cls"}[args.role]
            output = work / "native.bin"
            if output.is_file():
                native = np.fromfile(output, np.float32)
                source = np.load(work / f"{stem}_reference.npy")
                stored = np.load(work / f"{stem}_stored_reference.npy")
                if native.size != stored.size or not np.isfinite(native).all():
                    report["exit_code"] = 1
                    report["output_error"] = "invalid native probability shape or values"
                else:
                    native = native.reshape(source.shape)
                    probability_path = args.report.with_suffix(".probabilities.npy")
                    np.save(probability_path, native)
                    report["probability_path"] = str(probability_path)
                    delta = np.abs(native.astype(np.float64) - stored)
                    report["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                    report["max_diff"] = float(np.abs(native.astype(np.float64) - source).max())
                    report["mean_diff"] = float(np.abs(native.astype(np.float64) - source).mean())
                    report["stored_max_diff"] = float(delta.max())
                    report["stored_mean_diff"] = float(delta.mean())
                    if args.role != "det":
                        report["stored_argmax_match"] = float(np.mean(native.argmax(-1) == stored.argmax(-1)))
                        report["argmax_match"] = float(np.mean(native.argmax(-1) == source.argmax(-1)))
                    if (report["max_diff"] >= args.max_diff or
                            report["stored_max_diff"] >= args.stored_max_diff):
                        report["exit_code"] = 1
                    if args.role != "det" and (report["argmax_match"] != 1 or
                                                report["stored_argmax_match"] != 1):
                        report["exit_code"] = 1
                    # Native CPU parity is a separate diagnostic cohort. GPU
                    # qualification uses the independent source and stored
                    # weights, never another native backend as its truth.
                    report["native_cpu_comparison"] = "separate diagnostic; not a GPU release gate"
            else:
                report["exit_code"] = completed.returncode or 1
                report["output_error"] = "native probe did not export probabilities"
        else:
            report["reference"] = json.loads((work / "orientation.json").read_text())
            report["accuracy_reference"] = "original ONNX CUDA orientation and CTC" if gpu_reference else "ONNX CPU diagnostic"
        if args.source_benchmark_python and not orientation and args.backend in ("cuda", "vulkan"):
            controlled = args.measurement_profile == "release"
            source_report = args.report.with_suffix(".source.json")
            stem = {"det": "det_ch", "rec": "rec_ch", "cls": "cls"}[args.role]
            benchmark = [str(args.source_benchmark_python), str(tools / "bench_onnx_ocr.py"),
                         "--source", str(args.source), "--source-sha256", args.source_sha256,
                         "--input", str(work / f"{stem}_input.npy"), "--provider", "cuda",
                         "--warmups", str(args.warmups), "--runs", str(args.runs),
                         "--audit-hold-ms", "1500" if controlled else "0",
                         "--threads", str(args.threads), "--report", str(source_report)]
            # Keep the source provider's cuDNN separate from the application's
            # ggml dependencies. The benchmark environment is explicitly supplied.
            environment = source_environment(args.source_benchmark_python)
            try:
                source_run, source_audit = run_audited(
                    benchmark, args.report.with_suffix(".source.gpu.json"), strict=controlled,
                    idle_timeout=args.gpu_idle_timeout, env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                args.report.with_suffix(".source.log").write_text(source_run.stdout + source_run.stderr)
                report["source_gpu_audit_pass"] = int(source_audit["passed"])
                report["source_gpu_audit_id"] = source_audit["audit_id"]
                report["source_gpu_audit_archive"] = source_audit["archive_path"]
                report["source_gpu_sampled_process_peak_mib"] = source_audit["sampled_process_peak_mib"]
                if source_run.returncode or (controlled and not source_audit["passed"]):
                    report["exit_code"] = 1
                if source_report.is_file():
                    timing = json.loads(source_report.read_text())
                    report["source_p50_ms"], report["source_p95_ms"] = timing["p50_ms"], timing["p95_ms"]
                    report["native_source_latency_ratio"] = report["p50_ms"] / timing["p50_ms"]
                    report["source_performance_gate"] = {
                        "threshold_pct": 5, "absolute_noise_floor_ms": 3,
                        "comparison": "native versus original-source CUDA graph p50 and p95"}
                    report["source_performance_pass"] = int(source_performance_pass(report, timing))
                    report["source_performance_qualified"] = int(controlled and source_audit["passed"])
                    if controlled and not report["source_performance_pass"]:
                        report["exit_code"] = 1
                    report["graph_throughput_hz"] = 1000 / report["p50_ms"] if report["p50_ms"] > 0 else 0
                    source_output = np.load(source_report.with_suffix(".npy"))
                    source_reference = np.load(work / f"{stem}_reference.npy")
                    if (source_output.shape != source_reference.shape or
                            not np.isfinite(source_output).all() or
                            not output.is_file() or
                            native.shape != source_output.shape or
                            not np.isfinite(native).all()):
                        report["exit_code"] = 1
                        report["source_gpu_error"] = "invalid native/source CUDA probability shape or values"
                    else:
                        # Both accuracy and timing references use original
                        # CUDA. Keep any dispatch difference visible.
                        report["source_accuracy_benchmark_max_diff"] = float(np.abs(
                            source_output.astype(np.float64) - source_reference).max())
                        report["source_gpu_max_diff"] = float(np.abs(
                            native.astype(np.float64) - source_output).max())
                        report["source_gpu_mean_diff"] = float(np.abs(
                            native.astype(np.float64) - source_output).mean())
                        if report["source_gpu_max_diff"] >= args.max_diff:
                            report["exit_code"] = 1
                        if args.role != "det":
                            report["source_gpu_argmax_match"] = float(np.mean(
                                native.argmax(-1) == source_output.argmax(-1)))
                            if report["source_gpu_argmax_match"] != 1:
                                report["exit_code"] = 1
                else:
                    report["exit_code"] = 1
                    report["source_gpu_error"] = "source provider produced no benchmark report"
            except RuntimeError as error:
                rejected_audit = json.loads(args.report.with_suffix(".source.gpu.json").read_text())
                report["source_gpu_audit_pass"] = 0
                report["source_gpu_audit_id"] = rejected_audit["audit_id"]
                report["source_gpu_audit_archive"] = rejected_audit["archive_path"]
                report["source_gpu_error"] = str(error)
                report["exit_code"] = 1
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        # validate_all reads this file itself; emitting it twice makes leaf
        # metric names ambiguous in its generic parser.
        print(f"OCR probe report: {args.report}")
        return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
