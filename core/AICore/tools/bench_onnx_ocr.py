#!/usr/bin/env python3
"""Benchmark one pinned source graph; invoked by the common OCR probe.

CUDA input and output stay on device during timing, matching the native
graph timer's upload/readback boundary. CPU timing includes ORT dispatch.
This is one input, not a dataset or matrix runner.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--audit-hold-ms", type=int, default=0,
                        help="retain allocations after timing for driver sampling")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if (args.warmups < 0 or args.runs < 1 or args.threads < 1 or
            not 0 <= args.audit_hold_ms <= 10000):
        parser.error("invalid sampling or thread count")
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != args.source_sha256:
        parser.error("source digest mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
    providers = ([('CUDAExecutionProvider', {'use_tf32': '0'})]
                 if args.provider == "cuda" else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(args.source), options, providers=providers)
    expected_provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    if session.get_providers()[0] != expected_provider:
        raise RuntimeError("source benchmark provider silently fell back")
    x = np.load(args.input)
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid source benchmark input")
    if args.provider == "cuda":
        device_input = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        binding = session.io_binding()
        binding.bind_ortvalue_input(session.get_inputs()[0].name, device_input)
        for output in session.get_outputs():
            binding.bind_output(output.name, "cuda", 0)

        def forward():
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
    else:
        def forward():
            return session.run(None, {session.get_inputs()[0].name: x})
    times = []
    for index in range(-args.warmups, args.runs):
        begin = time.perf_counter()
        y = forward()
        elapsed = (time.perf_counter() - begin) * 1000
        if index >= 0:
            times.append(elapsed)
    if args.provider == "cuda":
        y = binding.copy_outputs_to_cpu()
    if len(y) != 1 or not np.isfinite(y[0]).all():
        raise RuntimeError("invalid source benchmark output")
    np.save(args.report.with_suffix(".npy"), y[0])
    # Outside the timed region; retain the session and device bindings.
    time.sleep(args.audit_hold_ms / 1000)
    times.sort()
    report = {"source_sha256": args.source_sha256,
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "input_shape": list(x.shape), "onnxruntime": ort.__version__,
              "providers": session.get_providers(), "provider_options": session.get_provider_options(),
              "warmups": args.warmups, "runs": args.runs, "threads": args.threads,
              "audit_hold_ms": args.audit_hold_ms,
              "p50_ms": times[len(times) // 2], "p95_ms": times[int(len(times) * .95)],
              "forward_ms": times,
              "timing_scope": "synchronous graph and dispatch; excludes input upload and output readback"}
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
