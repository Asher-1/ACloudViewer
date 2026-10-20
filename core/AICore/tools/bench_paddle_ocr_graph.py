#!/usr/bin/env python3
"""One original Paddle graph with a preverified input; no asset downloads.

The predictor keeps input/output allocations during timed forwards. Upload,
output copies, model loading, and the telemetry allocation hold are excluded.
The parent probe verifies source identity and audits this separate process.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ocr_paddle_metadata import distribution_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    identity_module = Path(distribution_environment.__code__.co_filename)
    identity_sha256 = digest(identity_module)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "parameters", "input", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("program-sha256", "parameters-sha256", "input-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--audit-hold-ms", type=int, default=0)
    parser.add_argument("--expected-outputs", type=int, choices=(1, 2, 8), default=1)
    args = parser.parse_args()
    if (args.warmups < 0 or args.runs < 1 or args.threads < 1 or
            not 0 <= args.audit_hold_ms <= 10000):
        parser.error("invalid measurement counts")
    for name in ("program", "parameters", "input"):
        if digest(getattr(args, name)) != getattr(args, name + "_sha256"):
            parser.error(f"{name} SHA-256 mismatch")
    distribution = "paddlepaddle-gpu" if args.device == "cuda" else "paddlepaddle"
    if importlib.metadata.version(distribution) != "3.0.0":
        parser.error("the independent Paddle reference requires version 3.0.0")
    # Apply the same FP32 math policy to both source-reference devices.
    # Paddle 3.0.0 has no allow_tf32 flags. NVIDIA's documented override
    # applies to both cuBLAS and cuDNN before either library is initialized.
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import paddle
    from paddle.inference import Config, create_predictor
    config = Config(str(args.program), str(args.parameters))
    if args.device == "cuda":
        config.enable_use_gpu(256, 0)
    else:
        config.disable_gpu()
    config.switch_ir_optim(False)
    config.disable_mkldnn()
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(args.threads)
    predictor = create_predictor(config)
    x = np.load(args.input)
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid input tensor")
    input_names, output_names = predictor.get_input_names(), predictor.get_output_names()
    if len(input_names) != 1 or len(output_names) != args.expected_outputs:
        raise ValueError("unexpected OCR input/output count")
    handle = predictor.get_input_handle(input_names[0])
    handle.reshape(list(x.shape))
    handle.copy_from_cpu(x)
    if args.device == "cuda":
        paddle.device.cuda.synchronize()
    times = []
    for i in range(-args.warmups, args.runs):
        begin = time.perf_counter()
        predictor.run()
        if args.device == "cuda":
            paddle.device.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        if i >= 0:
            times.append(elapsed)
    outputs = [predictor.get_output_handle(name).copy_to_cpu() for name in output_names]
    if any(output.dtype != np.float32 or not output.size or
           not np.isfinite(output).all() for output in outputs):
        raise ValueError("invalid source probability tensor")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.report.with_suffix(".npy" if len(outputs) == 1 else ".npz")
    if len(outputs) == 1:
        np.save(output_path, outputs[0])
    else:
        np.savez(output_path, **{f"head{i}": output for i, output in enumerate(outputs)})
    time.sleep(args.audit_hold_ms / 1000)
    # The caller requires a successful process exit before trusting outputs.
    del handle, predictor, config
    gc.collect()
    ordered = sorted(times)
    report = dict(program_sha256=args.program_sha256,
        parameters_sha256=args.parameters_sha256, input_sha256=args.input_sha256,
        input_shape=list(x.shape), output_shape=list(outputs[0].shape),
        output_names=output_names, output_shapes=[list(output.shape) for output in outputs],
        output_sha256=digest(output_path), device=args.device,
        paddle_version=importlib.metadata.version(distribution),
        **distribution_environment(),
        distribution_identity_sha256=identity_sha256,
        paddle_library_sha256=digest(Path(paddle.base.libpaddle.__file__)),
        logging_environment_keys=sorted(k for k in os.environ
                                        if k.startswith(("GLOG_", "MLOG_"))),
        threads=args.threads, warmups=args.warmups, runs=args.runs,
        forward_ms=times, p50_ms=ordered[len(ordered) // 2],
        p95_ms=ordered[int(len(ordered) * .95)], tf32=False,
        math_policy={"NVIDIA_TF32_OVERRIDE": os.environ["NVIDIA_TF32_OVERRIDE"]},
        ir_optim=False,
        audit_hold_ms=args.audit_hold_ms,
        timing_scope="synchronous predictor graph and dispatch; no input upload/output copy")
    if digest(identity_module) != identity_sha256:
        raise ValueError("package identity helper changed during execution")
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
