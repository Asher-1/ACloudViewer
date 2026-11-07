#!/usr/bin/env python3
"""真实 CUDA 路由测试；缺失融合或错误融合回退域都会失败，不是性能验收。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    modules = args.modules.resolve(strict=True)
    protected = {str(path): sha(path) for path in
                 (probe, modules / "libggml-cuda.so")}
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    help_result = subprocess.run(["nsys", "stats", "--help-reports"],
                                 text=True, capture_output=True)
    reports = help_result.stdout + help_result.stderr
    # 本机旧 nsys 打印有效 help 时退出 1；仍拒绝真正错误或缺失报告。
    if help_result.returncode not in (0, 1) or "ERROR:" in reports:
        raise RuntimeError(f"nsys report discovery failed: {reports}")
    report = next((name for name in ("cuda_gpu_kern_sum", "gpukernsum")
                   if name in reports), None)
    if report is None:
        raise RuntimeError("nsys has no supported CUDA kernel report")
    rows, failures = [], []
    # 期望独立列明：只有完整的显式 F32 行 bias 域允许融合。
    for mode, expected in (("plain", False), ("hint", True),
                           ("observable", False), ("users", False),
                           ("strided", False), ("equal-bias", False),
                           ("nonadjacent", False)):
        trace = out / mode
        command = ["nsys", "profile", "--trace=cuda", "--sample=none",
                   "--cpuctxsw=none", "--force-overwrite=false", "-o", str(trace),
                   str(probe), str(modules), "CUDA0", mode]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (out / (mode + ".log")).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{mode}: probe/trace failed, rc={result.returncode}\n{result.stdout}")
        payloads = [json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith("{") and '"mode"' in line]
        if len(payloads) != 1 or payloads[0].get("mode") != mode or \
                payloads[0].get("backend") != "CUDA0" or payloads[0].get("pass") is not True:
            raise RuntimeError(f"{mode}: missing real CUDA probe verdict")
        stats = subprocess.run(["nsys", "stats", "--report=" + report,
                                "--format=csv", str(trace.with_suffix(".nsys-rep"))],
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=True)
        (out / (mode + "-kernels.csv")).write_text(stats.stdout)
        if "ERROR:" in stats.stdout:
            raise RuntimeError(stats.stdout)
        # 只统计 CSV 中实际执行次数为正的 kernel，nsys 说明文字不构成 GPU 执行证据。
        lines = stats.stdout.splitlines()
        header = next((i for i, line in enumerate(lines)
                       if "Name" in next(csv.reader([line]), []) and
                       "Instances" in next(csv.reader([line]), [])), None)
        if header is None:
            raise RuntimeError(f"{mode}: no CUDA kernel CSV table")
        kernels = [row["Name"] for row in csv.DictReader(lines[header:])
                   if row.get("Name") and int(row["Instances"]) > 0]
        fused = any("add_gelu_erf_bias_kernel" in name for name in kernels)
        if not kernels or fused != expected:
            failures.append(f"{mode}: expected fused={expected}, actual={fused}, kernels={kernels}")
        rows.append(dict(mode=mode, expected_fused=expected, actual_fused=fused,
                         kernels=kernels, command=command))
    unchanged = all(sha(Path(path)) == digest for path, digest in protected.items())
    if not unchanged:
        failures.append("Probe/backend identity changed during execution")
    receipt = dict(rows=rows, failures=failures, inputs_sha256=protected,
                   identities_unchanged=unchanged, release_qualified=False)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
