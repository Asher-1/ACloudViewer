#!/usr/bin/env python3
"""LLM nodes match pristine bitwise; non-LLM nodes retain the frozen baseline bitwise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

LLAMA_ROWS = {"gemm-vision-anonymous", "gemm-projector-input", "gemm-text",
              "swiglu-text", "fused-f16-text", "fused-q8-text"}
OTHER_ROWS = {"gemm-task-default", "gemm-sam3d", "swiglu-task-default",
              "fused-f16-task", "fused-q8-task"}


def collect(probe, directory, fresh=False):
    directory = Path(directory).resolve()
    env = os.environ.copy()
    variable = "PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env[variable] = str(directory) + os.pathsep + env.get(variable, "")
    command = [str(probe), str(directory)]
    if fresh:
        command.append('--fresh-backend')
    result = subprocess.run(command, env=env,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"probe exit {result.returncode}: {result.stderr[-2000:]}")
    rows = {}
    for line in result.stdout.splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if row['id'] in rows:
            raise RuntimeError('Duplicate result row: ' + row['id'])
        rows[row['id']] = row['bits']
    if rows.keys() != LLAMA_ROWS | OTHER_ROWS:
        raise RuntimeError('Missing or unknown result rows: ' + str(rows.keys()))
    if any(not bits for bits in rows.values()):
        raise RuntimeError('Results must not be empty')
    modules = {}
    for path in directory.glob('*ggml*'):
        if path.is_file() and not path.name.endswith(('.a', '.lib')):
            modules[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, {'directory': str(directory), 'modules_sha256': modules}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        candidate, candidate_meta = collect(args.probe.resolve(), args.candidate)
        # Isolate reference rows; the candidate switches identical shapes on one backend,
        # also verifying that CUDA graph caches do not reuse another task's named arithmetic route.
        upstream, upstream_meta = collect(args.probe.resolve(), args.upstream, fresh=True)
        baseline, baseline_meta = collect(args.probe.resolve(), args.baseline, fresh=True)
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    checks = []
    for name in sorted(candidate):
        reference = upstream[name] if name in LLAMA_ROWS else baseline[name]
        same = candidate[name] == reference
        differences = sum(a != b for a, b in zip(candidate[name], reference))
        checks.append({'id': name, 'reference': 'pristine' if name in LLAMA_ROWS else 'baseline',
                       'bitwise_equal': same, 'different_elements': differences,
                       'candidate_elements': len(candidate[name]), 'reference_elements': len(reference)})
        print(f"{name}: {'PASS' if same else 'FAIL'} ({differences} changed elements)")
    report = {'candidate': candidate_meta, 'upstream': upstream_meta,
              'baseline': baseline_meta, 'checks': checks,
              'verdict': 'PASS' if all(row['bitwise_equal'] for row in checks) else 'FAIL'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
