#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--probe', type=Path, required=True)
p.add_argument('--modules', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
base = a.output.resolve()
base.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='run-', dir=base))
reports = subprocess.run(['nsys', 'stats', '--help-reports'], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
report = 'cuda_gpu_kern_sum' if 'cuda_gpu_kern_sum' in reports else 'gpukernsum'
failures = []
rows = []
for mode, expected in (('trace', True), ('fallback', False)):
    trace = out / mode
    assert not trace.with_suffix('.nsys-rep').exists(), 'Existing trace must be preserved; use a new output directory'
    command = ['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
               '--force-overwrite=false', '-o', str(trace), str(a.probe.resolve()),
               str(a.modules.resolve()), 'CUDA0', mode, '-']
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / (mode + '.log')).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    stats = subprocess.run(['nsys', 'stats', '--report=' + report, '--format=csv',
                            str(trace.with_suffix('.nsys-rep'))],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    (out / (mode + '-kernels.csv')).write_text(stats.stdout)
    if 'ERROR:' in stats.stdout:
        raise RuntimeError(stats.stdout)
    kernels = [line for line in stats.stdout.splitlines() if 'conv2d_dw_' in line]
    specialized = any('conv2d_dw_7x7_cwhn_kernel' in line for line in kernels)
    if not kernels or specialized != expected:
        failures.append(f'{mode}: expected DW7={expected}, actual DW7={specialized}, kernels={kernels}')
    rows.append(dict(mode=mode, specialized=specialized, expected=expected, kernels=kernels))
print(json.dumps(dict(rows=rows, failures=failures), indent=2))
(out / 'receipt.json').write_text(json.dumps(dict(rows=rows, failures=failures), indent=2) + '\n')
raise SystemExit(1 if failures else 0)
