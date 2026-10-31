#!/usr/bin/env python3
"""SLANet graph accuracy and audited timing against its original Paddle graph.

The common validation runner owns downloads, process repeats and same-host
baseline comparison. This consumer verifies source/model/input identities and
compares all autoregressive probabilities and locations, including token EOS.
GPU rows compare directly with official CUDA. Native CPU is never a GPU gate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from gen_ocr_fixtures import Gguf
from run_ocr_onnx_probe import source_performance_pass
from run_ocr_paddle_probe import comparison, digest, execute, source_environment


def decoded_outputs(outputs, vocab, eos):
    """Identify the two original fetches and retain the first EOS step."""
    if len(outputs) != 2 or vocab <= 8 or eos != vocab - 1:
        raise ValueError("invalid SLANet vocabulary or output count")
    by_channels = {}
    for value in outputs:
        if (value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 1
                or value.shape[1] < 2 or value.shape[2] not in (8, vocab)
                or value.shape[2] in by_channels or not np.isfinite(value).all()):
            raise ValueError("invalid SLANet output shape or values")
        by_channels[value.shape[2]] = value[0]
    if set(by_channels) != {8, vocab}:
        raise ValueError("missing SLANet probability or location output")
    probabilities, locations = by_channels[vocab], by_channels[8]
    if probabilities.shape[0] != locations.shape[0]:
        raise ValueError("SLANet output step counts differ")
    terminal = np.flatnonzero(np.argmax(probabilities, axis=-1) == eos)
    if not terminal.size or terminal[0] + 1 >= len(probabilities):
        raise ValueError("SLANet did not terminate before the padding row")
    steps = int(terminal[0]) + 1
    probabilities, locations = probabilities[:steps], locations[:steps]
    if (np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-5, rtol=0)
            or np.any(locations < 0) or np.any(locations > 1)):
        raise ValueError("invalid SLANet probability or sigmoid domain")
    return probabilities, locations


def compare_outputs(actual, expected):
    probabilities, locations = actual
    ref_probabilities, ref_locations = expected
    return dict(**comparison(probabilities, ref_probabilities, argmax=True),
                loc_max_diff=comparison(locations, ref_locations)['max_diff'],
                loc_mean_diff=comparison(locations, ref_locations)['mean_diff'])


def run(args):
    if not all(np.isfinite(value) and value > 0
               for value in (args.max_diff, args.loc_max_diff)):
        raise ValueError("accuracy tolerances must be finite and positive")
    if args.measurement_profile == 'release' and (
            args.backend == 'cpu' or args.warmups < 2 or args.runs < 10):
        raise ValueError("release requires GPU, two warmups and ten forwards")
    scripts = Path(__file__).resolve().parent
    benchmark = scripts.parent / 'tools/bench_paddle_ocr_graph.py'
    dependencies = [Path(__file__), benchmark, args.probe,
                    scripts / 'run_ocr_paddle_probe.py',
                    scripts / 'run_ocr_onnx_probe.py', scripts / 'gpu_audit.py',
                    scripts / 'ocr_paddle_environment.py',
                    scripts / 'ocr_paddle_metadata.py', args.input_index,
                    args.program, args.parameters, args.config, args.model]
    before = {str(path): digest(path) for path in dependencies}
    for key in ('program', 'parameters', 'config'):
        if digest(getattr(args, key)) != getattr(args, key + '_sha256'):
            raise ValueError(f"original source {key} SHA-256 mismatch")
    model = Gguf(str(args.model)).kv
    if (model.get('ocr.source.name') != 'SLANet_plus_infer.tar'
            or model.get('ocr.source.sha256') != args.parameters_sha256):
        raise ValueError("SLANet GGUF does not contain the original source weights")
    tokens = model['ocr.table.tokens']
    if not tokens or tokens[0] != 'sos' or tokens[-1] != 'eos':
        raise ValueError("SLANet vocabulary has no SOS/EOS contract")
    if digest(args.input_index) != args.input_index_sha256:
        raise ValueError("fixture index SHA-256 mismatch")
    index = json.loads(args.input_index.read_text())
    if not index or [entry['index'] for entry in index] != list(range(len(index))):
        raise ValueError("fixture indices must be contiguous and nonempty")
    # The common runner reuses the report path for repeated processes. Retain
    # each attempt's source tensors and dumps under its own receipt directory.
    fixture_dir = Path(tempfile.mkdtemp(prefix=args.report.stem + '.fixtures-',
                                      dir=args.report.parent))
    strict = args.measurement_profile == 'release'
    gpu = args.backend != 'cpu'
    device = 'cuda' if gpu else 'cpu'
    python = args.source_gpu_python if gpu else args.source_python
    if python is None:
        raise ValueError("GPU comparison requires the official CUDA Python environment")
    sources, references = [], []
    for entry in index:
        relative = Path(entry['input']['file'])
        if relative.name != str(relative):
            raise ValueError("fixture input must be a single file name")
        input_path = args.input_index.parent / relative
        if digest(input_path) != entry['input']['sha256']:
            raise ValueError("fixture input SHA-256 mismatch")
        before[str(input_path)] = entry['input']['sha256']
        value = np.load(input_path)
        pad = int(model['ocr.table.pad_size'])
        if value.dtype != np.float32 or value.shape != (1, 3, pad, pad):
            raise ValueError("SLANet input is not the declared padded NCHW tensor")
        target = fixture_dir / f'source-{entry["index"]}.json'
        command = [str(python), str(benchmark),
            '--program', str(args.program), '--program-sha256', args.program_sha256,
            '--parameters', str(args.parameters), '--parameters-sha256', args.parameters_sha256,
            '--input', str(input_path), '--input-sha256', entry['input']['sha256'],
            '--expected-outputs', '2', '--device', device,
            '--warmups', str(args.warmups), '--runs', str(args.runs),
            '--threads', str(args.threads), '--audit-hold-ms', '1500' if strict else '0',
            '--report', str(target)]
        _, audit = execute(command, target, environment=source_environment(python),
            gpu=strict, strict=strict, timeout=args.gpu_idle_timeout)
        source = json.loads(target.read_text())
        archive = target.with_suffix('.npz')
        if (source['device'] != device or source['output_sha256'] != digest(archive)
                or source['program_sha256'] != args.program_sha256
                or source['parameters_sha256'] != args.parameters_sha256
                or source['input_sha256'] != entry['input']['sha256']
                or any(source[name] != getattr(args, name)
                       for name in ('warmups', 'runs', 'threads'))
                or source['distribution_identity_sha256'] !=
                   before[str(scripts / 'ocr_paddle_metadata.py')]):
            raise ValueError("independent source report identity mismatch")
        with np.load(archive) as arrays:
            if set(arrays.files) != {'head0', 'head1'}:
                raise ValueError("independent source must export exactly two heads")
            expected = decoded_outputs([arrays[name] for name in arrays.files],
                                       len(tokens), len(tokens) - 1)
        stem = fixture_dir / f'table_{entry["index"]}'
        np.save(str(stem) + '_input.npy', value)
        np.save(str(stem) + '_probs.npy', expected[0])
        np.save(str(stem) + '_loc.npy', expected[1])
        references.append(expected)
        sources.append(dict(report=str(target), report_sha256=digest(target),
                            command=command, metadata=source, audit=audit))
    prefix = fixture_dir / 'native'
    command = [str(args.probe), str(fixture_dir), str(args.model), args.backend,
        str(args.max_diff), str(args.loc_max_diff), str(args.warmups), str(args.runs),
        str(args.threads), '1500' if strict else '0', str(prefix)]
    completed, audit = execute(command, fixture_dir / 'native.json',
        gpu=strict, strict=strict, timeout=args.gpu_idle_timeout, require_success=False)
    native_metrics = {}
    for line in completed.stdout.splitlines():
        if line.startswith('{'):
            native_metrics.update(json.loads(line))
    for name in ('warmups', 'runs', 'threads'):
        if native_metrics.get(name) != getattr(args, name):
            raise ValueError("native probe measurement protocol differs")
    rows, output_digest = [], hashlib.sha256()
    for i, expected in enumerate(references):
        values = []
        for name, channels in (('probs', len(tokens)), ('loc', 8)):
            path = Path(str(prefix) + f'.fixture{i}.{name}.bin')
            payload = path.read_bytes()
            output_digest.update(payload)
            values.append(np.frombuffer(payload, np.float32).reshape(-1, channels))
        rows.append(compare_outputs(values, expected))
    metrics = dict(max_diff=max(row['max_diff'] for row in rows),
        mean_diff=max(row['mean_diff'] for row in rows),
        argmax_match=min(row['argmax_match'] for row in rows),
        loc_max_diff=max(row['loc_max_diff'] for row in rows))
    passed = (completed.returncode == 0 and metrics['argmax_match'] == 1
              and metrics['max_diff'] < args.max_diff
              and metrics['loc_max_diff'] < args.loc_max_diff)
    performance = source_performance_pass(native_metrics, sources[-1]['metadata'])
    report = dict(**metrics, exit_code=int(not passed or strict and not performance),
        backend=args.backend, model_sha256=digest(args.model), dependencies=before,
        p50_ms=native_metrics['p50_ms'], p95_ms=native_metrics['p95_ms'],
        sampling=dict(warmups=args.warmups, runs=args.runs, threads=args.threads),
        input_index_sha256=args.input_index_sha256, output_sha256=output_digest.hexdigest(),
        source_files={key: getattr(args, key + '_sha256')
                      for key in ('program', 'parameters', 'config')},
        fixture_directory=str(fixture_dir), fixtures=rows, sources=sources,
        native_command=command, native=native_metrics,
        source_performance_pass=int(performance), source_performance_qualified=int(strict),
        native_cpu_comparison='not executed for GPU rows',
        timing_scope='last identical padded input; native includes input/output transfer; source excludes it',
        scope='graph probabilities and EOS argmax; page GT and repeated-process baseline are separate gates')
    if strict:
        report.update(source_gpu_audit_pass=int(all(source['audit']['passed'] for source in sources)),
            source_gpu_sampled_process_peak_mib=max(
                source['audit']['sampled_process_peak_mib'] for source in sources),
            source_p50_ms=sources[-1]['metadata']['p50_ms'],
            source_p95_ms=sources[-1]['metadata']['p95_ms'])
    if audit:
        report.update(gpu_audit_pass=int(audit['passed']),
            gpu_sampled_process_peak_mib=audit['sampled_process_peak_mib'],
            gpu_audit_archive=audit['archive_path'])
    if any(digest(Path(path)) != sha for path, sha in before.items()):
        raise ValueError("probe dependency changed during execution")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('program', 'parameters', 'config', 'input-index', 'model',
                 'source-python', 'probe', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('program', 'parameters', 'config', 'input-index'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'vulkan'), required=True)
    parser.add_argument('--source-gpu-python', type=Path)
    parser.add_argument('--measurement-profile', choices=('developer', 'release'), default='developer')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-idle-timeout', type=float, default=60)
    parser.add_argument('--max-diff', type=float, required=True)
    parser.add_argument('--loc-max-diff', type=float, required=True)
    args = parser.parse_args()
    if args.threads == 0:
        args.threads = 4
    if args.warmups < 0 or args.runs < 1 or args.threads < 1:
        parser.error('invalid measurement counts')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as error:
        report = dict(exit_code=1, error=str(error), backend=args.backend)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
