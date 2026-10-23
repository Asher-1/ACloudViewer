#!/usr/bin/env python3
"""Audited public table C ABI timing, using original CUDA source outputs.

The common runner owns model verification and process-level baseline repeats.
This consumer can run the same harness against an independently preserved old
AICore library, unlike a private probe that compiles its own runtime sources.
Full probability/location gates remain owned by run_table_paddle_probe.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from gen_ocr_fixtures import Gguf
from run_ocr_onnx_probe import source_performance_pass
from run_ocr_paddle_probe import digest, execute, source_environment
from run_table_paddle_probe import decoded_outputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from export_table_fixture import preprocess


def timing_percentiles(samples, count):
    if len(samples) != count or count < 1:
        raise ValueError('timing sample count differs from timed runs')
    values = [float(t) for t in samples]
    if any(not np.isfinite(t) or t <= 0 for t in values):
        raise ValueError('invalid public table timing')
    # Match the harness for both runtimes: sorting O(n log n) retains the
    # complete order needed by both quantiles; n is the small timed-run count.
    values.sort()
    return dict(p50_ms=values[int(.5 * (count - 1))],
                p95_ms=values[int(.95 * (count - 1))])


def native_rows(output, runs, warmups, threads, tokens, backend):
    documents = [json.loads(line) for line in output.splitlines()
                 if line.startswith('{')]
    if len(documents) != runs + 1 or not documents[-1].get('summary'):
        raise ValueError('public table output count differs from timed runs')
    summary = documents[-1]
    if (summary.get('images') != runs or summary.get('runs') != runs
            or summary.get('warmups') != warmups
            or summary.get('threads') != threads or summary.get('failures') != 0
            or backend not in summary.get('device', '').lower()):
        raise ValueError('public table measurement protocol/device differs')
    predictions, times = [], []
    for row in documents[:-1]:
        tables = row.get('tables', [])
        if len(tables) != 1 or tables[0].get('structure_tokens') != tokens:
            raise ValueError('public table structure tokens differ from original CUDA')
        predictions.append(tables)
        times.append(float(row['table_ms']))
    if any(row != predictions[0] for row in predictions[1:]):
        raise ValueError('public table repeated-forward output changed')
    return timing_percentiles(times, runs), predictions


def run(args):
    if (args.backend not in ('cuda', 'vulkan') or args.warmups < 2
            or args.runs < 10 or args.threads < 1):
        raise ValueError('controlled table timing requires GPU, 2 warmups and 10 forwards')
    if args.library.parent.resolve() != args.bench.parent.resolve():
        raise ValueError('library pin must belong to the harness source-owner directory')
    for name in ('program', 'parameters', 'index'):
        if digest(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise ValueError(name + ' SHA-256 mismatch')
    scripts = Path(__file__).resolve().parent
    benchmark = scripts.parent / 'tools/bench_paddle_ocr_graph.py'
    dependencies = [Path(__file__), benchmark, args.bench, args.library,
                    args.program, args.parameters, args.index,
                    scripts.parent / 'tools/export_table_fixture.py',
                    scripts / 'gen_ocr_fixtures.py',
                    scripts / 'run_ocr_onnx_probe.py',
                    scripts / 'run_ocr_paddle_probe.py',
                    scripts / 'run_table_paddle_probe.py',
                    scripts / 'gpu_audit.py', scripts / 'ocr_paddle_environment.py',
                    scripts / 'ocr_paddle_metadata.py', args.model]
    before = {str(p): digest(p) for p in dependencies}
    metadata = Gguf(str(args.model)).kv
    if (metadata.get('ocr.source.sha256') != args.parameters_sha256
            or metadata.get('ocr.source.name') != 'SLANet_plus_infer.tar'):
        raise ValueError('table source/model identity differs')
    vocabulary = metadata['ocr.table.tokens']
    if vocabulary[0] != 'sos' or vocabulary[-1] != 'eos':
        raise ValueError('table vocabulary SOS/EOS contract differs')
    index = json.loads(args.index.read_text())
    if not index or [x['index'] for x in index] != list(range(len(index))):
        raise ValueError('table timing inputs must be contiguous and nonempty')
    directory = Path(tempfile.mkdtemp(prefix=args.report.stem + '.',
                                      dir=args.report.parent))
    fixtures = []
    output_hash = hashlib.sha256()
    for entry in index:
        files = {}
        for name in ('image', 'input'):
            relative = Path(entry[name]['file'])
            if relative.name != str(relative):
                raise ValueError('timing input must be a single file name')
            path = args.index.parent / relative
            if digest(path) != entry[name]['sha256']:
                raise ValueError(name + ' input SHA-256 mismatch')
            before[str(path)] = digest(path)
            files[name] = path
        # The index ties a losslessly encoded image to the independent source
        # input. Recheck the actual preprocessing rather than trusting the index.
        x, _ = preprocess(files['image'], int(metadata['ocr.table.max_len']),
                          int(metadata['ocr.table.pad_size']))
        if not np.array_equal(x, np.load(files['input'])):
            raise ValueError('public image/source graph preprocessing differs')
        source_path = directory / f'source-{entry["index"]}.json'
        source_command = [str(args.source_gpu_python), str(benchmark),
            '--program', str(args.program), '--program-sha256', args.program_sha256,
            '--parameters', str(args.parameters), '--parameters-sha256', args.parameters_sha256,
            '--input', str(files['input']), '--input-sha256', digest(files['input']),
            '--device', 'cuda', '--expected-outputs', '2',
            '--warmups', str(args.warmups), '--runs', str(args.runs),
            '--threads', str(args.threads), '--audit-hold-ms', '1500',
            '--report', str(source_path)]
        _, source_audit = execute(source_command, source_path,
            environment=source_environment(args.source_gpu_python),
            gpu=True, strict=True, timeout=args.gpu_idle_timeout)
        source = json.loads(source_path.read_text())
        archive = source_path.with_suffix('.npz')
        if (source['device'] != 'cuda' or source['output_sha256'] != digest(archive)
                or source['program_sha256'] != args.program_sha256
                or source['parameters_sha256'] != args.parameters_sha256
                or source['input_sha256'] != digest(files['input'])
                or source['distribution_identity_sha256'] !=
                   before[str(scripts / 'ocr_paddle_metadata.py')]
                or any(source[name] != getattr(args, name)
                       for name in ('warmups', 'runs', 'threads'))):
            raise ValueError('original CUDA source report identity differs')
        with np.load(archive) as arrays:
            if set(arrays.files) != {'head0', 'head1'}:
                raise ValueError('original table graph must return two outputs')
            probabilities, _ = decoded_outputs([arrays[n] for n in arrays.files],
                                                len(vocabulary), len(vocabulary) - 1)
        tokens = [vocabulary[int(i)] for i in probabilities.argmax(-1)
                  if i not in (0, len(vocabulary) - 1)]
        target = directory / f'native-{entry["index"]}.json'
        command = [str(args.bench), '--table', str(args.model),
            '--device', args.backend, '--threads', str(args.threads),
            '--warmups', str(args.warmups), '--runs', str(args.runs),
            '--audit-hold-ms', '1500', str(files['image'])]
        completed, audit = execute(command, target, gpu=True, strict=True,
                                    timeout=args.gpu_idle_timeout)
        timing, predictions = native_rows(completed.stdout, args.runs,
            args.warmups, args.threads, tokens, args.backend)
        # The graph helper reports upper order statistics. Recompute from
        # its raw samples using the native harness's exact convention.
        source_timing = timing_percentiles(source['forward_ms'], args.runs)
        if not source_performance_pass(timing, source_timing):
            raise ValueError('public table latency regresses against original CUDA')
        output_hash.update(json.dumps(predictions, sort_keys=True).encode())
        fixtures.append(dict(index=entry['index'], **timing,
            source=source, source_timing=source_timing, source_report=str(source_path),
            source_report_sha256=digest(source_path), source_audit=source_audit,
            native_command=command, native_audit=audit))
    if any(digest(Path(path)) != sha for path, sha in before.items()):
        raise ValueError('table timing dependency changed during measurement')
    return dict(exit_code=0, backend=args.backend, dependencies=before,
        model_sha256=digest(args.model), library_sha256=digest(args.library),
        output_sha256=output_hash.hexdigest(), fixtures=fixtures,
        p50_ms=max(f['p50_ms'] for f in fixtures),
        p95_ms=max(f['p95_ms'] for f in fixtures), decoded_token_match=1,
        gpu_audit_pass=1, source_gpu_audit_pass=1, source_performance_pass=1,
        source_performance_qualified=1,
        gpu_sampled_process_peak_mib=max(f['native_audit']['sampled_process_peak_mib'] for f in fixtures),
        source_gpu_sampled_process_peak_mib=max(f['source_audit']['sampled_process_peak_mib'] for f in fixtures),
        sampling=dict(warmups=args.warmups, runs=args.runs, threads=args.threads),
        scope='Public table C ABI repeated timing and exact decoded source tokens; raw probability/location and labelled GT gates remain separate',
        timing_scope='Native includes preprocessing, transfer and postprocessing; original CUDA graph excludes them')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'bench', 'library', 'program', 'parameters', 'index',
                 'source-gpu-python', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('program', 'parameters', 'index'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--backend', choices=('cuda', 'vulkan'), required=True)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-idle-timeout', type=float, default=120)
    args = parser.parse_args()
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
