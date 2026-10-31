#!/usr/bin/env python3
"""Real probe/Nsight: NVTX ranges cover measured forwards only, never lifecycle calls."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


def sha256_file(path):
    # Hash with a fixed 1 MiB buffer, never loading whole models/profile databases into memory.
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


class MeasuredRangesTest(unittest.TestCase):
    def test_only_measured_forwards_have_real_nvtx_ranges(self):
        model = Path(os.environ['AICORE_TEST_LLM_GGUF']).resolve(strict=True)
        library = Path(os.environ['AICORE_TEST_NVTX_LIBRARY']).resolve(strict=True)
        before = sha256_file(model)
        if args.artifact_dir:
            args.artifact_dir.resolve().mkdir(parents=True, exist_ok=False)
        artifacts = (nullcontext(str(args.artifact_dir.resolve())) if args.artifact_dir else
                     tempfile.TemporaryDirectory(dir=args.output_parent))
        with artifacts as folder:
            folder = Path(folder).resolve()
            trace = folder / 'trace'
            report = folder / 'probe.json'
            command = [str(args.nsys), 'profile', '--trace=nvtx', '--sample=none',
                '--cpuctxsw=none', '--wait=primary', '--stats=true', '--output=' + str(trace),
                str(args.probe.resolve()), '--model', str(model), '--device', 'cpu', '--threads', '4',
                '--prompt', 'What is the capital of France?', '--max-tokens', '4',
                '--warmups', '1', '--repeats', '2', '--profile-nvtx', str(library),
                '--json-out', str(report)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            result = json.loads(report.read_text()) if report.exists() else {}
            self.assertEqual(completed.returncode, 0,
                             json.dumps(result, ensure_ascii=False) + '\n' +
                             completed.stdout + completed.stderr)
            self.assertTrue(result['pass'])
            self.assertTrue(result['profiling_only'])
            self.assertEqual(result['measured_nvtx_ranges'], 2)
            self.assertEqual(len(result['e2e_samples_ms']), 2)
            self.assertEqual(result['nvtx_library_sha256'], sha256_file(library))
            with sqlite3.connect(trace.with_suffix('.sqlite').as_uri() + '?mode=ro', uri=True) as connection:
                connection.row_factory = sqlite3.Row
                events = [dict(row) for row in connection.execute('SELECT * FROM NVTX_EVENTS')]
                strings = dict(connection.execute('SELECT id, value FROM StringIds'))
                measured = [row for row in events if str(row.get('text') or strings.get(row.get('textId'), '')).startswith('aicore/llm/measured/')]
                self.assertEqual(len(measured), 2, measured)
                names = [row.get('text') or strings[row['textId']] for row in measured]
                self.assertEqual(sorted(names), ['aicore/llm/measured/0', 'aicore/llm/measured/1'])
                self.assertTrue(all(row['end'] > row['start'] for row in measured), measured)
                # A missing pop includes recreate/cancel in the range; counts alone cannot detect this contamination.
                by_name = dict(zip(names, measured))
                for index, elapsed_ms in enumerate(result['e2e_samples_ms']):
                    event = by_name[f'aicore/llm/measured/{index}']
                    duration_ms = (event['end'] - event['start']) / 1e6
                    self.assertGreaterEqual(duration_ms, elapsed_ms)
                    self.assertLessEqual(duration_ms, elapsed_ms + max(5, elapsed_ms * 0.05))
            self.assertEqual(sha256_file(model), before)


class MeasuredCudaSelectionTest(unittest.TestCase):
    def make_fixture(self, folder, missing_range=False):
        # Transient SQLite events: exclude simultaneous reference kernels and out-of-range loads.
        database = folder / 'events.sqlite'
        pid, child = 101 << 24, 202 << 24
        with sqlite3.connect(database) as db:
            db.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
                CREATE TABLE NVTX_EVENTS(globalTid INTEGER, start INTEGER, end INTEGER, text TEXT, textId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER, demangledName INTEGER,
                    gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                    start INTEGER, end INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER, nameId INTEGER, start INTEGER, end INTEGER);
            ''')
            db.execute('INSERT INTO StringIds VALUES(1, "gemm")')
            db.executemany('INSERT INTO PROCESSES VALUES(?,?,?)',
                           [(pid, 101, 'test_llm_probe'), (child, 202, 'llama-mtmd-cli')])
            for index in range(1 if missing_range else 2):
                start = (index * 20 + 10) * 1000000
                db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)',
                           (pid + 1, start, start + 10000000, f'aicore/llm/measured/{index}'))
            for owner, start, end in [(pid, 0, 1000000), (pid, 12000000, 15000000),
                                      (pid, 32000000, 36000000), (child, 12000000, 15000000)]:
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?)',
                           (owner, 1, 1, 1, 1, 32, 1, 1, start, end))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)',
                           (owner + 1, 1, start, start + 1))
        report = folder / 'probe.json'
        report.write_text(json.dumps({'profiling_only': True, 'repeats': 2,
            'measured_nvtx_ranges': 2, 'measured_nvtx_names': ['aicore/llm/measured/0', 'aicore/llm/measured/1'],
            'e2e_samples_ms': [10, 10]}))
        return database, report

    def analyze(self, database, report, output):
        if not args.analyzer:
            self.skipTest('No --analyzer supplied; measured CUDA selection was not qualified')
        return subprocess.run([sys.executable, str(args.analyzer), '--database', str(database),
            '--measured-probe', str(report), '--output', str(output)], capture_output=True, text=True)

    def test_only_main_consumer_measured_kernels_are_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            database, report = self.make_fixture(folder)
            output = folder / 'routes.json'
            completed = self.analyze(database, report, output)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result['profiling_only'])
            self.assertFalse(result['release_qualified'])
            self.assertEqual(len(result['processes']), 1)
            self.assertEqual(result['processes'][0]['pid'], 101)
            self.assertEqual(result['processes'][0]['kernel_count'], 2)
            self.assertEqual(result['processes'][0]['kernels'][0]['total_ms'], 7)
            self.assertEqual([row['kernel_count'] for row in result['measured_ranges']], [1, 1])

    def test_missing_measured_range_cannot_qualify_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            database, report = self.make_fixture(folder, missing_range=True)
            output = folder / 'routes.json'
            completed = self.analyze(database, report, output)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('measured', completed.stdout + completed.stderr)
            self.assertFalse(output.exists())


class RealCudaMeasuredRangesTest(unittest.TestCase):
    def test_each_real_vlm_range_has_main_consumer_kernel_and_api(self):
        if not args.cuda_profile_root:
            self.skipTest('No real CUDA profile supplied; graph-node capture was not qualified')
        self.assertIsNotNone(args.analyzer)
        with tempfile.TemporaryDirectory(dir=args.output_parent) as directory:
            for side in ('current', 'frozen'):
                for quant in ('q8', 'f16'):
                    with self.subTest(side=side, quant=quant):
                        root = args.cuda_profile_root / side
                        stem = quant + '-' + args.cuda_profile_suffix
                        probe = root / ('probe-' + stem + '.json')
                        database = root / ('trace-' + stem + '.sqlite')
                        output = Path(directory) / (side + '-' + quant + '.json')
                        completed = subprocess.run([sys.executable, str(args.analyzer),
                            '--database', str(database), '--measured-probe', str(probe),
                            '--output', str(output)], capture_output=True, text=True)
                        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                        result = json.loads(output.read_text())
                        measured_probe = json.loads(probe.read_text())
                        self.assertTrue(measured_probe['pass'])
                        self.assertEqual(measured_probe['upstream_text_match'], 1)
                        self.assertEqual(measured_probe['warmups'], 2)
                        self.assertEqual(measured_probe['repeats'], 10)
                        self.assertIn('cuda', measured_probe['device'].lower())
                        self.assertEqual(result['measured_probe_sha256'], sha256_file(probe))
                        self.assertEqual(result['database_sha256'], sha256_file(database))
                        self.assertEqual(len(result['measured_ranges']), 10)
                        self.assertEqual(len(result['processes']), 1)
                        self.assertEqual(result['processes'][0]['kind'], 'consumer')
                        self.assertTrue(all(row['kernel_count'] > 0 and row['api_count'] > 0
                                            for row in result['measured_ranges']))
                        self.assertTrue(result['profiling_only'])
                        self.assertFalse(result['release_qualified'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--nsys', type=Path, required=True)
    p.add_argument('--output-parent', type=Path, required=True)
    p.add_argument('--artifact-dir', type=Path,
                   help='Preserve real probe/trace receipts; the output directory must not exist, protecting old failures')
    p.add_argument('--analyzer', type=Path,
                   help='Explicit CUDA profile analyzer; also run transient-event selection regression')
    p.add_argument('--cuda-profile-root', type=Path, help='Current/frozen parent directory of four real measured CUDA traces')
    p.add_argument('--cuda-profile-suffix', choices=('pluginfix', 'nodefix', '500m-nodefix'),
                   default='pluginfix', help='Select 256M or 500M receipts explicitly; never mix model qualification scopes')
    args = p.parse_args()
    unittest.main(argv=['test_llm_probe_nvtx'])
