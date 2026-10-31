#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run_ocr_paddle_layout_probe import accuracy_pass, canonical_heads, head_comparison, run
from run_ocr_paddle_probe import digest


class LayoutGraphTests(unittest.TestCase):
    def setUp(self):
        self.heads = [np.zeros((1, n, c), np.float32)
                      for n in (100, 25, 9, 4) for c in (10, 32)]

    def test_fetch_order_does_not_reorder_class_or_regression_channels(self):
        for i, head in enumerate(self.heads):
            head[:, :, i] = i + 1
        actual = canonical_heads(self.heads[::-1], self.heads)
        self.assertTrue(all(np.array_equal(a, e) for a, e in zip(actual, self.heads)))
        self.assertTrue(accuracy_pass(head_comparison(actual, self.heads), .001, .001))

    def test_missing_duplicate_and_wrong_channel_heads_fail(self):
        wrong = self.heads[:-1] + [np.zeros((1, 4, 31), np.float32)]
        for heads in (self.heads[:-1], self.heads[:-1] + self.heads[:1], wrong):
            with self.assertRaises(ValueError):
                canonical_heads(heads, self.heads)

    def test_nonfinite_head_cannot_hide_in_an_aggregate(self):
        for value in (np.nan, np.inf, -np.inf):
            actual = [h.copy() for h in self.heads]
            actual[7][0, -1, -1] = value
            with self.assertRaises(ValueError):
                head_comparison(actual, self.heads)

    def test_near_tie_argmax_change_is_diagnostic_when_probability_error_passes(self):
        expected = [h.copy() for h in self.heads]
        actual = [h.copy() for h in self.heads]
        expected[0][0, 0, :2] = [.50001, .49999]
        actual[0][0, 0, :2] = [.49999, .50001]
        metrics = head_comparison(actual, expected)
        self.assertLess(metrics['max_diff'], .001)
        self.assertTrue(accuracy_pass(metrics, .001, .001, quantized=True))
        self.assertFalse(accuracy_pass(metrics, .001, .001))

    def test_manifest_requires_each_pinned_head_and_backend_metrics(self):
        manifest = json.loads((Path(__file__).parent / 'validation_manifest.json').read_text())
        rows = [r for r in manifest['scenarios'] if r['id'].startswith('ocr-paddle-layout-')]
        self.assertEqual(len(rows), 3)
        assets = {a['id']: a for a in manifest['input_assets']}
        for row in rows:
            self.assertEqual(row['args'].count('--reference-head'), 8)
            self.assertIn('ocr-paddle-source-layout', row['input_ids'])
            for i in range(8):
                entry = assets[f'ocr-layout-head{i}']
                self.assertIn(entry['id'], row['input_ids'])
                self.assertIn(entry['sha256'], row['args'])
            if row['id'].endswith('-f32'):
                self.assertIn('json0/argmax_match',
                              [check['metric'] for check in row['accuracy']['checks']])
            else:
                self.assertNotIn('json0/argmax_match',
                                 [check['metric'] for check in row['accuracy']['checks']])
                self.assertIn('json0/argmax_match', row['accuracy']['require_metrics'])
            self.assertFalse(any('/cpu_comparison/' in metric
                                 for metric in row['accuracy']['require_metrics']))
            self.assertIn('{profile}', row['args'])
            self.assertIn('{reference:ocr_paddle_gpu}', row['args'])

    def test_dfl_compares_each_edge_even_when_global_argmax_is_unchanged(self):
        expected = [h.copy() for h in self.heads]
        actual = [h.copy() for h in self.heads]
        expected[1][0, 0, 0] = actual[1][0, 0, 0] = 10
        expected[1][0, 0, 8:10] = [.50001, .49999]
        actual[1][0, 0, 8:10] = [.49999, .50001]
        self.assertTrue(np.array_equal(np.argmax(actual[1], axis=-1),
                                       np.argmax(expected[1], axis=-1)))
        metrics = head_comparison(actual, expected)
        self.assertLess(metrics['max_diff'], .001)
        self.assertLess(metrics['heads'][1]['dfl_probability_max_diff'], .001)
        self.assertTrue(accuracy_pass(metrics, .001, .001, quantized=True))

    def test_gpu_run_uses_cuda_heads_without_cpu_argmax_gate(self):
        cpu = [head.copy() for head in self.heads]
        cpu[0][0, 0, :2] = [.50001, .49999]
        cuda = [head.copy() for head in cpu]
        cuda[0][0, 0, :2] = [.49999, .50001]
        for backend, matches_cuda in ((backend, match) for backend in
                ('cuda', 'vulkan') for match in (True, False)):
            with self.subTest(backend=backend, matches_cuda=matches_cuda), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                np.save(root/'input.npy', np.zeros((1, 3, 800, 608), np.float32))
                for name in ('model-f16.gguf', 'program', 'parameters', 'config', 'probe'):
                    (root/name).write_bytes(name.encode())
                (root/'pins.json').write_text('{}')
                heads = []
                for index, value in enumerate(cpu):
                    path = root/f'head{index}.npy'; np.save(path, value)
                    heads.append((str(path), digest(path)))
                args = SimpleNamespace(input=root/'input.npy',
                    input_sha256=digest(root/'input.npy'), reference_head=heads,
                    source_root=root, source_pins=root/'pins.json', model=root/'model-f16.gguf',
                    source_python=Path('cpu-source'), source_gpu_python=Path('cuda-source'),
                    probe=root/'probe', report=root/'report.json', backend=backend,
                    measurement_profile='release', max_diff=.001, mean_diff=.001,
                    source_max_diff=.001, warmups=2, runs=10, threads=4)
                devices = []

                def source(args, files, python, target, device):
                    devices.append(device)
                    self.assertEqual(str(python), 'cuda-source')
                    return cuda, dict(p50_ms=1, p95_ms=1), dict(passed=True,
                        archive_path='audit.json', sampled_process_peak_mib=20)

                def native(args, work, references, **kwargs):
                    self.assertFalse(kwargs.get('cpu', False))
                    self.assertTrue(all(np.array_equal(a, b)
                                        for a, b in zip(references, cuda)))
                    return (cuda if matches_cuda else cpu), dict(p50_ms=1, p95_ms=1), dict(passed=True,
                        archive_path='audit.json', sampled_process_peak_mib=20), 0

                with patch('run_ocr_paddle_layout_probe.verified_source',
                        return_value={key:root/key for key in ('program','parameters','config')}), \
                        patch('run_ocr_paddle_layout_probe.source_run', side_effect=source), \
                        patch('run_ocr_paddle_layout_probe.native_run', side_effect=native):
                    report = run(args)
                self.assertEqual(devices, ['cuda'])
                self.assertEqual(report['exit_code'], 0)
                self.assertAlmostEqual(report['argmax_match'], 1 if matches_cuda else .99)
                self.assertAlmostEqual(report['source_cuda_fixture_comparison']['argmax_match'], .99)


if __name__ == '__main__':
    unittest.main()
