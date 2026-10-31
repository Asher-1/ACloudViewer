#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from run_table_paddle_probe import compare_outputs, decoded_outputs, digest, run


class TableOriginalGraphTests(unittest.TestCase):
    def test_common_manifest_owns_original_source_and_all_quantizations(self):
        manifest = json.loads(Path(__file__).with_name('validation_manifest.json').read_text())
        source = next(asset for asset in manifest['input_assets']
                      if asset['id'] == 'table-paddle-original')
        self.assertTrue(source['archive'])
        self.assertEqual(source['tasks'], ['table'])
        self.assertEqual(source['sha256'],
                         '68526c5c6573a8a17cdc14805e026f11c2545e42d89a1317bb58d3e1e4b58318')
        self.assertEqual({Path(member['relative_path']).name
                          for member in source['extracted_files']},
                         {'inference.json', 'inference.pdiparams', 'inference.yml'})
        for quant in ('f32', 'f16', 'q8_0'):
            scenario = next(row for row in manifest['scenarios']
                            if row['id'] == 'table-structure-' + quant)
            self.assertEqual(scenario['executable'], 'python')
            self.assertEqual(scenario['accuracy']['reference'], 'upstream')
            self.assertIn(source['id'], scenario['input_ids'])
            self.assertIn('{repo}/core/AICore/scripts/run_table_paddle_probe.py', scenario['args'])
            self.assertIn('{reference:ocr_paddle_gpu}', scenario['args'])
            self.assertIn('{profile}', scenario['args'])
            self.assertIn('{warmup_runs}', scenario['args'])
            self.assertIn('{inference_runs}', scenario['args'])
            for metric in ('max_diff', 'argmax_match', 'loc_max_diff'):
                self.assertTrue(any(check['metric'] == 'json0/' + metric
                                    for check in scenario['accuracy']['checks']))

    def outputs(self):
        # Two actual decoder steps, EOS, then the exported program's extra
        # uniform-softmax / zero-location row. Fetch order is unspecified.
        probabilities = np.full((1, 4, 50), 1 / 50, np.float32)
        for i, token in enumerate((5, 9, 49)):
            probabilities[0, i] = 0
            probabilities[0, i, token] = 1
        locations = np.full((1, 4, 8), .5, np.float32)
        locations[0, -1] = 0
        return [locations, probabilities]

    def test_fetch_order_and_padding_keep_eos(self):
        first = decoded_outputs(self.outputs(), 50, 49)
        second = decoded_outputs(self.outputs()[::-1], 50, 49)
        self.assertEqual(first[0].shape, (3, 50))
        self.assertEqual(first[1].shape, (3, 8))
        np.testing.assert_array_equal(first[0].argmax(-1), [5, 9, 49])
        self.assertEqual(compare_outputs(first, second)['argmax_match'], 1)

    def test_missing_eos_or_missing_padding_is_rejected(self):
        missing = self.outputs()
        missing[1][0, 2, 49] = 0
        missing[1][0, 2, 3] = 1
        with self.assertRaisesRegex(ValueError, 'terminate'):
            decoded_outputs(missing, 50, 49)
        with self.assertRaisesRegex(ValueError, 'terminate'):
            decoded_outputs([value[:, :3] for value in self.outputs()], 50, 49)

    def test_nonfinite_in_padding_cannot_hide_behind_eos(self):
        for bad in (np.nan, np.inf, -np.inf):
            outputs = self.outputs()
            outputs[0][0, -1, 0] = bad
            with self.assertRaises(ValueError):
                decoded_outputs(outputs, 50, 49)

    def test_duplicate_wrong_steps_and_transposed_fetches_are_rejected(self):
        outputs = self.outputs()
        for invalid in ([outputs[0], outputs[0]],
                        [outputs[0][:, :3], outputs[1]],
                        [outputs[0], outputs[1].transpose(0, 2, 1)]):
            with self.assertRaises(ValueError):
                decoded_outputs(invalid, 50, 49)

    def test_probability_normalization_and_sigmoid_domain_are_required(self):
        for head, channel, value in ((1, 0, -.1), (1, 0, .2), (0, 0, 1.1)):
            outputs = self.outputs()
            outputs[head][0, 0, channel] = value
            with self.assertRaises(ValueError):
                decoded_outputs(outputs, 50, 49)

    def test_small_error_does_not_hide_changed_structure_token(self):
        expected = decoded_outputs(self.outputs(), 50, 49)
        actual = [value.copy() for value in expected]
        expected[0][0, :2] = [.50001, .49999]
        expected[0][0, 5] = 0
        actual[0][0, :2] = [.49999, .50001]
        actual[0][0, 5] = 0
        metrics = compare_outputs(actual, expected)
        self.assertLess(metrics['max_diff'], .001)
        self.assertLess(metrics['argmax_match'], 1)

    def test_location_shape_drift_fails_before_broadcast(self):
        expected = decoded_outputs(self.outputs(), 50, 49)
        with self.assertRaises(ValueError):
            compare_outputs([expected[0], expected[1][:1]], expected)

    def test_gpu_truth_and_release_timing_do_not_execute_native_cpu(self):
        for profile, mismatch, slow in (('developer', False, True),
                                        ('release', False, False),
                                        ('release', True, False),
                                        ('release', False, True)):
            with self.subTest(profile=profile, mismatch=mismatch, slow=slow), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ('model', 'program', 'parameters', 'config', 'probe'):
                    (root / name).write_bytes(name.encode())
                np.save(root / 'input.npy', np.zeros((1, 3, 2, 2), np.float32))
                index = [dict(index=0, input=dict(file='input.npy',
                                                sha256=digest(root / 'input.npy')))]
                (root / 'index.json').write_text(json.dumps(index))
                args = SimpleNamespace(backend='vulkan', measurement_profile=profile,
                    warmups=2, runs=10, threads=4, gpu_idle_timeout=10,
                    max_diff=.01, loc_max_diff=.01, model=root / 'model',
                    probe=root / 'probe', input_index=root / 'index.json',
                    input_index_sha256=digest(root / 'index.json'),
                    source_python=Path('cpu-source'), source_gpu_python=Path('cuda-source'),
                    report=root / 'report.json')
                for name in ('program', 'parameters', 'config'):
                    setattr(args, name, root / name)
                    setattr(args, name + '_sha256', digest(root / name))
                metadata = {'ocr.source.name': 'SLANet_plus_infer.tar',
                    'ocr.source.sha256': args.parameters_sha256,
                    'ocr.table.tokens': ['sos'] + [str(i) for i in range(1, 49)] + ['eos'],
                    'ocr.table.pad_size': 2}
                source = self.outputs()
                source[1][0, 0] = 0
                source[1][0, 0, :2] = [.50001, .49999]
                launched = []

                def execute_fake(command, target, **kwargs):
                    launched.append(command)
                    audit = dict(passed=True, archive_path='audit.json',
                                 sampled_process_peak_mib=20)
                    if command[0] == 'cuda-source':
                        output = target.with_suffix('.npz')
                        np.savez(output, head0=source[0], head1=source[1])
                        target.write_text(json.dumps(dict(device='cuda',
                            output_sha256=digest(output), program_sha256=args.program_sha256,
                            parameters_sha256=args.parameters_sha256,
                            input_sha256=index[0]['input']['sha256'],
                            warmups=2, runs=10, threads=4,
                            distribution_identity_sha256=digest(
                                Path(__file__).with_name('ocr_paddle_metadata.py')),
                            p50_ms=1, p95_ms=1)))
                        text = ''
                    else:
                        self.assertEqual(command[0], str(args.probe))
                        self.assertEqual(command[3], 'vulkan')
                        values = decoded_outputs(source, 50, 49)
                        if mismatch:
                            values[0][0, :2] = [.49999, .50001]
                        for name, value in zip(('probs', 'loc'), values):
                            value.tofile(command[-1] + '.fixture0.' + name + '.bin')
                        text = json.dumps(dict(warmups=2, runs=10, threads=4,
                                               p50_ms=100 if slow else 1,
                                               p95_ms=100 if slow else 1))
                    return subprocess.CompletedProcess(command, 0, text, ''), \
                        audit if kwargs.get('gpu') else None

                with patch('run_table_paddle_probe.Gguf', return_value=SimpleNamespace(kv=metadata)), \
                        patch('run_table_paddle_probe.execute', side_effect=execute_fake):
                    report = run(args)
                    repeated = run(args)
                self.assertNotEqual(report['fixture_directory'],
                                    repeated['fixture_directory'])
                self.assertTrue(Path(report['fixture_directory']).is_dir())
                self.assertTrue(Path(repeated['fixture_directory']).is_dir())
                self.assertEqual(len(launched), 4)
                self.assertNotIn('cpu-source', [command[0] for command in launched])
                self.assertEqual(report['exit_code'],
                                 int(mismatch or profile == 'release' and slow))
                self.assertEqual(report['source_performance_qualified'], int(profile == 'release'))


if __name__ == '__main__':
    unittest.main()
