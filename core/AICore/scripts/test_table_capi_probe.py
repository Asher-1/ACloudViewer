#!/usr/bin/env python3
"""Failure cases that must not produce a public table timing PASS."""

import copy
import json
import unittest

from run_table_capi_probe import native_rows, timing_percentiles


class PublicTableTimingTests(unittest.TestCase):
    def rows(self):
        tables = [dict(structure_tokens=['<tr>', '<td></td>', '</tr>'], rows=1)]
        return [dict(table_ms=10 + i, tables=copy.deepcopy(tables)) for i in range(10)] + [
            dict(summary=True, images=10, runs=10, warmups=2, threads=4,
                 failures=0, device='CUDA0')]

    def evaluate(self, rows):
        return native_rows('\n'.join(json.dumps(row) for row in rows),
            10, 2, 4, ['<tr>', '<td></td>', '</tr>'], 'cuda')

    def test_samples_exclude_warmup_and_match_harness_percentile(self):
        timing, predictions = self.evaluate(self.rows())
        self.assertEqual(timing, dict(p50_ms=14, p95_ms=18))
        self.assertEqual(len(predictions), 10)

    def test_source_and_native_use_identical_quantiles(self):
        samples = list(range(10, 20))
        self.assertEqual(timing_percentiles(samples[::-1], 10),
                         self.evaluate(self.rows())[0])
        for invalid in (samples[:-1], samples + [20], [0] * 10,
                        [float('nan')] * 10, [float('inf')] * 10):
            with self.subTest(samples=invalid), self.assertRaises(ValueError):
                timing_percentiles(invalid, 10)

    def test_missing_extra_and_unsummarized_outputs_fail(self):
        rows = self.rows()
        for invalid in (rows[1:], rows + [rows[-1]], rows[:-1], rows[::-1]):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                self.evaluate(invalid)

    def test_protocol_or_silent_cpu_fallback_fails(self):
        for name, value in (('images', 9), ('runs', 9), ('warmups', 1),
                            ('threads', 2), ('failures', 1), ('device', 'CPU')):
            rows = self.rows()
            rows[-1][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.evaluate(rows)

    def test_every_forward_must_retain_original_structure_tokens(self):
        for index in range(10):
            rows = self.rows()
            rows[index]['tables'][0]['structure_tokens'][1] = '<td colspan="2">'
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.evaluate(rows)

    def test_finite_positive_time_is_required(self):
        for invalid in (0, -1, float('nan'), float('inf')):
            rows = self.rows()
            rows[0]['table_ms'] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.evaluate(rows)

    def test_changed_typed_result_cannot_hide_behind_stable_tokens(self):
        rows = self.rows()
        rows[1]['tables'][0]['rows'] = 2
        with self.assertRaisesRegex(ValueError, 'repeated-forward'):
            self.evaluate(rows)


if __name__ == '__main__':
    unittest.main()
