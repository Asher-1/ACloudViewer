"""Checkpoint bounds must reject drift hidden by an aggregate error score."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'relation_accuracy', Path(__file__).with_name('accuracy.py'))
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)


class AccuracyTests(unittest.TestCase):
    def output(self, logits, pairs=None, subjects=None, objects=None):
        count = len(logits)
        return dict(logits=np.asarray(logits, dtype=np.float32),
                    pairs=np.asarray(pairs if pairs is not None else [0]*count,
                                     dtype=np.float32),
                    sub=np.asarray(subjects if subjects is not None else [0]*count),
                    obj=np.asarray(objects if objects is not None else range(1, count+1)),
                    valid=np.ones(count, dtype=np.int32))

    def test_local_regression_is_visible_when_mean_error_improves(self):
        truth = self.output([[0, 0]])
        native = self.output([[1, 1]])
        candidate = self.output([[1.1, .1]])
        result = accuracy.checkpoint_metrics(truth, native, candidate)
        self.assertLess(np.abs(candidate['logits']).mean(),
                        np.abs(native['logits']).mean())
        self.assertAlmostEqual(result['official_output_error_excess'], .1,
                               places=6)

    def test_pair_score_regression_is_checked_independently(self):
        truth = self.output([[0]], pairs=[0])
        native = self.output([[0]], pairs=[.1])
        candidate = self.output([[0]], pairs=[.2])
        result = accuracy.checkpoint_metrics(truth, native, candidate)
        self.assertEqual(result['official_output_error_excess'], 0)
        self.assertAlmostEqual(result['official_pair_output_error_excess'], .1)

    def test_pooling_weights_logits_and_pairs_instead_of_image_means(self):
        first = self.output([[0, 0]])
        second = self.output([[0, 0], [0, 0], [0, 0]])
        actual = self.output([[2, 0], [2, 0], [2, 0]])
        result = accuracy.pooled_metrics([
            accuracy.checkpoint_metrics(first, first, first),
            accuracy.checkpoint_metrics(second, actual, actual)])
        self.assertAlmostEqual(result['official_logits_rmse'], np.sqrt(1.5))
        self.assertEqual(result['official_predicate_top1_agreement'], 1)
        self.assertEqual(result['official_logit_count'], 8)
        self.assertEqual(result['official_logit_squared_error_sum'], 12)
        self.assertEqual(result['official_matched_pair_count'], 4)

    def test_pair_order_does_not_change_output_alignment(self):
        truth = self.output([[1, 0], [0, 1]])
        actual = self.output([[0, 1], [1, 0]], objects=[2, 1])
        result = accuracy.checkpoint_metrics(truth, actual, actual)
        self.assertEqual(result['official_logit_squared_error_sum'], 0)
        self.assertEqual(result['official_predicate_matches'], 2)

    def test_unmatched_pairs_reduce_absolute_jaccard(self):
        truth = self.output([[1, 0], [0, 1]])
        actual = self.output([[1, 0], [0, 1]], objects=[1, 3])
        result = accuracy.pooled_metrics([
            accuracy.checkpoint_metrics(truth, actual, actual)])
        self.assertEqual(result['official_pairs_jaccard'], 1/3)

    def test_duplicate_native_pair_cannot_hide_drift(self):
        output = self.output([[0], [0]], objects=[1, 1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            accuracy.checkpoint_metrics(output, output, output)

    def test_nonfinite_truth_or_baseline_cannot_pass(self):
        valid = self.output([[0, 0]])
        invalid = self.output([[float('nan'), 0]])
        for outputs in ((invalid, valid, valid), (valid, invalid, valid),
                        (valid, valid, invalid)):
            with self.subTest(outputs=outputs):
                with self.assertRaisesRegex(ValueError, 'invalid'):
                    accuracy.checkpoint_metrics(*outputs)

    def test_absolute_quantization_bounds_and_added_error_are_enforced(self):
        metrics = dict(official_logits_rmse=.006,
                       official_predicate_top1_agreement=.993,
                       official_pairs_jaccard=1,
                       official_output_error_excess=0,
                       official_pair_output_error_excess=0)
        self.assertEqual(len(accuracy.absolute_failures(metrics,.005,.995,.995)),2)
        self.assertEqual(accuracy.absolute_failures(metrics,.010,.990,.990),[])
        metrics['official_output_error_excess']=.00002
        self.assertEqual(len(accuracy.absolute_failures(metrics,.05,.97,.98)),1)
        del metrics['official_pairs_jaccard']
        self.assertEqual(len(accuracy.absolute_failures(metrics,.05,.97,.98)),2)

    def test_every_manifest_model_owns_explicit_bounds_and_shared_truth_cache(self):
        import json
        manifest = json.loads((Path(__file__).resolve().parents[2] /
                               'scripts/validation_manifest.json').read_text())
        expected = {'f32': (.005, .995, .995),
                    'f16': (.010, .990, .990),
                    'q8_0': (.050, .970, .980)}
        for scenario in manifest['scenarios']:
            if scenario['task'] != 'relation':
                continue
            # Plugin-level rows (temporal tracker / upstream replay parity)
            # are model-free and do not run the checkpoint accuracy probe.
            if not any(str(a).endswith('upstream_probe.py')
                       for a in scenario.get('args', [])):
                continue
            with self.subTest(scenario=scenario['id']):
                args = scenario['args']
                self.assertEqual(args[args.index('--reference-cache')+1],
                                 '{build}/Testing/framework-references/relation')
                checks = {row['metric']: row for row in
                          scenario['accuracy']['checks']}
                self.assertEqual(checks['official_output_error_excess']['value'],
                                 1e-5)
                if '--text' in args:
                    continue
                quantization = scenario['model_glob'].rsplit('-', 1)[1][:-5]
                limits = expected[quantization]
                for metric, option, operator, limit in zip(
                        ('official_logits_rmse',
                         'official_predicate_top1_agreement',
                         'official_pairs_jaccard'),
                        ('--official-rmse-limit', '--official-top1-limit',
                         '--official-jaccard-limit'),
                        ('<=', '>=', '>='), limits):
                    self.assertEqual(float(args[args.index(option)+1]), limit)
                    self.assertEqual(checks[metric]['value'], limit)
                    self.assertEqual(checks[metric]['op'], operator)
                    self.assertIn(metric,
                                  scenario['accuracy']['require_metrics'])
                self.assertEqual(
                    checks['official_pair_output_error_excess']['value'],1e-5)


if __name__ == '__main__':
    unittest.main()
