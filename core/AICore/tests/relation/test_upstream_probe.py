#!/usr/bin/env python3
"""Reject semantic edge drift and mismatched upstream build ownership."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    'relation_upstream_probe', Path(__file__).with_name('upstream_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class UpstreamProbeTests(unittest.TestCase):
    def test_cpu_and_text_framework_truth_never_query_or_initialize_cuda(self):
        torch=mock.Mock()
        torch.cuda.is_available.side_effect=AssertionError('CPU truth queried CUDA')
        self.assertEqual(probe.framework_device('cpu',False,torch),'cpu')
        self.assertEqual(probe.framework_device('cuda',True,torch),'cpu')
        torch.cuda.is_available.assert_not_called()

    def test_gpu_framework_truth_uses_available_device_and_falls_back_to_cpu(self):
        torch=mock.Mock()
        torch.cuda.is_available.return_value=True
        self.assertEqual(probe.framework_device('vulkan',False,torch),'cuda')
        torch.cuda.is_available.return_value=False
        self.assertEqual(probe.framework_device('cuda',False,torch),'cpu')

    def test_framework_reference_finishes_in_spawned_process_before_return(self):
        context=mock.Mock();receiver=mock.Mock();sender=mock.Mock();child=mock.Mock()
        context.Pipe.return_value=(receiver,sender)
        context.Process.return_value=child
        receiver.recv.return_value=(True,{'values':'independent truth'})
        child.exitcode=0;child.is_alive.return_value=False
        with mock.patch.object(probe.multiprocessing,'get_context',return_value=context) as create:
            self.assertEqual(probe.isolated_official_reference('upstream',text=True),
                             {'values':'independent truth'})
        create.assert_called_once_with('spawn')
        child.start.assert_called_once();child.join.assert_called_once()
        sender.close.assert_called_once();receiver.close.assert_called_once()

    def test_framework_reference_failure_is_propagated_and_child_is_reaped(self):
        context=mock.Mock();receiver=mock.Mock();sender=mock.Mock();child=mock.Mock()
        context.Pipe.return_value=(receiver,sender)
        context.Process.return_value=child
        receiver.recv.return_value=(False,'checkpoint unavailable')
        child.exitcode=0;child.is_alive.return_value=False
        with mock.patch.object(probe.multiprocessing,'get_context',return_value=context):
            with self.assertRaisesRegex(RuntimeError,'checkpoint unavailable'):
                probe.isolated_official_reference('upstream')
        child.join.assert_called_once();receiver.close.assert_called_once()

    def test_timed_run_reuses_shared_gpu_audit_without_allowing_other_pids(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            def audited(command,log,evaluation,allowed,env):
                self.assertEqual(allowed,set())
                self.assertEqual(env,{'TASK_RUNTIME':'configured'})
                log.write_text('latency_ms=10\niteration_ms=9,11\n'
                               '\n--- stderr ---\n\n--- gpu timing audit ---\n{}')
            module=mock.Mock();module.execute_audited.side_effect=audited
            result=probe.run(['native'],root,'upstream-before',
                             {'TASK_RUNTIME':'configured'},(module,object()))
            self.assertEqual(result['iteration_ms'],[9,11])
            module.execute_audited.assert_called_once()

    def test_iteration_samples_preserve_order_and_fingerprint_precision(self):
        values = probe.metrics('fingerprint=12345678901234567890\n'
                               'forward_iteration_ms=12,10.5,1e1\n')
        self.assertEqual(values['fingerprint'], 12345678901234567890)
        self.assertEqual(values['forward_iteration_ms'], [12, 10.5, 10])

    def test_timing_percentiles_use_matching_before_and_after_samples(self):
        result = probe.timing_distribution(
            {'forward_iteration_ms': [12, 10, 11]},
            {'iteration_ms': [8, 9, 10]},
            {'iteration_ms': [10, 11, 12]}, 3)
        self.assertEqual(result['candidate_p50_ms'], 11)
        self.assertEqual(result['upstream_p50_ms'], 10)
        self.assertEqual(result['candidate_p95_ms'], 12)
        self.assertEqual(result['upstream_p95_ms'], 11)
        with self.assertRaisesRegex(ValueError, 'timed iteration'):
            probe.timing_distribution(
                {'forward_iteration_ms': [12, 10]},
                {'iteration_ms': [8, 9, 10]},
                {'iteration_ms': [10, 11, 12]}, 3)

    def test_failure_identifies_sample_metric_and_unchanged_limit(self):
        rows = [dict(sample=0, native_max_diff=0, official_max_diff=0,
                     official_output_error_excess=0, performance_ratio=1),
                dict(sample=50, native_max_diff=0, official_max_diff=0,
                     official_output_error_excess=0, performance_ratio=1.06)]
        self.assertEqual(probe.parity_failures(rows, text=True), [
            'sample 50: performance_ratio=1.06 requires <= 1.05'])

    def test_nonfinite_or_missing_metrics_cannot_pass_parity(self):
        row = dict(sample=0, native_max_diff=0, official_max_diff=0,
                   official_output_error_excess=0, performance_ratio=1)
        for value in (float('nan'), float('inf'), None):
            with self.subTest(value=value):
                candidate = dict(row)
                if value is None:
                    del candidate['performance_ratio']
                else:
                    candidate['performance_ratio'] = value
                self.assertTrue(probe.parity_failures([candidate], text=True))

    def test_text_local_error_is_rejected_despite_passing_absolute_bound(self):
        row = dict(sample=0, native_max_diff=.00002, official_max_diff=.002,
                   official_output_error_excess=.00002, performance_ratio=1)
        self.assertEqual(probe.parity_failures([row], text=True), [
            'sample 0: official_output_error_excess=2e-05 requires <= 1e-05'])

    def test_numeric_policy_change_invalidates_upstream_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            policy = root/'accuracy-policy.md'
            policy.write_text('original numeric policy')
            snapshot = root/'snapshot.json'
            snapshot.write_text(json.dumps(dict(
                commit='pinned',files={},ggml_commit='ggml-pinned',
                accuracy_policy=dict(source=policy.name,
                                     sha256=probe.digest(policy)))))
            policy.write_text('changed numeric policy')
            with mock.patch.object(probe,'SNAPSHOT',snapshot), \
                 mock.patch.object(probe.subprocess,'check_output',
                                   return_value='pinned\n'):
                with self.assertRaisesRegex(RuntimeError,'numeric policy'):
                    probe.verify_upstream(root,root/'native',{},'cpu')

    def test_cuda_uses_recorded_runtime_without_candidate_ggml_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);library=root/'toolkit';library.mkdir()
            names=('cudart','cublas','cublasLt')
            for item in names:(library/('lib'+item+'.so')).touch()
            (root/'CMakeCache.txt').write_text(''.join(
                f'CUDA_{item}_LIBRARY:FILEPATH={library}/lib{item}.so\n'
                for item in names))
            candidate,native=probe.runtime_environments(
                {'LD_LIBRARY_PATH':'/candidate/bin','GGML_BACKEND_PATH':'/candidate/bin'},root,'cuda')
            self.assertEqual(native['LD_LIBRARY_PATH'],str(library))
            self.assertNotIn('GGML_BACKEND_PATH',native)
            self.assertEqual(candidate['LD_LIBRARY_PATH'],str(library)+':/candidate/bin')

    def test_summary_keeps_worst_error_drop_and_agreement(self):
        rows=[dict(latency_ms=10, native_max_diff=0,
                   official_pair_recall_drop=0, native_top1_agreement=1),
              dict(latency_ms=12, native_max_diff=.02,
                   official_pair_recall_drop=.1, native_top1_agreement=.9)]
        summary=probe.summarize_rows(rows)
        self.assertEqual(summary['native_max_diff'], .02)
        self.assertEqual(summary['official_pair_recall_drop'], .1)
        self.assertEqual(summary['native_top1_agreement'], .9)

    def edge(self, **changes):
        row = dict(subject=0, object=1, predicate_id=7,
                   predicate='holding', score=0.75)
        row.update(changes)
        return row

    def test_same_pair_with_changed_predicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'edge set'):
            probe.compare_edges({'predictions': [self.edge()]},
                                {'relations': [self.edge(predicate_id=8)]})

    def test_changed_predicate_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'predicate name'):
            probe.compare_edges({'predictions': [self.edge()]},
                                {'relations': [self.edge(predicate='above')]})

    def test_decomposition_stream_drift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'spatial edge set'):
            probe.compare_edges({'spatial': [self.edge()]}, {'spatial': []})

    def test_score_error_is_reported_across_all_streams(self):
        result = probe.compare_edges(
            {'predictions': [self.edge()], 'semantic': [self.edge()]},
            {'relations': [self.edge()],
             'semantic': [self.edge(score=0.5)]})
        self.assertEqual(result['score_max_diff'], 0.25)

    def test_symlinked_upstream_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'cpp_ggml').mkdir()
            (root / 'other').mkdir()
            (root / 'cpp_ggml/build-cpu').symlink_to(root / 'other')
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                probe.upstream_build_dir(root, 'cpu')

    def test_build_from_different_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            build = root / 'cpp_ggml/build-cpu'
            build.mkdir(parents=True)
            (build / 'CMakeCache.txt').write_text(
                'CMAKE_HOME_DIRECTORY:INTERNAL=/another/source\n')
            with self.assertRaisesRegex(RuntimeError, 'different source'):
                probe.upstream_build_dir(root, 'cpu')

    def test_selected_build_still_requires_pinned_source_and_real_executable(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            build = root / 'matching-flags'
            build.mkdir()
            (root / 'cpp_ggml').mkdir()
            cache = build / 'CMakeCache.txt'
            cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={root}/cpp_ggml\n')
            binary = build / 'relateanything-ggml'
            with self.assertRaisesRegex(RuntimeError, 'executable is missing'):
                probe.upstream_build_dir(root, 'cpu', build)
            binary.touch()
            self.assertEqual(probe.upstream_build_dir(root, 'cpu', build), build)
            cache.write_text('CMAKE_HOME_DIRECTORY:INTERNAL=/another/source\n')
            with self.assertRaisesRegex(RuntimeError, 'different source'):
                probe.upstream_build_dir(root, 'cpu', build)

    def test_selected_binary_must_match_the_requested_backend(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps(dict(commit='pinned', files={},
                                                ggml_commit='ggml-pinned')))
            info = json.dumps(dict(backend='cuda', ggml_commit='ggml-pinned',
                                   source=dict(files={})))
            with mock.patch.object(probe, 'SNAPSHOT', snapshot), \
                 mock.patch.object(probe.subprocess, 'check_output',
                                   side_effect=['pinned\n', info]):
                with self.assertRaisesRegex(RuntimeError, 'different backend'):
                    probe.verify_upstream(root, root / 'native', {}, 'cpu')


if __name__ == '__main__':
    unittest.main()
