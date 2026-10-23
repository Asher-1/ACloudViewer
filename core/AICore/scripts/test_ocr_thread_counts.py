"""The common runner's zero thread value must reach both engines consistently."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_ocr_source_pipeline as source
import run_ocr_paddle_pipeline as paddle
import run_table_paddle_probe as table


class ThreadProtocolTests(unittest.TestCase):
    def test_source_child_receives_resolved_threads(self):
        for requested, expected in ((0, 4), (3, 3)):
            argv = ['probe', '--upstream', 'unused', '--assets', 'unused',
                    '--library', 'unused', '--backend', 'cuda', '--stem',
                    'ppocrv5-mobile', '--report', 'unused', '--threads', str(requested)]
            with self.subTest(requested=requested), patch.object(sys, 'argv', argv), \
                    patch.object(source, 'run_source_child', return_value=0) as child:
                self.assertEqual(source.main(), 0)
                self.assertEqual(child.call_args.args[0].threads, expected)

    def test_paddle_export_receives_resolved_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = root / 'pins.json'
            pins.write_text('{}')
            for requested, expected in ((0, 4), (3, 3)):
                argv = ['probe', '--assets', str(root), '--source-pins', str(pins),
                        '--source-root', str(root), '--library', 'unused', '--backend',
                        'cpu', '--language', 'ch', '--quant', 'f32', '--report', 'unused',
                        '--ground-truth', 'unused', '--ground-truth-sha256', 'unused',
                        '--image-root', str(root), '--threads', str(requested),
                        '--export-reference']
                with self.subTest(requested=requested), patch.object(sys, 'argv', argv), \
                        patch.object(paddle, 'verified_source', return_value={}), \
                        patch.object(paddle, 'export_reference', return_value=0) as export:
                    self.assertEqual(paddle.main(), 0)
                    self.assertEqual(export.call_args.args[0].threads, expected)

    def table_argv(self, root, requested):
        argv = ['probe']
        for name in ('program', 'parameters', 'config', 'input-index', 'model',
                     'source-python', 'probe'):
            argv += ['--' + name, 'unused']
        for name in ('program', 'parameters', 'config', 'input-index'):
            argv += ['--' + name + '-sha256', 'unused']
        return argv + ['--backend', 'cuda', '--max-diff', '.001', '--loc-max-diff',
                       '.001', '--threads', str(requested), '--report', str(root / 'report.json')]

    def test_table_run_receives_resolved_threads_and_negative_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for requested, expected in ((0, 4), (3, 3)):
                with self.subTest(requested=requested), \
                        patch.object(sys, 'argv', self.table_argv(root, requested)), \
                        patch.object(table, 'run', return_value={'exit_code': 0}) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(table.main(), 0)
                    self.assertEqual(run.call_args.args[0].threads, expected)
            with patch.object(sys, 'argv', self.table_argv(root, -1)), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(table, 'run') as run, self.assertRaises(SystemExit) as failure:
                table.main()
            self.assertEqual(failure.exception.code, 2)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
