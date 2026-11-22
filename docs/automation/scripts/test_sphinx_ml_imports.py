#!/usr/bin/env python3
"""Exercise Sphinx parent/worker ML import visibility without compiling C++."""
import ast
import importlib
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def load_bootstrap():
    path = Path(__file__).resolve().parents[2] / 'source' / 'conf.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_configure_ml_import_path')
    namespace = {'os': os, 'sys': sys}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_configure_ml_import_path']


def send_ml_type(connection, root):
    # Autodoc imports the dependency in a worker; its sys.path mutations are
    # process-local and do not make the defining module visible to the parent.
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    from ml3d._sphinx_pickle_probe import ModelType
    connection.send(ModelType)
    connection.close()


class SphinxMLImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        package = self.root / 'ml3d'
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / '_sphinx_pickle_probe.py').write_text('class ModelType:\n    pass\n', encoding='utf-8')
        self.path_patch = patch.object(sys, 'path', sys.path[:])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.module_patch = patch.dict(sys.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        for name in list(sys.modules):
            if name == 'ml3d' or name.startswith('ml3d.'):
                del sys.modules[name]
        self.env_patch = patch.dict(os.environ, {'CLOUDVIEWER_ML_ROOT': str(self.root)})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.bootstrap = load_bootstrap()

    def receive(self):
        context = multiprocessing.get_context('fork')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=send_ml_type, args=(sender, str(self.root)))
        process.start()
        sender.close()
        try:
            self.assertTrue(receiver.poll(10), 'worker did not return an ML type')
            return receiver.recv()
        finally:
            receiver.close()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
            self.assertEqual(process.exitcode, 0)

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'Sphinx Unix worker reproduction')
    def test_worker_only_path_cannot_be_unpickled_by_parent(self):
        self.assertNotIn(str(self.root), sys.path)
        with self.assertRaisesRegex(ModuleNotFoundError, 'ml3d'):
            self.receive()

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'Sphinx Unix worker reproduction')
    def test_bootstrap_makes_worker_type_importable_by_parent(self):
        self.bootstrap()
        result = self.receive()
        self.assertEqual(result.__module__, 'ml3d._sphinx_pickle_probe')
        self.assertEqual(result.__name__, 'ModelType')

    def test_configuration_registers_path_before_extensions(self):
        path = Path(__file__).resolve().parents[2] / 'source' / 'conf.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        calls = [n for n in tree.body if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == '_configure_ml_import_path']
        self.assertEqual(len(calls), 1)
        extensions = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'extensions'
                                  for t in n.targets))
        self.assertLess(calls[0].lineno, extensions.lineno)

    def test_unconfigured_dependency_does_not_change_path(self):
        del os.environ['CLOUDVIEWER_ML_ROOT']
        before = sys.path[:]
        self.bootstrap()
        self.assertEqual(sys.path, before)

    def test_bootstrap_is_idempotent(self):
        self.bootstrap()
        self.bootstrap()
        self.assertEqual(sys.path.count(str(self.root)), 1)

    def test_invalid_dependency_fails_before_starting_workers(self):
        os.environ['CLOUDVIEWER_ML_ROOT'] = str(self.root / 'missing')
        with self.assertRaisesRegex(RuntimeError, 'CLOUDVIEWER_ML_ROOT'):
            self.bootstrap()


if __name__ == '__main__':
    unittest.main()
