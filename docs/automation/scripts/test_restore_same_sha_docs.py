#!/usr/bin/env python3
"""High-risk provenance and extraction contracts for catalogue-only publication."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import restore_same_sha_docs as owner

SHA = 'a' * 40
REPO = owner.REPOSITORY


class ReuseTests(unittest.TestCase):

    def fixture(self):
        run = {
            'id': 7,
            'head_sha': SHA,
            'head_branch': 'main',
            'event': 'push',
            'status': 'completed',
            'conclusion': 'success',
            'run_attempt': 1,
            'path': '.github/workflows/documentation.yml',
            'repository': {
                'id': 1,
                'full_name': REPO
            },
            'head_repository': {
                'id': 1,
                'full_name': REPO
            }
        }
        artifact = {
            'id': 9,
            'name': f'acloudviewer-{SHA}-docs',
            'expired': False,
            'size_in_bytes': 123,
            'digest': 'sha256:' + '0' * 64,
            'workflow_run': {
                'id': 7,
                'head_sha': SHA,
                'head_branch': 'main',
                'repository_id': 1,
                'head_repository_id': 1
            }
        }
        data = {
            'workflow_runs': [run],
            'jobs': [{
                'steps': [{
                    'name': 'Build documentation Docker image',
                    'conclusion': 'success'
                }, {
                    'name': 'Extract documentation from Docker',
                    'conclusion': 'success'
                }],
                'name': 'build-docs',
                'status': 'completed',
                'conclusion': 'success'
            }],
            'artifacts': [artifact]
        }

        def request(path):
            key = 'workflow_runs' if '/workflows/' in path else 'jobs' if '/jobs?' in path else 'artifacts'
            return {key: data[key]}

        return data, request

    def archive(self, root, entries=None, zipname=None):
        entries = entries if entries is not None else [
            ('index.html',
             f'<title>ACloudViewer main-{SHA[:9]} documentation</title>'.encode(
             ), None)
        ]
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w:gz') as tar:
            for name, body, kind in entries:
                info = tarfile.TarInfo(name)
                if kind:
                    info.type = kind
                    info.linkname = 'index.html'
                else:
                    info.size = len(body)
                tar.addfile(info, io.BytesIO(body) if not kind else None)
        path = root / 'artifact.zip'
        with zipfile.ZipFile(path, 'w') as zipped:
            zipped.writestr(zipname or f'acloudviewer-{SHA}-docs.tar.gz',
                            raw.getvalue())
        artifact = {
            'size_in_bytes': path.stat().st_size,
            'digest': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()
        }
        return path, artifact

    def test_exact_native_push_is_selected(self):
        data, request = self.fixture()
        self.assertEqual(owner.select_artifact(SHA, request)[1]['id'], 9)
        for field, bad in [('event', 'pull_request'), ('head_sha', 'b' * 40),
                           ('head_branch', 'feature'),
                           ('conclusion', 'failure')]:
            original = data['workflow_runs'][0][field]
            data['workflow_runs'][0][field] = bad
            with self.assertRaises(ValueError):
                owner.select_artifact(SHA, request)
            data['workflow_runs'][0][field] = original

    def test_failed_build_or_wrong_repository_refused(self):
        data, request = self.fixture()
        data['jobs'][0]['conclusion'] = 'failure'
        with self.assertRaises(ValueError):
            owner.select_artifact(SHA, request)
        data['jobs'][0]['conclusion'] = 'success'
        data['workflow_runs'][0]['head_repository']['full_name'] = 'fork/repo'
        with self.assertRaises(ValueError):
            owner.select_artifact(SHA, request)

    def test_push_job_without_native_steps_is_not_a_source(self):
        data, request = self.fixture()
        data['jobs'][0]['steps'][0]['conclusion'] = 'skipped'
        with self.assertRaises(ValueError):
            owner.select_artifact(SHA, request)

    def test_missing_expired_ambiguous_and_bad_digest_refused(self):
        for mutate in [
                lambda d: d.update(artifacts=[]),
                lambda d: d['artifacts'][0].update(expired=True), lambda d: d[
                    'artifacts'].append(copy.deepcopy(d['artifacts'][0])),
                lambda d: d['artifacts'][0].update(digest='missing'),
                lambda d: d['artifacts'][0]['workflow_run'].update(id=8)
        ]:
            data, request = self.fixture()
            mutate(data)
            with self.assertRaises(ValueError):
                owner.select_artifact(SHA, request)

    def test_api_failure_is_not_an_empty_native_build(self):

        def failure(path):
            raise RuntimeError('HTTP 404')

        with self.assertRaises(RuntimeError):
            owner.select_artifact(SHA, failure)

    def test_valid_archive_restores_atomically_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, artifact = self.archive(root)
            dest = root / 'docs-output'
            owner.restore_archive(archive, artifact, SHA, dest)
            self.assertIn('main-aaaaaaaaa', (dest / 'index.html').read_text())
            with self.assertRaises(ValueError):
                owner.restore_archive(archive, artifact, SHA, dest)

    def test_symlink_output_is_refused_before_api_or_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'output').symlink_to(root / 'not-created',
                                         target_is_directory=True)
            with self.assertRaises(ValueError):
                owner.restore(SHA,
                              root / 'output',
                              request=lambda _: self.fail('must not query API'))
            self.assertFalse((root / 'not-created').exists())

    def test_digest_and_zip_identity_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, artifact = self.archive(root)
            dest = root / 'docs-output'
            artifact['digest'] = 'sha256:' + '0' * 64
            with self.assertRaises(ValueError):
                owner.restore_archive(archive, artifact, SHA, dest)
            archive, artifact = self.archive(root, zipname='../unsafe.tar.gz')
            with self.assertRaises(ValueError):
                owner.restore_archive(archive, artifact, SHA, dest)
            self.assertFalse(dest.exists())

    def test_unsafe_duplicate_and_missing_entry_fail_closed(self):
        for entries in [[('../index.html', b'x', None)],
                        [('/index.html', b'x', None)],
                        [('index.html', b'', tarfile.SYMTYPE)],
                        [('index.html', b'', tarfile.LNKTYPE)],
                        [('index.html', b'', tarfile.CHRTYPE)],
                        [('a\\b', b'x', None)],
                        [('index.html', b'x', None),
                         ('./index.html', b'x', None)],
                        [('other.html', b'x', None)],
                        [('index.html', b'main-bbbbbbbbb', None)]]:
            with self.subTest(
                    entries=entries), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive, artifact = self.archive(root, entries)
                dest = root / 'docs-output'
                with self.assertRaises(ValueError):
                    owner.restore_archive(archive, artifact, SHA, dest)
                self.assertFalse(dest.exists())

    def test_all_pages_are_checked(self):
        calls = []

        def request(path):
            calls.append(path)
            return {'rows': [1] * 100 if path.endswith('&page=1') else [2]}

        self.assertEqual(len(owner.pages('/path', 'rows', request)), 101)
        self.assertEqual(len(calls), 2)

    def test_workflow_reuse_cannot_enter_native_steps(self):
        import yaml
        workflow = yaml.safe_load(
            (Path(__file__).parents[3] /
             '.github/workflows/documentation.yml').read_text())
        steps = workflow['jobs']['build-docs']['steps']
        for name in [
                'Maximize build space', 'Set up Docker Buildx',
                'Build documentation Docker image',
                'Extract documentation from Docker',
                'Upload documentation artifact'
        ]:
            self.assertEqual(
                next(s for s in steps if s['name'] == name)['if'],
                'inputs.reuse_docs != true')
        self.assertIn(
            'inputs.reuse_docs != true',
            next(s for s in steps
                 if s['name'] == 'Update main-devel release')['if'])
        for name in [
                'Upload docs-output for deploy job',
                'Prepare the website for deployment',
                'Upload website-deploy artifact'
        ]:
            self.assertNotIn('reuse_docs',
                             next(s for s in steps if s['name'] == name)['if'])


if __name__ == '__main__':
    unittest.main()
