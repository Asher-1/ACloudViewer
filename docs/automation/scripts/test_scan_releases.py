#!/usr/bin/env python3
"""Offline regressions for release inventory and the workflow's real filters."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import yaml

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('scan_releases',
                                              HERE / 'scan_releases.py')
scanner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scanner)
WORKFLOW = HERE.parents[2] / '.github/workflows/update-downloads.yml'


def response(rows):
    result = io.BytesIO(json.dumps(rows).encode())
    return result


class ReleaseInventoryTests(unittest.TestCase):

    def test_fetches_beyond_default_api_page_size(self):
        with patch.object(scanner,
                          'urlopen',
                          return_value=response([{
                              'id': i
                          } for i in range(75)])) as get:
            self.assertEqual(len(scanner.fetch_releases(100)), 75)
        self.assertIn('per_page=100&page=1', get.call_args.args[0].full_url)

    def test_paginates_and_honors_limit(self):
        with patch.object(scanner,
                          'urlopen',
                          side_effect=[
                              response([{
                                  'id': i
                              } for i in range(100)]),
                              response([{
                                  'id': i
                              } for i in range(100, 170)])
                          ]) as get:
            rows = scanner.fetch_releases(150)
        self.assertEqual(len(rows), 150)
        self.assertEqual(rows[-1]['id'], 149)
        self.assertIn('page=2', get.call_args.args[0].full_url)

    def test_partial_inventory_is_not_success(self):
        with patch.object(
                scanner,
                'urlopen',
                side_effect=[response([{}] * 100),
                             URLError('offline')]):
            with self.assertRaises(URLError):
                scanner.fetch_releases(150)

    def test_invalid_api_shape_fails(self):
        with patch.object(scanner,
                          'urlopen',
                          return_value=response({'message': 'error'})):
            with self.assertRaises(ValueError):
                scanner.fetch_releases(100)

    def test_failed_generation_preserves_existing_download_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            output = root / 'docs/downloads_data.json'
            output.write_text('existing catalogue')
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(scanner,
                                  'fetch_releases',
                                  side_effect=URLError('offline')):
                    self.assertEqual(scanner.main(), 1)
                self.assertEqual(output.read_text(), 'existing catalogue')
            finally:
                os.chdir(previous)

    def test_metadata_date_is_utc_and_does_not_change_release_inclusion(self):
        info = {
            tag: {
                'display_name': tag,
                'published_at': '2026-01-02T01:00:00+02:00',
                'assets': []
            } for tag in ['v3.9.4', 'main-devel', 'ubuntu-prebuild-deps']
        }
        metadata = scanner.generate_version_metadata(info)
        self.assertEqual([x['value'] for x in metadata], list(info))
        self.assertTrue(all(
            x['release_date'] == '2026-01-01' for x in metadata))


class WorkflowFiltersTests(unittest.TestCase):

    def run_filters(self, cleanup='0', maximum='0'):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        step = next(
            x for x in workflow['jobs']['update-downloads-data']['steps']
            if x['name'] == 'Generate latest downloads_data.json')
        script = step['run']
        self.assertNotIn('${{', script)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'docs/automation/scripts/scan_releases.py'
            stub.parent.mkdir(parents=True)
            metadata = [{
                'value': value,
                'release_date': date
            } for value, date in [('old', '2000-01-01'), (
                'new', '2999-01-01'), ('unknown', None)]]
            data = {
                'generated_at': 'fixture',
                'version_metadata': metadata,
                'download_data': {
                    x['value']: {
                        'fixture': x['value']
                    } for x in metadata
                }
            }
            stub.write_text(
                'from pathlib import Path\nPath("docs/downloads_data.json").write_text('
                + repr(json.dumps(data)) + ')\n')
            env = {
                **os.environ, 'EVENT_NAME': 'workflow_dispatch',
                'FORCE_UPDATE': 'false',
                'TRIGGER_DOCS_DEPLOY': 'false',
                'DRY_RUN': 'true',
                'CLEANUP_DAYS': cleanup,
                'MAX_VERSIONS': maximum
            }
            result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script],
                                    cwd=root,
                                    env=env,
                                    capture_output=True,
                                    text=True)
            return result, json.loads(
                (root / 'downloads_data.json.new').read_text())

    def test_max_versions_keeps_array_and_matching_data_keys(self):
        result, data = self.run_filters(maximum='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([x['value'] for x in data['version_metadata']],
                         ['new'])
        self.assertEqual(list(data['download_data']), ['new'])
        self.assertNotIn('download_links', data)

    def test_cleanup_removes_dated_old_release_and_keeps_unknown(self):
        result, data = self.run_filters(cleanup='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([x['value'] for x in data['version_metadata']],
                         ['new', 'unknown'])
        self.assertEqual(set(data['download_data']), {'new', 'unknown'})

    def test_input_is_validated_as_data_not_executed(self):
        for value in ['-1', '1.5', '$(false)', '1; false']:
            with self.subTest(value=value):
                result, _ = self.run_filters(maximum=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('non-negative integers', result.stderr)


class DocumentationWorkflowTests(unittest.TestCase):

    def step(self, name):
        path = WORKFLOW.with_name('documentation.yml')
        workflow = yaml.safe_load(path.read_text())
        return next(s for s in workflow['jobs']['build-docs']['steps']
                    if s.get('name') == name)

    def test_release_tag_is_data_and_rejected_before_path_use(self):
        step = self.step('Determine version and build mode')
        self.assertNotIn('${{', step['run'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'injected'
            tag = 'v1.2.3$(touch${IFS}' + str(marker) + ')'
            result = subprocess.run(
                ['bash', '-euo', 'pipefail', '-c', step['run']],
                env={
                    **os.environ, 'EVENT_NAME': 'release',
                    'EVENT_REF': 'refs/tags/' + tag,
                    'EVENT_RELEASE_TAG': tag,
                    'SOURCE_SHA': 'a' * 40,
                    'DEVELOPER_BUILD': 'OFF',
                    'GITHUB_OUTPUT': str(root / 'output')
                },
                capture_output=True,
                text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

    def test_valid_configuration_keeps_release_and_branch_deploy_contracts(
            self):
        script = self.step('Determine version and build mode')['run']
        for event, ref, tag, release, deploy in [
            ('release', 'refs/tags/v3.9.4', 'v3.9.4', 'true', 'true'),
            ('workflow_dispatch', 'refs/heads/main', '', 'false', 'true'),
            ('workflow_dispatch', 'refs/heads/feature', '', 'false', 'false')
        ]:
            with self.subTest(
                    event=event,
                    ref=ref), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'output'
                result = subprocess.run(
                    ['bash', '-euo', 'pipefail', '-c', script],
                    env={
                        **os.environ, 'EVENT_NAME': event,
                        'EVENT_REF': ref,
                        'EVENT_RELEASE_TAG': tag,
                        'SOURCE_SHA': 'a' * 40,
                        'DEVELOPER_BUILD': 'ON',
                        'GITHUB_OUTPUT': str(output)
                    },
                    capture_output=True,
                    text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = dict(
                    line.split('=', 1)
                    for line in output.read_text().splitlines())
                self.assertEqual(values['is_release'], release)
                self.assertEqual(values['should_deploy'], deploy)
                self.assertEqual(values['doc_version'], tag or 'latest')

    def test_main_dev_archive_name_is_unique_per_run_attempt(self):
        script = self.step('Update main-devel release')['run']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'gh'
            stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args=sys.argv[1:]
if args[:2]==['release','upload']:
    path=Path(args[3])
    if not path.is_file() or path.read_bytes()!=b'archive fixture':
        raise SystemExit(3)
    with open(os.environ['TEST_LOG'],'a') as stream: stream.write(path.name+'\\n')
    raise SystemExit(0)
if args[:2]==['release','view']:
    expected='acloudviewer-%s-%s-%s-docs.tar.gz' % (os.environ['SOURCE_SHA'],os.environ['SOURCE_RUN_ID'],os.environ['SOURCE_RUN_ATTEMPT'])
    print(json.dumps({'assets':[{'name':expected},{'name':'acloudviewer-oldsha-docs.tar.gz'}]}))
if args[:2]==['release','delete-asset']:
    raise SystemExit(0)
''')
            stub.chmod(0o755)
            (root / ('acloudviewer-' + 'a' * 40 + '-docs.tar.gz')
            ).write_bytes(b'archive fixture')
            names = []
            for attempt in ['1', '2']:
                result = subprocess.run(
                    ['bash', '-euo', 'pipefail', '-c', script],
                    cwd=root,
                    env={
                        **os.environ, 'PATH':
                            str(root) + os.pathsep + os.environ['PATH'],
                        'SOURCE_SHA':
                            'a' * 40,
                        'SOURCE_RUN_ID':
                            '12345',
                        'SOURCE_RUN_ATTEMPT':
                            attempt,
                        'TEST_LOG':
                            str(root / 'uploads')
                    },
                    capture_output=True,
                    text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                names.append('acloudviewer-' + 'a' * 40 + '-12345-' + attempt +
                             '-docs.tar.gz')
            self.assertNotEqual(*names)
            self.assertEqual((root / 'uploads').read_text().splitlines(), names)

    def test_pages_artifact_name_is_unique_for_each_workflow_attempt(self):
        workflow = yaml.safe_load(
            WORKFLOW.with_name('documentation.yml').read_text())
        steps = workflow['jobs']['deploy-docs']['steps']
        upload = next(
            s for s in steps if s.get('name') == 'Upload Pages artifact')
        deploy = next(
            s for s in steps if s.get('name') == 'Deploy to GitHub Pages')
        expected = 'github-pages-${{ github.run_id }}-${{ github.run_attempt }}'
        self.assertEqual(upload['with']['name'], expected)
        self.assertEqual(deploy['with']['artifact_name'], expected)

    def test_archive_source_fetch_is_narrow_and_fail_closed(self):
        workflow = yaml.safe_load(
            WORKFLOW.with_name('documentation.yml').read_text())
        step = next(s for s in workflow['jobs']['deploy-docs']['steps']
                    if s.get('name') == 'Restore archived source revisions')
        for state in [
                'missing', 'exists', 'fetch-fails', 'fetch-omits-ref',
                'no-history', 'invalid-version'
        ]:
            with self.subTest(
                    state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                versions = root / 'preserved-versions/documentation'
                versions.mkdir(parents=True)
                if state != 'no-history':
                    name = ('v3.9.4$(touch injected)'
                            if state == 'invalid-version' else 'v3.9.4')
                    (versions / name).mkdir()
                stub = root / 'git'
                stub.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
with open('calls', 'a') as stream: stream.write(' '.join(args) + '\\n')
state = os.environ['SOURCE_TEST_STATE']
if args[0] == 'rev-parse':
    if args[-1] != 'refs/tags/v3.9.4^{commit}': raise SystemExit(90)
    if state == 'exists' or Path('tag-ready').is_file():
        print('a' * 40)
        raise SystemExit(0)
    raise SystemExit(1)
if args == ['fetch', '--depth=1', '--no-tags', 'origin',
            'refs/tags/v3.9.4:refs/tags/v3.9.4']:
    if state == 'fetch-fails': raise SystemExit(13)
    if state != 'fetch-omits-ref': Path('tag-ready').touch()
    raise SystemExit(0)
raise SystemExit(91)
''')
                stub.chmod(0o755)
                result = subprocess.run(
                    ['bash', '-euo', 'pipefail', '-c', step['run']],
                    cwd=root,
                    env={
                        **os.environ, 'PATH':
                            str(root) + os.pathsep + os.environ['PATH'],
                        'SOURCE_TEST_STATE':
                            state
                    },
                    capture_output=True,
                    text=True)
                calls = ((root / 'calls').read_text().splitlines() if
                         (root / 'calls').exists() else [])
                if state in ['fetch-fails', 'fetch-omits-ref']:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('Archived source', result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                fetches = [call for call in calls if call.startswith('fetch ')]
                if state in ['missing', 'fetch-fails', 'fetch-omits-ref']:
                    self.assertEqual(fetches, [
                        'fetch --depth=1 --no-tags origin refs/tags/v3.9.4:refs/tags/v3.9.4'
                    ])
                else:
                    self.assertEqual(fetches, [])
                if state in ['no-history', 'invalid-version']:
                    self.assertEqual(calls, [])
                self.assertFalse((root / 'injected').exists())

    def test_failed_upload_retains_last_backup(self):
        self.run_backup(upload_fails=True)

    def test_successful_upload_only_prunes_previous_docs_backups(self):
        self.run_backup(upload_fails=False)

    def test_asset_verification_failure_preserves_old_backups(self):
        self.run_backup(upload_fails=False, fault='missing')
        self.run_backup(upload_fails=False, fault='view')

    def test_delete_failure_stops_pruning(self):
        self.run_backup(upload_fails=False, fault='delete')

    def run_backup(self, upload_fails, fault=''):
        script = self.step('Update main-devel release')['run']
        self.assertNotIn('${{', script)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'gh'
            stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args=sys.argv[1:]
log=Path(os.environ['TEST_LOG'])
with log.open('a') as f: f.write(' '.join(args)+'\\n')
if args[:2]==['release','upload']:
    if not Path(args[3]).is_file() or Path(args[3]).read_bytes()!=b'archive fixture':
        raise SystemExit(3)
    raise SystemExit(int(os.environ['FAIL_UPLOAD']))
if args[:2]==['release','view']:
    if os.environ['FAULT']=='view': raise SystemExit(4)
    archive='acloudviewer-%s-%s-%s-docs.tar.gz' % (os.environ['SOURCE_SHA'],os.environ['SOURCE_RUN_ID'],os.environ['SOURCE_RUN_ATTEMPT'])
    names=['acloudviewer-bbbbbbbb-docs.tar.gz','downloads_data.json','other.zip']
    if os.environ['FAULT']=='delete': names.append('acloudviewer-cccccccc-docs.tar.gz')
    if os.environ['FAULT']!='missing': names.insert(0,archive)
    print(json.dumps({'assets':[{'name':name} for name in names]}))
if args[:2]==['release','delete-asset'] and os.environ['FAULT']=='delete': raise SystemExit(5)
''')
            stub.chmod(0o755)
            (root / ('acloudviewer-' + 'a' * 40 + '-docs.tar.gz')
            ).write_bytes(b'archive fixture')
            result = subprocess.run(
                ['bash', '-euo', 'pipefail', '-c', script],
                cwd=root,
                env={
                    **os.environ, 'PATH':
                        str(root) + os.pathsep + os.environ['PATH'],
                    'SOURCE_SHA':
                        'a' * 40,
                    'SOURCE_RUN_ID':
                        '12345',
                    'SOURCE_RUN_ATTEMPT':
                        '2',
                    'TEST_LOG':
                        str(root / 'calls'),
                    'FAIL_UPLOAD':
                        '1' if upload_fails else '0',
                    'FAULT':
                        fault
                },
                capture_output=True,
                text=True)
            calls = (root / 'calls').read_text().splitlines()
            self.assertTrue(calls[0].startswith('release upload '))
            self.assertNotIn('--clobber', calls[0])
            deleted = [
                c for c in calls if c.startswith('release delete-asset ')
            ]
            if upload_fails or fault in ('missing', 'view'):
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(deleted, [])
            elif fault == 'delete':
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(deleted, [
                    'release delete-asset main-devel acloudviewer-bbbbbbbb-docs.tar.gz --yes'
                ])
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(deleted, [
                    'release delete-asset main-devel acloudviewer-bbbbbbbb-docs.tar.gz --yes'
                ])


class PlatformReleaseCleanupTests(unittest.TestCase):
    CURRENT = 'dddcdb92657254f9fe1ecb01351d458d489f8fa2'
    RECENT = [
        CURRENT, '4a6e4bb85f8368f959a8e4341d9548b3fb20a542',
        'fc450c9a53290558119a71c16c41bcc6ff72048b'
    ]

    def step(self):
        workflow = yaml.safe_load(
            WORKFLOW.with_name('update-release.yml').read_text())
        job = workflow['jobs']['clean-release']
        checkout = next(
            s for s in job['steps'] if s.get('uses') == 'actions/checkout@v4')
        self.assertEqual(checkout['with']['fetch-depth'], 3)
        return next(s['run']
                    for s in job['steps']
                    if s.get('name') == 'Clean old release assets')

    def run_cleanup(self, names, fault='', commits=None, payload=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inventory').write_text(
                json.dumps({'assets': [{
                    'name': n
                } for n in names]}) if payload is None else payload)
            (root / 'commits').write_text(
                '\n'.join(self.RECENT if commits is None else commits) + '\n')
            for tool in ['gh', 'git']:
                stub = root / tool
                stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open('calls', 'a') as stream: stream.write(json.dumps([Path(sys.argv[0]).name] + args) + '\\n')
fault = os.environ['CLEANUP_TEST_FAULT']
if Path(sys.argv[0]).name == 'git':
    if args != ['log', '--format=%H', '-n', '3']: raise SystemExit(91)
    if fault == 'git': raise SystemExit(12)
    print(Path('commits').read_text(), end='')
    raise SystemExit(0)
if args == ['release', 'view', 'main-devel', '--json', 'assets']:
    if fault == 'inventory': raise SystemExit(13)
    print(Path('inventory').read_text())
    raise SystemExit(0)
if args[:3] == ['release', 'delete-asset', 'main-devel']:
    if len(args) != 5 or args[-1] != '--yes': raise SystemExit(92)
    raise SystemExit(14 if fault == 'delete' else 0)
if args == ['release', 'view', 'main-devel']: raise SystemExit(0)
raise SystemExit(93)
''')
                stub.chmod(0o755)
            result = subprocess.run(
                ['bash', '-euo', 'pipefail', '-c',
                 self.step()],
                cwd=root,
                env={
                    **os.environ, 'PATH':
                        str(root) + os.pathsep + os.environ['PATH'],
                    'RUNNER_TEMP':
                        str(root),
                    'CLEANUP_TEST_FAULT':
                        fault
                },
                capture_output=True,
                text=True)
            calls = [
                json.loads(line)
                for line in (root / 'calls').read_text().splitlines()
            ]
            deleted = [
                args[4]
                for args in calls
                if args[:4] == ['gh', 'release', 'delete-asset', 'main-devel']
            ]
            self.assertFalse(list(root.glob('release-cleanup.*')))
            return result, deleted, calls

    def test_cleanup_protects_metadata_docs_shared_and_current_assets(self):
        protected = [
            'downloads_data.json', 'shared-assets.zip', 'notes with spaces.txt',
            'acloudviewer-' + self.CURRENT + '-45020940979-1-docs.tar.gz',
            'acloudviewer-6c3f6650-docs.tar.gz',
            'acloudviewer-v3.9.4-docs.tar.gz',
            'ACloudViewer-3.9.5-win-cpu-amd64.exe',
            'ACloudViewer-3.9.5+dddcdb92-win-cpu-amd64.exe',
            'ACloudViewer-3.9.5+' + self.CURRENT + '-mac-cpu-ARM64.dmg',
            'cloudviewer-3.9.5+dddcdb92-cp312-cp312-win_amd64.whl',
            'cloudviewer_cpu-3.9.5+4a6e4bb8-cp313-cp313-manylinux_2_35_x86_64.whl',
            'ACloudViewer-3.9.5+fc450c9a-ubuntu22.04-cuda-amd64.run'
        ]
        old = [
            'ACloudViewer-3.9.5+6c3f6650-win-cpu-amd64.exe',
            'ACloudViewer-3.9.5+6c3f6650-mac-cpu-ARM64.dmg',
            'ACloudViewer-3.9.5+6c3f6650-ubuntu20.04-cuda-amd64.run',
            'cloudviewer-3.9.5+6c3f6650-cp310-cp310-macosx_11_0_arm64.whl',
            'cloudviewer-3.9.5+6c3f6650-cp312-cp312-win_amd64.whl',
            'cloudviewer_cpu-3.9.5+6c3f6650-cp313-cp313-manylinux_2_35_x86_64.whl'
        ]
        result, deleted, _ = self.run_cleanup(protected + old)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(deleted, old)

    def test_cleanup_uses_stamp_boundaries_not_arbitrary_sha_substrings(self):
        wrong_place = 'ACloudViewer-3.9.5+6c3f6650-win-cpu-amd64-dddcdb92.exe'
        unknown = [
            'ACloudViewer-3.9.5+6c3f6650-win-cpu-amd64.exe.sha256', wrong_place,
            'unrelated-dddcdb92.zip',
            'cloudviewer-3.9.5+6c3f6650-cp312-abi3-win_amd64.whl'
        ]
        result, deleted, _ = self.run_cleanup(unknown)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(deleted, [])

    def test_cleanup_failures_cannot_delete_from_partial_or_invalid_inventory(
            self):
        asset = 'ACloudViewer-3.9.5+6c3f6650-win-cpu-amd64.exe'
        for fault in ['inventory', 'git']:
            with self.subTest(fault=fault):
                result, deleted, _ = self.run_cleanup([asset], fault=fault)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(deleted, [])
        for payload in [
                'not-json', '{"assets":null}',
                '{"assets":[{"name":"x"},{"name":"x"}]}',
                '{"assets":[{"name":123}]}'
        ]:
            with self.subTest(payload=payload):
                result, deleted, _ = self.run_cleanup([], payload=payload)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(deleted, [])
        for commits in [self.RECENT[:1], ['dddcdb92'] * 3]:
            with self.subTest(commits=commits):
                result, deleted, _ = self.run_cleanup([asset], commits=commits)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(deleted, [])

    def test_cleanup_delete_failure_stops_before_later_candidates(self):
        first = 'ACloudViewer-3.9.5+6c3f6650-win-cpu-amd64.exe'
        second = 'cloudviewer-3.9.5+6c3f6650-cp310-cp310-win_amd64.whl'
        result, deleted, calls = self.run_cleanup([first, second],
                                                  fault='delete')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(deleted, [first])
        self.assertNotIn(['gh', 'release', 'view', 'main-devel'], calls)


if __name__ == '__main__':
    unittest.main()
