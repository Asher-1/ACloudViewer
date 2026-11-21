#!/usr/bin/env python3
"""Offline contracts for safe downloads_data.json recovery."""
import io
import json
import subprocess
import threading
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from fetch_downloads_data import API_URL, PAGES_URL, FetchError, _request, fetch_existing

VALID = json.dumps({
    'generated_at': 'fixture',
    'version_metadata': [{
        'value': 'v3.9.4',
        'release_date': '2026-01-01'
    }],
    'download_data': {
        'v3.9.4': {
            'linux': {
                'app': {
                    'cpu': {
                        'amd64': {
                            'url': 'asset'
                        }
                    }
                }
            }
        }
    },
}).encode()


def response(requests, statuses):

    def fetch(url, token):
        requests.append(url)
        status, body = statuses[url].pop(0)
        if isinstance(body, Exception):
            raise body
        return status, body

    return fetch


class RequestContracts(unittest.TestCase):

    def success(self):
        response = io.BytesIO(VALID)
        response.status = 200
        return response

    def test_transient_http_and_transport_failures_retry_then_succeed(self):
        for failure in [
                HTTPError(API_URL, 429, 'limited', {}, None),
                HTTPError(API_URL, 503, 'offline', {}, None),
                URLError('offline'),
                TimeoutError('timed out'),
                IncompleteRead(b'partial', 100)
        ]:
            with self.subTest(failure=type(failure).__name__):
                opener = Mock(side_effect=[failure, self.success()])
                sleep = Mock()
                self.assertEqual(
                    _request(API_URL, 'secret', opener=opener, sleep=sleep),
                    (200, VALID))
                self.assertEqual(opener.call_count, 2)
                sleep.assert_called_once_with(1)
                self.assertEqual(opener.call_args.kwargs['timeout'], 30)

    def test_retry_exhaustion_is_bounded(self):
        opener = Mock(side_effect=TimeoutError('offline'))
        sleep = Mock()
        with self.assertRaisesRegex(FetchError, 'after 3 attempts'):
            _request(API_URL, '', opener=opener, sleep=sleep)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual([x.args for x in sleep.call_args_list], [(1,), (2,)])

    def test_permission_errors_fail_without_retry_and_404_is_absence(self):
        for status in [401, 403]:
            with self.subTest(status=status):
                opener = Mock(
                    side_effect=HTTPError(API_URL, status, 'denied', {}, None))
                sleep = Mock()
                with self.assertRaisesRegex(FetchError, f'HTTP {status}'):
                    _request(API_URL, 'secret', opener=opener, sleep=sleep)
                self.assertEqual(opener.call_count, 1)
                sleep.assert_not_called()
        opener = Mock(side_effect=HTTPError(API_URL, 404, 'missing', {}, None))
        self.assertEqual(_request(API_URL, '', opener=opener), (404, b''))

    def test_real_redirect_does_not_forward_authorization(self):
        received = []

        class Handler(BaseHTTPRequestHandler):

            def do_GET(self):
                received.append((self.path, self.headers.get('Authorization')))
                if self.path == '/start':
                    self.send_response(302)
                    self.send_header('Location', '/destination')
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(VALID)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/start'
            self.assertEqual(_request(url, 'redirect-test'), (200, VALID))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertEqual(received, [('/start', 'Bearer redirect-test'),
                                    ('/destination', None)])


class ExistingDownloadsRecovery(unittest.TestCase):

    def test_release_asset_is_used_and_pages_is_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            requests, run = [], Mock()

            def fake_run(args, **kwargs):
                Path(args[-1], 'downloads_data.json').write_bytes(VALID)
                return Mock(returncode=0)

            run.side_effect = fake_run
            source = fetch_existing(
                Path(temp) / 'out.json',
                request=response(
                    requests, {
                        API_URL: [(200,
                                   json.dumps({
                                       'assets': [{
                                           'name': 'downloads_data.json'
                                       }]
                                   }).encode())]
                    }),
                run=run)
            self.assertEqual(source, 'main-devel release asset')
            self.assertEqual(requests, [API_URL])
            self.assertEqual(json.loads((Path(temp) / 'out.json').read_bytes()),
                             json.loads(VALID))

    def test_missing_asset_uses_valid_pages_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            requests = []
            source = fetch_existing(
                Path(temp) / 'out.json',
                request=response(
                    requests, {
                        API_URL: [
                            (200, json.dumps({
                                'assets': [{
                                    'name': 'other.zip'
                                }]
                            }).encode())
                        ],
                        PAGES_URL: [(200, VALID)]
                    }))
            self.assertEqual(source, 'existing GitHub Pages file')
            self.assertEqual(requests, [API_URL, PAGES_URL])
            self.assertEqual((Path(temp) / 'out.json').read_bytes(), VALID)

    def test_empty_placeholder_only_after_two_explicit_not_founds(self):
        with tempfile.TemporaryDirectory() as temp:
            requests = []
            path = Path(temp) / 'out.json'
            source = fetch_existing(path,
                                    request=response(
                                        requests, {
                                            API_URL: [(404, b'')],
                                            PAGES_URL: [(404, b'')]
                                        }))
            self.assertIn('release asset absent and Pages returned 404', source)
            self.assertEqual(requests, [API_URL, PAGES_URL])
            self.assertEqual(json.loads(path.read_bytes()), {
                'generated_at': '',
                'version_metadata': [],
                'download_data': {}
            })

    def test_release_rate_limit_fails_without_pages_fallback_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            path.write_text('known good')
            requests = []
            with self.assertRaisesRegex(FetchError, 'unexpected HTTP 429'):
                fetch_existing(path,
                               request=response(requests,
                                                {API_URL: [(429, b'limited')]}))
            self.assertEqual(requests, [API_URL])
            self.assertEqual(path.read_text(), 'known good')

    def test_declared_asset_download_failure_preserves_catalogue(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            path.write_text('known good')
            run = Mock(return_value=Mock(returncode=1))
            release = (200,
                       json.dumps({
                           'assets': [{
                               'name': 'downloads_data.json'
                           }]
                       }).encode())
            with self.assertRaisesRegex(FetchError, 'download failed'):
                fetch_existing(path,
                               request=response([], {API_URL: [release]}),
                               run=run,
                               sleep=Mock())
            self.assertEqual(path.read_text(), 'known good')
            self.assertEqual(run.call_count, 3)

    def test_html_error_page_is_not_accepted_as_pages_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            path.write_text('known good')
            with self.assertRaisesRegex(FetchError, 'not valid JSON'):
                fetch_existing(
                    path,
                    request=response(
                        [], {
                            API_URL: [(404, b'')],
                            PAGES_URL: [(200, b'<html>proxy failure</html>')]
                        }))
            self.assertEqual(path.read_text(), 'known good')

    def test_download_retry_uses_fresh_directory_and_bounded_process(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            directories = []

            def download(args, **kwargs):
                self.assertEqual(kwargs['timeout'], 60)
                directory = Path(args[-1])
                directories.append(directory)
                target = directory / 'downloads_data.json'
                self.assertFalse(target.exists())
                if len(directories) == 1:
                    target.write_bytes(b'partial')
                    raise subprocess.TimeoutExpired(args, 60)
                target.write_bytes(VALID)
                return Mock(returncode=0)

            release = (200,
                       json.dumps({
                           'assets': [{
                               'name': 'downloads_data.json'
                           }]
                       }).encode())
            fetch_existing(path,
                           request=response([], {API_URL: [release]}),
                           run=download,
                           sleep=Mock())
            self.assertEqual(len(set(directories)), 2)
            self.assertEqual(path.read_bytes(), VALID)

    def test_pages_never_receives_github_token(self):
        calls = []

        def request(url, token):
            calls.append((url, token))
            return (404, b'') if url == API_URL else (200, VALID)

        with tempfile.TemporaryDirectory() as temp:
            fetch_existing(Path(temp) / 'out.json',
                           token='github-secret',
                           request=request)
        self.assertEqual(calls, [(API_URL, 'github-secret'), (PAGES_URL, '')])

    def test_schema_version_mismatch_and_duplicate_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            path.write_bytes(VALID)
            bad = json.dumps({
                'version_metadata': [{
                    'value': 'v1'
                }],
                'download_data': {}
            }).encode()
            with self.assertRaisesRegex(FetchError, 'mismatched versions'):
                fetch_existing(path,
                               request=response([], {
                                   API_URL: [(404, b'')],
                                   PAGES_URL: [(200, bad)]
                               }))
            release = json.dumps({
                'assets': [{
                    'name': 'downloads_data.json'
                }] * 2
            }).encode()
            with self.assertRaisesRegex(FetchError, 'duplicate'):
                fetch_existing(path,
                               request=response([],
                                                {API_URL: [(200, release)]}))
            self.assertEqual(path.read_bytes(), VALID)

    def test_atomic_replace_failure_preserves_previous_file_and_cleans_temp(
            self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            path.write_text('previous')
            request = response([], {
                API_URL: [(404, b'')],
                PAGES_URL: [(200, VALID)]
            })
            with patch('fetch_downloads_data.os.replace',
                       side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    fetch_existing(path, request=request)
            self.assertEqual(path.read_text(), 'previous')
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_schema_mismatch_is_not_accepted_from_release(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'out.json'
            path.write_text('known good')
            run = Mock()

            def bad_asset(args, **kwargs):
                Path(args[-1], 'downloads_data.json').write_text('{}')
                return Mock(returncode=0)

            run.side_effect = bad_asset
            release = (200,
                       json.dumps({
                           'assets': [{
                               'name': 'downloads_data.json'
                           }]
                       }).encode())
            with self.assertRaisesRegex(FetchError, 'version_metadata'):
                fetch_existing(path,
                               request=response([], {API_URL: [release]}),
                               run=run)
            self.assertEqual(path.read_text(), 'known good')


if __name__ == '__main__':
    unittest.main()
