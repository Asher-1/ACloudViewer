# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Exercise the Linux SDK installer with real curl transfers and broken sockets."""

import collections
import hashlib
import http.server
import io
import pathlib
import shlex
import socket
import struct
import subprocess
import tarfile
import tempfile
import threading
import unittest

INSTALLER = pathlib.Path(
    __file__).resolve().parents[1] / 'install_vulkan_linux.sh'
SOURCE = INSTALLER.read_text()
FUNCTIONS = SOURCE[SOURCE.index('version_at_least() {'):SOURCE.
                   index('\nmkdir -p "${ACLOUDVIEWER_LOCAL_ROOT}"\n')]


class VulkanDownloadTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.requests = collections.Counter()
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:xz') as output:
            data = b'# SDK fixture\n'
            entry = tarfile.TarInfo('sdk/setup-env.sh')
            entry.size = len(data)
            output.addfile(entry, io.BytesIO(data))
        self.archive = archive.getvalue()
        self.digest = hashlib.sha256(self.archive).hexdigest()
        self.always_reset = False
        self.bad_checksum = False
        self.invalid_checksum = False
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):

            def do_GET(self):
                owner.requests[self.path] += 1
                count = owner.requests[self.path]
                checksum = '/sha/' in self.path
                body = ((('0' * 64 if owner.bad_checksum else owner.digest) +
                         '\n').encode() if checksum else owner.archive)
                if checksum and owner.invalid_checksum:
                    body = b'invalid response\n'
                # A real TCP reset after a partial response yields curl exit 56.
                if owner.always_reset or count == 1:
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[:8])
                    self.wfile.flush()
                    self.connection.setsockopt(socket.SOL_SOCKET,
                                               socket.SO_LINGER,
                                               struct.pack('ii', 1, 0))
                    self.connection.close()
                    self.close_connection = True
                    return
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def run_shell(self, body):
        # Load the actual production functions without running host provisioning.
        # Only the endpoint and sleep are replaced; curl and archive/hash tools run.
        script = ('set -euo pipefail\n' + FUNCTIONS + '\n' +
                  'VULKAN_SDK_VERSION=test\n' + 'VULKAN_SDK_ROOT=' +
                  shlex.quote(str(self.root / 'sdk')) + '\n' +
                  'curl() { local arg; local args=(); for arg; do '
                  r'args+=("${arg/https:\/\/sdk.lunarg.com/' + self.url +
                  '}"); '
                  'done; command curl "${args[@]}"; }\n' + 'sleep() { :; }\n' +
                  body)
        return subprocess.run(['bash', '-c', script],
                              text=True,
                              capture_output=True)

    def test_reset_restarts_entire_file(self):
        destination = self.root / 'download'
        result = self.run_shell('download_vulkan_file ' +
                                shlex.quote(self.url + '/file') + ' ' +
                                shlex.quote(str(destination)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('curl 56, attempt 1/5', result.stderr)
        self.assertEqual(self.requests['/file'], 2)
        self.assertEqual(destination.read_bytes(), self.archive)

    def test_exhaustion_fails_and_removes_partial_file(self):
        self.always_reset = True
        destination = self.root / 'download'
        result = self.run_shell('download_vulkan_file ' +
                                shlex.quote(self.url + '/file') + ' ' +
                                shlex.quote(str(destination)))
        self.assertEqual(result.returncode, 56, result.stderr)
        self.assertEqual(self.requests['/file'], 5)
        self.assertFalse(destination.exists())

    def test_both_checksum_and_archive_retry_before_install(self):
        result = self.run_shell('install_vulkan_sdk_tarball')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'sdk/setup-env.sh').is_file())
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(count == 2 for count in self.requests.values()))

    def test_checksum_mismatch_rejects_archive(self):
        self.bad_checksum = True
        result = self.run_shell('install_vulkan_sdk_tarball')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256 mismatch', result.stderr)
        self.assertFalse((self.root / 'sdk').exists())

    def test_checksum_download_exhaustion_stops_install(self):
        self.always_reset = True
        result = self.run_shell('install_vulkan_sdk_tarball')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(next(iter(self.requests.values())), 5)
        self.assertFalse((self.root / 'sdk').exists())

    def test_malformed_checksum_stops_before_archive_download(self):
        self.invalid_checksum = True
        result = self.run_shell('install_vulkan_sdk_tarball')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid Vulkan SDK SHA256 response', result.stderr)
        self.assertEqual(len(self.requests), 1)
        self.assertFalse((self.root / 'sdk').exists())


if __name__ == '__main__':
    unittest.main()
