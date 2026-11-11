# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Exercise the production CMake probe against real partial HTTP transfers."""

import http.server
import pathlib
import socket
import struct
import subprocess
import tempfile
import threading
import unittest

HELPER = pathlib.Path(
    __file__).resolve().parents[1] / 'check_connectivity.cmake'


class RealSenseConnectivityTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.requests = 0
        self.reset_count = 0
        self.payload = b'RealSense firmware service is available\n'
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):

            def do_GET(self):
                owner.requests += 1
                self.send_response(200)
                self.send_header('Content-Length', str(len(owner.payload)))
                self.end_headers()
                if owner.requests <= owner.reset_count:
                    self.wfile.write(owner.payload[:8])
                    self.wfile.flush()
                    self.connection.setsockopt(socket.SOL_SOCKET,
                                               socket.SO_LINGER,
                                               struct.pack('ii', 1, 0))
                    self.connection.close()
                    self.close_connection = True
                    return
                self.wfile.write(owner.payload)

            def log_message(self, *_):
                pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port) + '/probe'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def run_probe(self):
        driver = self.root / 'probe.cmake'
        destination = self.root / 'response'
        driver.write_text('include("' + HELPER.as_posix() + '")\n' +
                          'cloudviewer_check_realsense_connectivity("' +
                          self.url + '" "' + destination.as_posix() + '")\n')
        return subprocess.run(['cmake', '-P', str(driver)],
                              capture_output=True,
                              text=True,
                              timeout=60)

    def test_success_needs_one_request(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.requests, 1)
        self.assertEqual((self.root / 'response').read_bytes(), self.payload)

    def test_partial_reset_is_retried_from_start(self):
        self.reset_count = 1
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.requests, 2)
        self.assertEqual((self.root / 'response').read_bytes(), self.payload)

    def test_exhaustion_fails_without_partial_file(self):
        self.reset_count = 100
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.requests, 5)
        self.assertIn('unavailable after 5 attempts', result.stderr)
        self.assertFalse((self.root / 'response').exists())


if __name__ == '__main__':
    unittest.main()
