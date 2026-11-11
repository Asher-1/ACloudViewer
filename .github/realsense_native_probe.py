import collections
import http.server
import os
import pathlib
import shutil
import socket
import struct
import subprocess
import sys
import threading

ROOT = pathlib.Path.cwd()
OLD = '24f5fff46d62b7fb0b9179423cd944b79f642ee1'
TEMP = pathlib.Path(os.environ['RUNNER_TEMP']) / 'realsense-control'
TEMP.mkdir(exist_ok=True)
COUNTS = collections.Counter()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        COUNTS[self.path] += 1
        payload = b'firmware connectivity control\n'
        self.send_response(200)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        if COUNTS[self.path] == 1:
            self.wfile.write(payload[:8])
            self.wfile.flush()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                       struct.pack('ii', 1, 0))
            self.connection.close()
            self.close_connection = True
        else:
            self.wfile.write(payload)

    def log_message(self, *_):
        pass


def run(command, logfile):
    print('RUN', command, flush=True)
    with logfile.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    print(logfile.read_text(errors='replace')[-14000:], flush=True)
    return result.returncode


if len(sys.argv) > 1 and sys.argv[1] == 'rewrite':
    path = pathlib.Path(sys.argv[2])
    source = path.read_text()
    original = 'https://librealsense.intel.com/Releases/connectivity_check'
    assert original in source
    path.write_text(source.replace(original, sys.argv[3]))
    sys.exit(0)

server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    for mode in ('baseline', 'fixed'):
        components = ROOT / '3rdparty'
        if mode == 'baseline':
            components = TEMP / 'baseline-components'
            shutil.copytree(ROOT / '3rdparty/librealsense',
                            components / 'librealsense')
            original = subprocess.check_output(
                ['git', 'show', OLD + ':3rdparty/librealsense/librealsense.cmake'])
            (components / 'librealsense/librealsense.cmake').write_bytes(original)
        source = TEMP / (mode + '-harness')
        source.mkdir()
        build = TEMP / (mode + '-build')
        endpoint = 'http://127.0.0.1:' + str(server.server_port) + '/' + mode
        cmake = '''cmake_minimum_required(VERSION 3.10)
project(RealSenseCIControl LANGUAGES C CXX)
set(CloudViewer_3RDPARTY_DIR "@COMPONENTS@")
set(CloudViewer_INSTALL_LIB_DIR lib)
set(CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR "@TEMP@/downloads")
set(BUILD_CUDA_MODULE OFF)
set(GLIBCXX_USE_CXX11_ABI ON)
set(ExternalProject_CMAKE_ARGS_hidden
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release)
include("@COMPONENTS@/librealsense/librealsense.cmake")
ExternalProject_Add_Step(ext_librealsense inject_probe
    COMMAND "@PYTHON@" "@SCRIPT@" rewrite
        <SOURCE_DIR>/CMake/connectivity_check.cmake "@URL@"
    DEPENDEES patch DEPENDERS configure)
'''
        for key, value in {'COMPONENTS': components, 'TEMP': TEMP,
                           'PYTHON': sys.executable,
                           'SCRIPT': pathlib.Path(__file__).resolve(),
                           'URL': endpoint}.items():
            cmake = cmake.replace('@' + key + '@', str(value))
        (source / 'CMakeLists.txt').write_text(cmake)
        assert run(['cmake', '-S', str(source), '-B', str(build)],
                   TEMP / (mode + '-parent-configure.log')) == 0
        status = run(['cmake', '--build', str(build), '--target',
                      'ext_librealsense', '-j3'],
                     TEMP / (mode + '-build.log'))
        if mode == 'baseline':
            log = (TEMP / 'baseline-build.log').read_text(errors='replace')
            assert status != 0
            assert 'BUILD_WITH_TM2 requires IMPORT_DEPTH_CAM_FW' in log
            assert COUNTS['/baseline'] == 1
            print('BASELINE reproduced actual TM2/firmware configuration error',
                  flush=True)
        else:
            assert status == 0
            assert COUNTS['/fixed'] == 2, COUNTS
            child = build / 'librealsense/src/ext_librealsense-build'
            cache = (child / 'CMakeCache.txt').read_text()
            assert 'BUILD_WITH_TM2:BOOL=ON' in cache
            assert 'IMPORT_DEPTH_CAM_FW:BOOL=ON' in cache
            flags = (child / 'CMakeFiles/realsense2.dir/flags.make').read_text()
            assert 'WITH_TRACKING=1' in flags
            install = build / 'librealsense'
            assert list(install.rglob('librealsense2.a'))
            assert list(install.rglob('libfw.a'))
            print('FIXED built actual ARM64 RealSense + firmware + TM2; '
                  'one reset recovered on second request; pinned firmware hashes '
                  'retained', flush=True)
finally:
    server.shutdown()
    server.server_close()
    thread.join()
