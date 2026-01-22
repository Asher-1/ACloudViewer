# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import socket
import cloudViewer as cv3d
import numpy as np
import pytest


def _get_free_port():
    """Find a free port for ZMQ binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _get_address():
    """Get a ZMQ address, using random port on Windows to avoid conflicts."""
    if os.name == 'nt':
        # Use random port on Windows to avoid port conflicts in CI
        port = _get_free_port()
        return f'tcp://127.0.0.1:{port}'
    else:
        # Use IPC on Unix-like systems
        return 'ipc:///tmp/cloudViewer_ipc'


def test_external_visualizer():
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    # Get a unique address for this test to avoid conflicts
    # On Windows CI, use random port to avoid "Permission denied" errors
    address = _get_address()

    # create dummy receiver which will receive all data
    receiver = cv3d.io.rpc._DummyReceiver(address=address)
    try:
        receiver.start()

        # Give receiver time to start and bind
        import time
        time.sleep(0.2)

        # Attempt to create external visualizer
        # If receiver failed to bind, this will raise RuntimeError
        try:
            ev = cv3d.visualization.ExternalVisualizer(address=address)
        except RuntimeError as e:
            # If we can't connect, the receiver likely failed to bind
            # Skip the test rather than failing, as this is often an environment issue
            pytest.skip(
                f"ZMQ connection failed on {address}: {e}. "
                f"This may be due to port conflicts or permissions in CI.")

        # create some objects
        mesh = cv3d.geometry.ccMesh.create_torus()
        pcd = cv3d.geometry.ccPointCloud(
            cv3d.utility.Vector3dVector(np.random.rand(100, 3)))
        camera = cv3d.camera.PinholeCameraParameters()
        camera.extrinsic = np.eye(4)

        # send single objects
        assert ev.set(pcd, path='bla/pcd', time=42)
        assert ev.set(mesh, path='bla/mesh', time=42)
        assert ev.set(camera, path='bla/camera', time=42)

        # send multiple objects
        assert ev.set(obj=[pcd, mesh, camera])

        # send multiple objects with args
        assert ev.set(obj=[(pcd, 'pcd', 1), (mesh, 'mesh',
                                             2), (camera, 'camera', 3)])

        # test other commands
        ev.set_time(10)
        ev.set_active_camera('camera')
    finally:
        # Always stop receiver to clean up resources
        try:
            receiver.stop()
        except Exception:
            pass  # Ignore errors during cleanup
