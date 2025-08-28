# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import cloudViewer as cv3d
import numpy as np
import pytest

if os.name == 'nt':
    address = 'tcp://127.0.0.1:51455'
else:
    address = 'ipc:///tmp/open3d_ipc'


def test_external_visualizer():
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    # create dummy receiver which will receive all data
    receiver = cv3d.io.rpc._DummyReceiver(address=address)
    receiver.start()

    # create ev with the same address
    ev = cv3d.visualization.ExternalVisualizer(address=address)

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
    assert ev.set(obj=[(pcd, 'pcd', 1), (mesh, 'mesh', 2), (camera, 'camera',
                                                            3)])

    # test other commands
    ev.set_time(10)
    ev.set_active_camera('camera')

    receiver.stop()
