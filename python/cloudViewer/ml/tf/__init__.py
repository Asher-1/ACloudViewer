# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Tensorflow specific machine learning functions."""
import os as _os
from tensorflow import __version__ as _tf_version
from cloudViewer import _build_config

if not _build_config["Tensorflow_VERSION"]:
    raise Exception('CloudViewer was not built with Tensorflow support!')

_cv3d_tf_version = _build_config["Tensorflow_VERSION"].split('.')
if _tf_version.split('.')[:2] != _cv3d_tf_version[:2]:
    _cv3d_tf_version[2] = '*'  # Any patch level is OK
    match_tf_ver = '.'.join(_cv3d_tf_version)
    raise Exception(
        'Version mismatch: CloudViewer needs Tensorflow version {}, but'
        ' version {} is installed!'.format(match_tf_ver, _tf_version))

from . import layers
from . import ops

# put framework independent modules here for convenience
from . import configs
from . import datasets
from . import vis

# framework specific modules from cloudViewer-ml
from . import models
from . import modules
from . import pipelines
from . import dataloaders

# put contrib at the same level
from cloudViewer.ml import contrib
