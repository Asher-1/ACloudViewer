# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as _cv3d
if _cv3d.__DEVICE_API__ == 'cuda':
    from cloudViewer.cuda.pybind.ml.contrib import *
else:
    from cloudViewer.cpu.pybind.ml.contrib import *
