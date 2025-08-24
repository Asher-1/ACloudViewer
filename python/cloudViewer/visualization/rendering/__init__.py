# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

if "@BUILD_GUI@" == "ON":
    import cloudViewer
    if cloudViewer.__DEVICE_API__ == 'cuda':
        from cloudViewer.cuda.pybind.visualization.rendering import *
    else:
        from cloudViewer.cpu.pybind.visualization.rendering import *
else:
    print("CloudViewer was not compiled with BUILD_GUI, but script is importing "
          "cloudViewer.visualization.rendering")
