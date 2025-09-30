# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer
if cloudViewer.__DEVICE_API__ == 'cuda':
    if cloudViewer._build_config["BUILD_GUI"]:
        from cloudViewer.cuda.pybind.visualization import gui
    from cloudViewer.cuda.pybind.visualization import *
else:
    if cloudViewer._build_config["BUILD_GUI"]:
        from cloudViewer.cpu.pybind.visualization import gui
    from cloudViewer.cpu.pybind.visualization import *

from ._external_visualizer import *
from .draw_plotly import get_plotly_fig
from .draw_plotly import draw_plotly
from .draw_plotly import draw_plotly_server
from .to_mitsuba import to_mitsuba

if cloudViewer._build_config["BUILD_GUI"]:
    from .draw import draw
