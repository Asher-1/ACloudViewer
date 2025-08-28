# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os as _os
from cloudViewer import _build_config

if _build_config['BUNDLE_CLOUDVIEWER_ML']:
    if 'CLOUDVIEWER_ML_ROOT' in _os.environ:
        from ml3d.torch.models import *
    else:
        from cloudViewer._ml3d.torch.models import *
