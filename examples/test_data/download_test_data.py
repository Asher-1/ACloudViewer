# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import sys
if sys.version_info < (3, 6):
    raise RuntimeError(
        "Python version must be >= 3.6, however, Python {}.{} is used.".format(
            sys.version_info[0], sys.version_info[1]))

import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_utils import download_all_files

if __name__ == "__main__":
    download_all_files()
