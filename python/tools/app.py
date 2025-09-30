# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys

import cloudViewer as cv3d


def main():
    args = [os.path.abspath(__file__)]
    if len(sys.argv) > 1:
        args.append(sys.argv[1])
    cv3d.visualization.app.run_viewer(args)


if __name__ == "__main__":
    main()
