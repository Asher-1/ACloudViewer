# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import sys
import argparse
import cloudViewer

parser = argparse.ArgumentParser(
    description=
    "This module can be run from the command line to start an external visualizer window.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--external-vis",
    action="store_true",
    help="Starts the external visualizer with the RPC interface")

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

if args.external_vis:
    if cloudViewer._build_config['BUILD_GUI']:
        from .draw import draw
        draw(show_ui=True, rpc_interface=True)
    else:
        print(
            "CloudViewer must be build with BUILD_GUI=ON to start an external visualizer."
        )
