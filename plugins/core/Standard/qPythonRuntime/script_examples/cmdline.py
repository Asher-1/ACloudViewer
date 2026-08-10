"""
Example: interact with the ACloudViewer runtime (command-line or GUI).

This script works in two modes:

* Headless (command-line) mode::

      ACloudViewer -SILENT -PYTHON_SCRIPT cmdline.py <file.bin>

  Here the runtime is the command-line interface, which can list the clouds /
  meshes that were passed on the command line and import files.

* GUI mode (run from the Python plugin): the runtime is the GUI instance,
  which mirrors the DB tree and imports files through the GUI loader.

``pycc.GetInstance()`` returns the runtime instance that matches the current
mode, so this single script runs unchanged in both environments.
"""
import os
import sys

import pycc

CC = pycc.GetInstance()
if CC is None:
    raise RuntimeError(
        "Could not obtain the ACloudViewer runtime instance.\n"
        "Run this script headless with:\n"
        "  ACloudViewer -SILENT -PYTHON_SCRIPT cmdline.py <file>"
    )

if isinstance(CC, pycc.ccCommandLineInterface):
    # --- Headless / command-line mode --------------------------------------
    print("Number of loaded Clouds: {}".format(len(CC.clouds())))
    print("Number of loaded Meshes: {}".format(len(CC.meshes())))

    print(sys.argv)

    if len(sys.argv) > 2:
        # importFile requires a GlobalShiftOptions (the mode controls how the
        # file coordinates are shifted on load).
        opts = pycc.ccCommandLineInterface.GlobalShiftOptions()
        CC.importFile(sys.argv[1], opts)
else:
    # --- GUI mode (Python plugin) ------------------------------------------
    root = CC.dbRootObject()
    print("Running inside the ACloudViewer GUI (Python plugin).")
    print("DB tree root: '{}' ({} entities recursively)".format(
        root.getName(), root.getChildCountRecursive()))

    print(sys.argv)

    if len(sys.argv) > 2:
        filepath = sys.argv[1]
        if os.path.isfile(filepath):
            params = pycc.FileIOFilter.LoadParameters()
            params.parentWidget = CC.getMainWindow()
            params.alwaysDisplayLoadDialog = False
            CC.loadFile(filepath, params)
        else:
            print("File not found: {}".format(filepath))
