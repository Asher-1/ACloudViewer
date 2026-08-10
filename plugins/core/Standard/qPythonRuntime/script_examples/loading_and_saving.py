"""
Example: load and save point clouds through the ACloudViewer I/O layer.

Demonstrates:
  * Loading a file WITHOUT auto-adding it to the DB, using
    ``pycc.FileIOFilter.LoadFromFile`` (returns ``None`` if loading fails),
    then adding it explicitly with ``CC.addToDB``.
  * Saving an entity to disk in multiple formats (headless, without any
    interactive dialog).

For auto-adding on load (and raising on error) you can instead use
``CC.loadFile(path, params)`` -- see ``script.py`` for an example.
"""
import os
import tempfile

import pycc

CC = pycc.GetInstance()

# Use the test data file shipped with the plugin
_script_dir = os.path.dirname(os.path.abspath(__file__))
_path = os.path.normpath(os.path.join(_script_dir, "..", "tests", "data", "a_cloud.bin"))

if not os.path.isfile(_path):
    raise FileNotFoundError(f"Test data file not found: {_path}")

# --- Loading ---------------------------------------------------------------
# Disable the interactive load dialog so this script can run headless.
load_params = pycc.FileIOFilter.LoadParameters()
load_params.parentWidget = CC.getMainWindow()
load_params.alwaysDisplayLoadDialog = False

try:
    # LoadFromFile does NOT add the entity to the DB; it returns the loaded
    # hierarchy, or None if loading failed.
    obj = pycc.FileIOFilter.LoadFromFile(_path, load_params)
except RuntimeError as exc:
    print(f"Failed to load '{_path}': {exc}")
    raise

if obj is None:
    raise RuntimeError(f"File '{_path}' could not be loaded")

print(f"Success to load the file: {obj.getName()}")
CC.addToDB(obj)

# --- Saving ----------------------------------------------------------------
# Disable the interactive save dialog so the script can run headless. Without
# this, some filters (e.g. the LAS one used for '.laz') open a modal dialog
# that blocks/cancels the save.
save_params = pycc.FileIOFilter.SaveParameters()
save_params.alwaysDisplaySaveDialog = False

# Write outputs to the system temporary directory instead of the process
# working directory (which depends on where ACloudViewer was launched from).
# This is the same location Qt uses for temporary files
# (QStandardPaths::TempLocation -> e.g. /tmp on Linux). The output names are
# derived from the input file so they stay unique and recognizable.
_out_dir = tempfile.gettempdir()
_out_base = os.path.splitext(os.path.basename(_path))[0]

# Save the whole loaded hierarchy as a binary cloud file.
bin_out = os.path.join(_out_dir, _out_base + "_saved.bin")
pycc.FileIOFilter.SaveToFile(obj, bin_out, save_params)
print(f"Saved to: {bin_out}")

# Save the first child (a point cloud) as LAS. Requires the LAS I/O plugin.
child = obj.getChild(0)
if child is not None:
    laz_out = os.path.join(_out_dir, _out_base + "_saved.laz")
    pycc.FileIOFilter.SaveToFile(child, laz_out, save_params)
    print(f"Saved to: {laz_out}")
else:
    print("Nothing to save as '.laz': loaded hierarchy has no child.")

print("Saving finished.")
