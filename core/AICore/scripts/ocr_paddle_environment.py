"""Process isolation shared by independent Paddle accuracy/performance probes."""

import os
import sys


def source_environment(python):
    environment = os.environ.copy()
    for key in ("LD_PRELOAD", "PYTHONPATH"):
        environment.pop(key, None)
    # Paddle 3.0.0 links duplicate glog static string owners. A host-supplied
    # heap-allocated logging string can be freed twice on process teardown.
    # Keep normal destructors and use the source framework's default logging.
    for key in list(environment):
        if key.startswith(("GLOG_", "MLOG_")):
            environment.pop(key)
    root = python.absolute().parent.parent
    candidates = list(root.glob("lib/python*/site-packages/nvidia/*/lib"))
    candidates += list(root.glob("Lib/site-packages/nvidia/*/bin"))
    variable = ("PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH"
                if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    environment[variable] = os.pathsep.join(map(str, candidates))
    environment["PYTHONNOUSERSITE"] = "1"
    return environment
