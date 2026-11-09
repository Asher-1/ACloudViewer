"""Process isolation shared by independent Paddle accuracy/performance probes."""

import os
from runtime_environment import loader_variable, reference_runtime_environment


def source_environment(python):
    environment = reference_runtime_environment(os.environ)
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
    variable = loader_variable()
    runtime_paths = list(map(str, candidates))
    runtime_paths.extend([environment[variable]] if environment.get(variable) else [])
    environment[variable] = os.pathsep.join(runtime_paths)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment
