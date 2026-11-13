"""Record selected package versions without losing duplicate metadata evidence."""

import importlib.metadata as metadata
import re


def distribution_environment():
    candidates = {}
    for distribution in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        candidates.setdefault(name, set()).add(distribution.version)
    # Paddle's pkg_resources import adds vendored distributions to sys.path.
    # Last-write-wins enumeration can then report a vendored version instead
    # of the selected package, hiding changes to the installed dependency.
    selected = {name: metadata.version(name) for name in sorted(candidates)}
    return dict(environment=selected,
                distribution_candidates={name: sorted(versions)
                    for name, versions in sorted(candidates.items())})
