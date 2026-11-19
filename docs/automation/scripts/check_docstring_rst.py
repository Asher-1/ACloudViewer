"""Check every pybind docstring for RST diagnostics, without running Sphinx.

The real build takes minutes and stops at the first -W failure, so it can only
report one problem per round. This parses every docstring reachable from the
documented modules, runs each through docutils the same way autodoc does, and
prints the full remaining set in one pass.

Usage:
    PYTHONPATH=<build>/lib/python_package python3 docs/automation/scripts/check_docstring_rst.py
"""

from __future__ import annotations

import sys
import warnings

# Import the conf.py helpers so the test exercises exactly what the build uses.
sys.path.insert(0, "docs/source")

import docutils.core
import docutils.utils


def parse_like_sphinx(doc: str, google: bool = True, numpy: bool = True) -> list[str]:
    """Reproduce autodoc's docstring pipeline exactly as the build runs it.

    Order matters, in two ways that both caused wrong measurements here:

      1. prepare_docstring()      removes the common indent
      2. conf.py's process_docstring   (registered at priority 400)
      3. Napoleon                  converts Google/NumPy sections ("Args:")
      4. docutils parses it

    Step 2 runs *before* step 3 in the build: conf.py connects
    `autodoc-process-docstring` with priority=400 while napoleon registers at
    the default 500. An earlier version of this checker ran Napoleon first and
    then the transform, so it measured text the build never produces -- it
    reported "CLEAN" for docstrings the real build failed on.

    Napoleon is still needed because a checker that skips it reports every
    "Args:" block as malformed: `Args:` followed by deeper indentation looks
    like a broken definition list. That produced 106 phantom errors here.
    """
    from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring
    from sphinx.util.docstrings import prepare_docstring

    lines = prepare_docstring(doc)
    # conf.py's transform runs first, exactly as the build orders it.
    transform = _load_transform()
    if transform is not None:
        transform(None, None, "check", None, None, lines)
    if google:
        lines = str(GoogleDocstring(lines)).splitlines()
    if numpy:
        lines = str(NumpyDocstring(lines)).splitlines()
    return lines


def collect_docstrings(root_modules):
    """Yield (qualified_name, docstring) for every callable with a __doc__."""
    import types

    seen = set()
    stack = list(root_modules)
    while stack:
        obj, name = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        is_module = isinstance(obj, types.ModuleType)
        try:
            mod = getattr(obj, "__module__", "") or ""
        except Exception:
            mod = ""

        # Only report docstrings this project generates. Modules such as
        # cloudViewer.visualization re-export torch/tensorboard/matplotlib, and
        # their Sphinx-flavoured docstrings produce ~9000 "Unknown role" errors
        # that the real build resolves through its own role registry. Those
        # belong to another project; checking them is noise, not coverage.
        # `__module__` is the discriminator: pybind members report
        # 'cloudViewer.cuda.pybind.<...>', re-exported third-party callables
        # report 'torch' / 'numpy' / 'tensorboard'.
        doc = getattr(obj, "__doc__", None)
        if (mod.startswith("cloudViewer")
                and isinstance(doc, str) and doc.strip()):
            yield name, doc

        # Never walk third-party packages. `cloudViewer.visualization` re-exports
        # tensorboard, whose lazy modules run arbitrary import code inside
        # __dir__ and crash the walk. Only cloudViewer's own names are in scope.
        if is_module and not name.split(".")[0].startswith("cloudViewer"):
            continue
        if not (is_module or mod.startswith("cloudViewer")):
            continue
        try:
            attrs = dir(obj)
        except Exception:
            continue                # lazy/third-party __dir__ can raise
        for attr in attrs:
            if attr.startswith("__") and attr != "__dlpack__":
                continue
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if callable(child) or is_module:
                stack.append((child, f"{name}.{attr}"))


def diagnostics(text):
    """Return docutils WARNING+ messages for a block of text."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = docutils.core.publish_doctree(
            text,
            settings_overrides={
                "report_level": 2,
                "halt_level": 5,      # never raise, collect instead
                "warning_stream": False,
                "output_encoding": "unicode",
            },
        )
    out = []
    for node in tree.traverse():
        if node.tagname == "system_message" and node.get("level", 0) >= 2:
            out.append(node.astext().strip().splitlines()[0])
    return out


def _documented_roots():
    """Import every module the build actually documents.

    The roots used to be a hand-written list of five modules, which silently
    missed `cloudViewer.reconstruction` and others: the checker reported 0
    problems while the real build failed on
    `SequentialMatchingOptions.vocab_tree_path`. The build derives its pages
    from docs/documented_modules.txt, so the checker must read that same file
    instead of keeping a second, stale copy of the list.
    """
    import importlib
    import os

    root_pkg = "cloudViewer"
    pkg = importlib.import_module(root_pkg)
    roots = [(pkg, root_pkg)]

    listing = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "documented_modules.txt")
    for line in open(listing, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("cloudViewer"):
            continue
        try:
            mod = importlib.import_module(line)
        except Exception:
            continue                    # reported by the real build, not here
        roots.append((mod, line))
    return roots


def main() -> int:
    from sphinx.util.docstrings import prepare_docstring

    # conf.py imports sphinx, which is available; import just the helpers it
    # needs rather than executing the whole module.
    transform = _load_transform()
    if transform is None:
        print("❌ could not load conf.py")
        return 1

    roots = _documented_roots()

    total = bad = 0
    by_msg: dict[str, int] = {}
    samples: dict[str, tuple[str, str]] = {}
    for name, doc in collect_docstrings(roots):
        total += 1
        # Go through the SAME preparation autodoc uses (indent removal +
        # Napoleon). Skipping either stage produces phantom diagnostics.
        lines = parse_like_sphinx(doc)
        transform(None, None, name, None, None, lines)
        for msg in diagnostics("\n".join(lines)):
            bad += 1
            key = msg.split(":")[-1].strip() if ":" in msg else msg
            by_msg[key] = by_msg.get(key, 0) + 1
            samples.setdefault(key, (name, msg))

    print(f"docstrings checked: {total}")
    print(f"docstrings with remaining diagnostics: {bad}")
    for key, count in sorted(by_msg.items(), key=lambda kv: -kv[1]):
        name, msg = samples[key]
        print(f"  {count:4}x  {key[:88]}")
        print(f"        e.g. {name}")
    return 1 if bad else 0


def _load_transform():
    """Extract process_docstring (and every helper it calls) from conf.py.

    Collect by convention rather than by an explicit list: a name allowlist
    silently goes stale every time a helper is added, and the failure looks
    like a NameError instead of a missing-check.
    """
    import ast
    import types

    src = open("docs/source/conf.py", encoding="utf-8").read()
    tree = ast.parse(src)

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "process_docstring" not in funcs:
        print("missing process_docstring in conf.py")
        return None

    # Everything reachable from process_docstring: module-level functions plus
    # the names it references.
    wanted = {"process_docstring", "fix_namespace"}
    changed = True
    while changed:
        changed = False
        for name in list(wanted):
            node = funcs.get(name)
            if node is None:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in funcs:
                    if child.id not in wanted:
                        wanted.add(child.id)
                        changed = True

    nodes = [funcs[n] for n in wanted if n in funcs]
    mod = types.ModuleType("conf_helpers")
    mod.__dict__["__builtins__"] = __builtins__
    mod.__dict__["re"] = __import__("re")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<conf>", "exec"),
         mod.__dict__)
    return mod.process_docstring


if __name__ == "__main__":
    sys.exit(main())
