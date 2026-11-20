# Strict Sphinx: parallel ML imports and docstrings

## Exact parent/worker import regression

Start with the complete failing exact-SHA job log as in [CI evidence](<exact-sha-ci.md>). In the observed run, `docs-guards`, all 28 notebooks and the GUI/Doxygen phases had progressed successfully; the first substantive failure was **full parallel Sphinx**, not Poisson, a warning to suppress, or a missing source file: `multiprocessing.connection.recv → _ForkingPickler.loads → ModuleNotFoundError: No module named 'ml3d'`. Locate external `ml3d` via `CLOUDVIEWER_ML_ROOT` and the real `python/cloudViewer` import path; a guessed source file `python/cloudViewer/_ml3d/__init__.py` was not present. Worker import of `cloudViewer` adds external ML to its process-local `sys.path`; that side effect cannot travel back to the Sphinx parent. A single-process import smoke test can mask this.

Using the verified venv, freshly built GUI package, runtime library path and valid external ML checkout from [native replay](<native-replay.md>), save this probe as `repro_sphinx_ml_pickle.py` under the run evidence directory. Invoke it in two separate Python processes with no global ML root on `PYTHONPATH` and no prior parent `cloudViewer` import. On Ubuntu explicitly use the fork context with a clean parent path, matching the worker/readback scenario; use bounded Pipe wait only for this probe. Negative returns 1 **only for** the expected missing `ml3d`; any child failure/no send also invalidates the control (the parent can see EOF rather than the expected traceback). A class instance is wrong: `Config` instance unpickling can hit its unrelated `__getattr__` recursion.

```python
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback


def worker(send):
    import cloudViewer
    from cloudViewer.ml.utils import Config
    assert Config.__module__.startswith('ml3d.')
    print('WORKER_TYPE', Config.__module__, Config.__name__, flush=True)
    send.send(Config)  # actual external type reference, not Config(...)
    send.close()


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ('worker-only', 'parent-first')
    root = Path(os.environ['CLOUDVIEWER_ML_ROOT']).resolve()
    assert (root / 'ml3d').is_dir()
    print('PARENT_SPEC_BEFORE', importlib.util.find_spec('ml3d'), flush=True)
    assert importlib.util.find_spec('ml3d') is None, 'ML is already visible in the parent'
    if mode == 'parent-first':
        sys.path.insert(0, str(root))
        assert importlib.util.find_spec('ml3d') is not None
    ctx = mp.get_context('fork')
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=worker, args=(send,))
    p.start()
    send.close()
    status = 0
    try:
        assert recv.poll(60), 'worker did not send'
        value = recv.recv()
        assert mode == 'parent-first', 'negative unexpectedly succeeded'
        assert isinstance(value, type) and value.__module__.startswith('ml3d.')
        print('PARENT_RECV_OK', value.__module__, value.__name__, flush=True)
    except ModuleNotFoundError as e:
        if mode != 'worker-only' or e.name != 'ml3d':
            raise
        traceback.print_exc()
        status = 1
    finally:
        recv.close()
        p.join(5)
        if p.is_alive():
            p.terminate()
            p.join()
        assert p.exitcode == 0, f'worker exit {p.exitcode}'
    sys.exit(status)
```

```bash
if python "$EVIDENCE/repro_sphinx_ml_pickle.py" worker-only > "$EVIDENCE/ml-before.log" 2>&1; then
    echo 'unexpected negative success' >&2; exit 1
else
    test "$?" -eq 1
    grep -F 'WORKER_TYPE ml3d.' "$EVIDENCE/ml-before.log"
    grep -F "ModuleNotFoundError: No module named 'ml3d'" "$EVIDENCE/ml-before.log"
fi
python "$EVIDENCE/repro_sphinx_ml_pickle.py" parent-first > "$EVIDENCE/ml-after.log" 2>&1
```

Fix the owner in `docs/source/conf.py`: register the resolved root on `sys.path` early, before loading Sphinx extensions or triggering autodoc, only when `CLOUDVIEWER_ML_ROOT` is set and points at a directory containing `ml3d/`; do not add duplicate entries, a machine-specific path, `ml3d/` itself, or an empty/current-directory path. With no configured external checkout leave the path unchanged; a bad root must not be inserted. Keep the existing bundled/optional path, ML API and parallelism. Do not rely on importing `cloudViewer` first in the parent as a hidden bootstrap.

Use or add `docs/automation/scripts/test_sphinx_ml_imports.py` as a **no-compilation** guard. Use a temp importable `ml3d` tree for deterministic class transport and exercise the production bootstrap (e.g. extract that function from the conf AST without importing the build). Cover parent-unconfigured vs registered class transport, invalid/unset root no-op, idempotence and bootstrapping before extensions (the observed guard had six tests). Run `python docs/automation/scripts/test_sphinx_ml_imports.py -v`; add the invocation to `docs-guards` in `.github/workflows/documentation.yml`, not the expensive compiled build. The synthetic guard does not replace real class controls or full Sphinx.

Run this with source and API docs for the completed GUI build, not a reduced source tree. Set `DOC_VERSION` and `DOC_RELEASE` to the actual phase values used by the helper. Keep any additional current workflow options. Use a fresh sibling doctree, not a warmed pickle from a prior invocation. Use a fresh HTML staging directory; seed it only with verified unchanged Doxygen `cpp_api/api/` and any other required non-Sphinx outputs from the current build. Reapply the normal plugin showcase asset copy/final entry checks if doing this phase directly rather than through the helper.

```bash
fresh=$(mktemp -d "$EVIDENCE/sphinx-ml.XXXXXX")
mkdir -p "$fresh/html/cpp_api"
test -f docs/_out/html/cpp_api/api/index.html
cp -a docs/_out/html/cpp_api/api "$fresh/html/cpp_api/"
python -m sphinx -E -a -j4 -W -b html -d "$fresh/doctrees" \
    -D "version=$DOC_VERSION" -D "release=$DOC_RELEASE" \
    docs/source "$fresh/html" > "$EVIDENCE/sphinx-ml-full.log" 2>&1
```

Capture exact exit code and `build succeeded`, verify full API/ML/C++ and entry files. If fresh Sphinx is still running, the log may not yet contain final entries; inspect the owned job rather than inferring failure from the missing deployment log. Combined with unchanged verified compiled/Notebook/Doxygen inputs this isolates the follow-up without another C++ build. A requested no-push replay stops after local evidence.

## GUI-only docstring RST failures

Use current GUI/WebRTC/ML/PyTorch package and run `python docs/automation/scripts/check_docstring_rst.py` to collect all reachable diagnostics; `python docs/automation/scripts/test_docstring_pipeline.py` exercises ordering and directives when present. First align the checker with real Sphinx:

1. `prepare_docstring()` removes common indentation, ignoring line 1.
2. `conf.py::process_docstring` at priority **400** transforms raw docstrings.
3. Napoleon at **500** converts Google/NumPy sections, including `Args:`.
4. `_second_pass_docstring` at **600** fixes the post-Napoleon field-list separator, then docutils parses. The handler must have `(app, what, name, obj, options, lines)`, not just `lines`.

Running the first transform after Napoleon can report clean while Sphinx fails. Skipping Napoleon produced 106 spurious indent errors. Traverse roots from `docs/documented_modules.txt` exactly as the generator does; fixed five roots missed `cloudViewer.reconstruction.SequentialMatchingOptions.vocab_tree_path`. Limit objects to `__module__` starting in `cloudViewer` (including pybind CUDA bindings); torch, TensorBoard, NumPy and matplotlib re-exports can cause ~9000 bogus roles/directives. Observed scope ~2300 correct vs 2800 fixed-root vs 14900 unfiltered. Unregistered `:eq:`, `:file:`, `:class:` and `todo` in standalone docutils should be checked in real Sphinx, not rewritten if Sphinx resolves them.

WebRTC example: the binding for `register_data_channel_message_callback` is in `libs/Python/pybind/visualization/webrtc_server/webrtc_window_system.cpp`. Distinguish directives from literal blocks and fence both `.. note::` and `.. note:: inline body` with their continuation indent **before** dedent/line-wise reflow; preserve adjacent `.. code-block::`/JS content and surrounding paragraph boundaries. Trimming trailing `::` on `.. ` directives turned markup into plain text; testing bare directives only missed the inline form. Add a regression with the inline directive plus indented subsequent body that runs the real handler order. A GUI import check or successful Doxygen run cannot confirm this is fixed. Re-run strict complete Sphinx on verified unchanged outputs; when source/static assets change rebuild HTML before staging.

### Transform lookup

Historical baseline 197 diagnostics over 2830 docstrings: 65 unexpected indents, 36 block quote ends, 14 definition list ends → `_separate_indent_changes`; 60 substitution starts → `_promote_pipe_art_blocks`; 8 strong and 2 emphasis starts → `_escape_bare_stars`; 1 inline literal → `_escape_unbalanced_backticks`; 4 class role, 2 eq role, 2 todo → resolve in Sphinx. The indent transform removed 101 reported diagnostics in one measured pass. The following are targeted rules, not suppression:

- Pybind aligns wrapped code under brackets: add separation at consecutive prose indentation changes without touching fenced directives or code. For directory art (`|`, `'--`, backtick followed by `--`), emit a real `::` literal block; detect by marker because dedent flattens it. Always advance scan index even on flush-left pipe lines (`start == end == i` used to hang the checker).
- Fence `.. math::` before reflow, escaping star, trimming `::`, or changing whitespace. Flush-left multiline equations can contain blank lines and lines without backslashes: retain through `\\end{...}` for those environments, then collect colon options even after blank lines. Replace the block with a sentinel, reflow the rest, restore the fenced block intact except deliberate normalization. Indent flush-left body and lift `:label:`/`:nowrap:` directly below the directive, with a blank line before nonempty body; an empty body fails `Content block expected`. Late labels give `invalid option block` then `equation not found`. After restoration, namespace labels and matching `:eq:` references by object name, e.g. distinct `RobustKernel` docstrings both used `robust_loss`.
- Separate flush-left field lists from prose in pass 600 because Napoleon creates them. Inline literals need ordered delimiter runs of exactly two backticks; even total count is insufficient (a four-backtick run is not two delimiters).
- Duplicate objects from inherited nested classes (e.g. `ccObject.Type` per subclass) are generator errors, not text. In `_generate_class_doc` compute `:exclude-members:` for nested classes only inherited via MRO, retain real declarations.
- Docutils `WarningStream.write` logs untyped warnings, so `suppress_warnings=['docutils']` cannot match them. `autodoc_warningiserror` only affects importing; a later logging filter runs after `WarningIsErrorFilter`. Fix the transform rather than hide diagnostics. Typed warning suppression is separate: `py:duplicate` is not `ref.duplicate`.
