# Native Documentation CI replay (Ubuntu)

Run from the current checkout, not a detached copy unless requested. The current [workflow](<../../../.github/workflows/documentation.yml>), Docker build recipe and `util/ci_utils.sh` are authoritative for flags, native packages, Qt, PyTorch and ML revision. Ubuntu need not match the image release for a requested native replay; report the difference. CI's image was Ubuntu 22.04, set `CI=true` (not necessarily `GITHUB_ACTIONS`), and used two builds because `ENABLE_HEADLESS_RENDERING` and `BUILD_GUI` are mutually exclusive. Headless uses OSMesa, not a display. Historically the first compile took ~1985 s, notebooks another ~772 s, followed by a second configure/build. These are not timeouts or estimates for every machine.

## Commands and phase evidence

Resolve suitable host compiler, Qt, OSMesa, Doxygen, and external CloudViewer-ML first. Use Python 3.11 if it matches the current workflow, select its real interpreter (`pyenv which python` when managed by pyenv), and use the current workflow's PyTorch source and version (the observed native run used CPU Torch `2.7.1+cpu`). Do not guess wheel URLs or host package versions. **TODO when replaying:** record the exact current native dependency/bootstrap commands and resolved paths in the run report; the historical wrapper is not a versioned prerequisite.

```bash
# Set PYTHON_EXE to the verified base interpreter; use a dedicated docs venv.
repo=$(pwd)
venv="$HOME/.cache/acloudviewer-docs-ci-venv"
"$PYTHON_EXE" -m venv "$venv"
export PATH="$venv/bin:$PATH"
python -m pip install -r docs/requirements.txt
# Install the pinned torch and native-equivalent requirements from the current CI recipe.
python -m pip check
python -m pip install pypandoc_binary
pandoc_bin=$(python -c 'import pypandoc; print(pypandoc.get_pandoc_path())')
export PATH="$(dirname "$pandoc_bin"):$PATH"
pandoc --version
python -m pip check
```

Validate Pandoc version before trusting heading diagnostics: observed CI parity was 3.9; distro 2.9.2.1 produced different heading adornments. Record `python -V`, `sys.executable`, Torch version, `pip check`, and kernel selection. If a notebook uses `python3`, check the kernelspec resolves to this venv: with ipykernel installed per the recipe, `python -m ipykernel install --prefix "$venv" --name python3` registers it without replacing a user's global spec; verify the effective spec used by nbconvert. Do not install into global site-packages. Missing `pytest` in a docs venv means the optional pytest check did not run, not that the project test failed.

```bash
# From repo root, after inspecting capacity and setting DOC_JOBS to a safe cap.
export CI=true NPROC="$DOC_JOBS" CMAKE_BUILD_PARALLEL_LEVEL=1
# For probes/replays use actual build path and Qt lib directory verified on this host.
export PYTHONPATH="$repo/build/lib/python_package"
export LD_LIBRARY_PATH="$repo/build/bin:$QT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CLOUDVIEWER_ML_ROOT="$ML_ROOT"  # dependency root containing ml3d/; record revision
# Record any thread caps you use; for import controls one thread each is enough.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
(
  set -eo pipefail
  source util/ci_utils.sh
  build_docs ON
) > "$EVIDENCE/native-build.log" 2>&1
```

Use a managed background job for the long command; set an existing writable `EVIDENCE` directory outside generated trees (the previous run used `build_headless_docs/evidence/`). Record job ID, start/finish, phase exit codes and output. Inspect the current helper for any environment it sets itself, where the tree lives, and where to snapshot each stage. Verify `NPROC` actually bounds Sphinx; the old generic `--parallel` expanded to `-j32` on a 32-core host. Run `python docs/automation/scripts/test_docs_parallelism.py` when the current checkout contains it. The observed guard ran seven tests. Disable neither parallel Sphinx nor warnings-as-errors. `build_docs` can run with `set +u`, and the helper cannot rely on caller `set -e`: explicitly return nonzero when commands fail (`if ! python make_docs.py ...; then set +x; echo 'docs build failed' >&2; return 1; fi`).

```bash
rg -n '^(CMAKE_BUILD_TYPE|BUILD_DOCUMENTATION|BUILD_GUI|ENABLE_HEADLESS_RENDERING|BUILD_WEBRTC|BUNDLE_CLOUDVIEWER_ML|BUILD_PYTORCH_OPS|BUILD_CUDA_MODULE):' build/CMakeCache.txt
python -c 'import cloudViewer as c, torch; print(c.__file__); print(c._build_config); print(torch.__version__)'
```

Check the package comes from the current build, not a wheel or `build_app`. Capture both caches/configs before the second configure overwrites the first. Stage 1 expected Release with headless ON / GUI OFF, docs and PyTorch ops enabled, notebook execution and strict Sphinx exit 0. Preserve the `Successfully executed` / `Failed to execute` counts (observed 28/0, current CI set authoritative). Stage 2 expected Release with headless OFF and GUI, WebRTC, ML bundle, docs and PyTorch ops ON. Probe `cloudViewer.visualization.gui`, `.rendering`, `.webrtc_server`, and `cloudViewer.ml.torch` under the same Python/library paths; verify generated pages for those APIs, not just imports. Check Doxygen log plus rendered `cpp_api/api/index.html` and Sphinx exit 0 plus final `index.html`, API/tutorial pages. `DoxygenDocsBuilder.run` has printed and returned on failure; the directory alone is not success. A successful first stage is not a completed second stage.

Builder order: Doxyfile template → Doxygen; RST templates → generated RST; API generator → `source/python_api`; examples → `source/python_example`; Jupyter staging/execution → `source/tutorial`; Sphinx → `_out/html`; then plugin showcase assets. Doxygen `cpp_api/api/` is distinct from Sphinx `cpp_api/{overview,plugins,quickstart}.html`. `make_docs.py` clears HTML on `--sphinx` or `--doxygen` even without `--clean`. Keep explicit `-d` outside HTML; default `.doctrees` inside output is deleted. Historical warm sibling-cache run loaded the environment (27.99 s cold → 23.76 s warm); for the ML regression always use a fresh doctree.

## Jobs, generated input and notebook evidence

```bash
ps -eo pid,ppid,comm,etime,%cpu,rss,state,wchan:32,args
free -h
df -h .
tail -n 80 "$EVIDENCE/native-build.log"
rg -n 'CMAKE_GENERATOR:|CMAKE_BUILD_TYPE:|CMAKE_CXX_COMPILER:|ENABLE_HEADLESS_RENDERING:|BUILD_GUI:' build/CMakeCache.txt
```

Use the real path for the task (including `build_headless_docs` for an independent repro). Correlate compiler/linker CPU, RSS, state, elapsed time, disk and log activity. A sleeping parent can be waiting for a busy compiler or link child. Read each nested ExternalProject cache first, then inspect its `build.ninja` or `Makefile` as appropriate and actual nested `cmake --build ... --parallel N`/Make/Ninja flags. Count aggregate active jobs and reserve memory and capacity for unrelated builds. A user ceiling is a hard upper bound, not an instruction to fill it. Example: with an independent `make -j6` and total budget 12, stopped only the owned job, verified its children exited, then resumed with `NPROC=6`, `CMAKE_BUILD_PARALLEL_LEVEL=1` and verified actual activity. No overlapping old/new build; no restart just for silence. Report health, change of cap, and completion separately.

Do not delete local source because a CUDA name appears in a CPU build. When the failing path names an ignored generated page such as `cloudViewer.ml.contrib.rst`/`iou_3d_cuda`, inspect both `python_api` and `python_example`: `git ls-files -- docs/source/python_api docs/source/python_example`, `git status --short --ignored -- docs/source/python_api docs/source/python_example`, and `git check-ignore -- docs/source/python_api/cloudViewer.ml.contrib.rst`. Check the generator in `docs/make_docs.py` and inspect contents (ignored is not proof of generated ownership). Only if **all** files in a tree are stale generated inputs, no tracked or handwritten contents, no active writer and the destination does not exist, move it intact to `"$EVIDENCE/${name}-before-native-ci"`, using `name=python_api` or `name=python_example`. For mixed trees, isolate only individually verified generated files and keep relative paths. Record and verify the moves; use the clean CI-equivalent headless inputs, then GUI generates its own pages. Do not edit generated CUDA references or mask missing-object warnings.

If the log is interleaved one character per line, parse the executed `.ipynb` copies while evidence is still present. This probe writes one record per code cell with its original source and raw errors, plus image and execution counts; run it against current replay outputs, not untouched or stale notebooks. Adjust the root to the selected replay output if not `docs/source/tutorial`.

```bash
python - docs/source/tutorial > "$EVIDENCE/notebook-cells.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.rglob('*.ipynb')):
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = []
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        outputs = cell.get('outputs', [])
        cells.append(dict(cell_index=idx, execution_count=cell.get('execution_count'),
                          source=''.join(cell.get('source', '')),
                          errors=[{k: o.get(k) for k in ('ename', 'evalue', 'traceback')}
                                  for o in outputs if o.get('output_type') == 'error'],
                          images=sum(any(k.startswith('image/') for k in o.get('data', {}))
                                     for o in outputs)))
    rows.append(dict(notebook=str(path.relative_to(root)), cells=cells,
                     executed=sum(c['execution_count'] is not None for c in cells),
                     errors=sum(len(c['errors']) for c in cells),
                     images=sum(c['images'] for c in cells)))
json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
PY
```

Compare the staged file list and current exclusions in `make_docs.py` to the tracked `docs/jupyter/` set; historically 34 total, 6 excluded, 28 executed. Compare expected runnable cells to execution counts (empty/explicitly excluded cells aren't evidence of a failure); count per notebook, do not infer success from an empty JSON array or pre-existing outputs. Keep original cell exception/traces even if execution stops early. Never fetch replacement notebook content from another repository: a download of Open3D's `t_icp_registration.ipynb` previously overwrote the tracked input with a 1670-line level-4-heading variant. CloudViewer-ML is a dependency, not substitute docs: the checkout supplies 31 documented ML modules and TensorBoard support in the observed configuration.

Known fresh-venv issue: `ray_casting`, `rgbd_image`, and `distance_queries` failed while `surface_reconstruction` passed. Matplotlib 3.7.3 relies on IPython `backend2gui`, removed in 9.16; reproduce with the same venv and Agg/IPython before changing pins. When the target branch pins Matplotlib 3.7.3 and demonstrates this incompatibility, constrain `ipython<9.16` in docs requirements, then verify same probe and `python -m pip check` (observed good 9.15.0). Do not apply the cap blindly to later dependency sets.

```bash
MPLBACKEND=Agg python - <<'PY'
import IPython, matplotlib
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.instance().enable_matplotlib('agg')
from matplotlib import pyplot as plt
plt.imshow([[0, 1], [1, 0]])
plt.close()
print('MATPLOTLIB_IPYTHON_OK', matplotlib.__version__, IPython.__version__)
PY
```

Local shortcuts are diagnostic only: `bash docs/automation/scripts/local_docs_check.sh` (all) or append a notebook name; inspect its package discovery (`build_app/lib/python_package` historically) before using it for a specific rebuilt headless package. Never use `python -I` with `PYTHONPATH`. With an already verified GUI package and notebook outputs, an isolated API/Sphinx/Doxygen diagnostic can use `cd docs; python make_docs.py --sphinx --strict --doxygen --execute-notebooks=never --py-api-rst=always --py-example-rst=always --output-dir /tmp/fullbuild` from the configured venv. It does **not** run the full CI notebook stage. A standalone Markdown `---` becomes an illegal initial RST transition; delete the rule, don't shuffle blank lines. Check through `nbsphinx.markdown2rst` per Markdown cell then docutils (`system_message` level ≥2), and keep each notebook in a toctree. `check_notebook_headings.py` checks the source corpus.

## Headless Poisson and dataset diagnostics

Use an independent Release headless build, not an existing GUI package, with cache verified before Python. If a tutorial dataset search timed out, do not rerun a broad recursive scan. Start with `ls -lah "$HOME/cloudViewer_data/download" "$HOME/cloudViewer_data/extract" "$HOME/.cache/cloudViewer" 2>/dev/null | head -80`; check the current notebook's expected filename(s) or dataset member paths under those bounded roots (e.g. for EaglePointCloud). A generic cache listing is not a match; if missing report it as unconfirmed and do not edit tutorial inputs or start a token replay.

For Python/C++ capture, resolve `pyenv which python` rather than using its shim. Check the resolved executable (`file "$PYTHON_EXE"`) and recheck its package import in the headless environment. A shim 'not in executable format' is a GDB setup failure, not the business reproduction. Use the original notebook cells/inputs for a minimal replay script (or the actual kernel process); save the exact reproducer and raw logs. Example for a script created from the notebook:

```bash
gdb --args "$PYTHON_EXE" "$REPRO_SCRIPT"
# At GDB prompt (with symbols where possible):
# break std::__throw_out_of_range_fmt
# run
# bt full
# If the symbol isn't available, catch throw, continue to the matching exception,
# and inspect the throwing frame and callers with frame/info locals.
```

Match the specific pybind `IndexError` / `std::vector::at` exception to the throwing call chain, actual vector and access index/size before changing code; equality of the reported length with vertex count does not identify the vector. Archive failing stack, inputs and config; **TODO if symbols/inputs prevent capture:** report missing concrete evidence, do not fabricate a frame. Compare last green and first red same-image runs before blaming environment: the v3.9.4 `0ac4b09e` run had 28/0 under the same Dockerfile, and the mesh inputs, not renderer code, changed across the investigated transition. Adding Xvfb or a notebook guard did not fix the root cause.

After fixing indexes, rebuild and prove `cloudViewer.__file__` points at the new package. Check both colored and uncolored meshes, every triangle index nonnegative and strictly below vertex count (minimum need not be zero), continue through original low-density vertex deletion and ensuing normal operations. If screenshot works but deletion fails, get the **new** stack rather than declaring success. Review the current `python/test/test_surface_reconstruction*` files before selecting the exact pytest invocation; no guessed test filename. Re-run original full notebook plus the full CI notebook set under the original exclusions, verify executed cells/no errors and image count via JSON, archive before/after stacks, regression output and per-notebook/phase results. Zero notebooks executed is never a pass.
