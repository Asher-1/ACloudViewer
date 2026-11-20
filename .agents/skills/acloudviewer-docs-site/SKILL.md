---
name: acloudviewer-docs-site
description: Build, size, version and publish the ACloudViewer website plus all documentation versions. Use when changing docs build/deploy scripts, adding or fixing docs pages or notebooks, cutting a doc version, debugging the live site, or reducing artifact size.
---

# ACloudViewer Docs Site

Everything here was learned by breaking this pipeline and measuring the result.
Where a number appears it was measured; where a rule appears it came from an
incident.

---

## 1. Layout contract

Two products ship from one Pages artifact:

| Path | Content | Owner |
|---|---|---|
| `/` | marketing website (project info, downloads, quick links) | `docs/` minus build output |
| `/documentation/` | redirect stub → newest `vX.Y.Z` | `deploy_unified_docs.sh` |
| `/documentation/dev/` | main-branch docs | CI on `main` |
| `/documentation/vX.Y.Z/` | immutable per-release docs | release assets |
| `/documentation/versions.json` | version manifest | generated per deploy |
| `/404.html` | legacy deep-link rewriter | calibrated per deploy |
| `/sitemap.xml` | regenerated per deploy | generated per deploy |

**The manifest is derived from directories on disk, not from a release list.**
Deliberate: it is the only answer that cannot disagree with what is served. The
cost is that a missing directory removes the version from the site *and* from
every probe that iterates the manifest — see §7.

---

## 2. Build chain

```
workflow documentation.yml
  └─ docs-guards       seconds; blocks the expensive job
  └─ build-docs
       └─ docker/Dockerfile.docs   (ubuntu 22.04, sets CI=true, NOT GITHUB_ACTIONS)
            └─ util/ci_utils.sh::build_docs
                 build 1  headless  → executes notebooks, writes source/tutorial/**
                 build 2  GUI       → renders everything into docs/_out/html
            └─ docs/make_docs.py --sphinx --doxygen --strict
       └─ archive acloudviewer-<tag|sha>-docs.tar.gz
  └─ deploy-docs        main or release only
       └─ prepare_website.sh + restore archives + deploy_unified_docs.sh
```

### Why two builds

`ENABLE_HEADLESS_RENDERING` and `BUILD_GUI` are mutually exclusive in
`CMakeLists.txt`. Notebooks need an offscreen context; autodoc for
`visualization.{gui,rendering}` needs the real GUI module.

**Measured cost:** build 1 ≈ 1985 s of C++ compile plus ≈ 772 s notebook
execution. Build 2 recompiles the same tree with four flags flipped. Known and
accepted — do not rework it without measuring a replacement.

### builder order

```
process_doxyfile_template → Doxygen → cpp_api/api/
process_rst_templates     → source/**.rst from *.in.rst
PyAPIDocsBuilder          → source/python_api/   (needs importable cloudViewer)
PyExampleDocsBuilder      → source/python_example/
JupyterDocsBuilder        → source/tutorial/
SphinxDocsBuilder         → _out/html/
copy_plugin_showcase_assets
```

Doxygen writes into `_out/html/cpp_api/api/`, which must not collide with the
Sphinx-generated `cpp_api/{overview,plugins,quickstart}.html`.

### Incremental builds

`main()` deletes the output directory whenever `--sphinx` or `--doxygen` is
passed — not only with `--clean`:

```python
if args.clean or (args.sphinx or args.doxygen):
    _create_or_clear_dir(html_output_dir)
```

Sphinx's doctree cache defaults to `<build>/.doctrees`, inside that directory.
Pass an explicit `-d` pointing at a **sibling**; putting it inside is deleted
with everything else. Measured: inside → `[new config] 155 added` on the second
run; sibling → `loading pickled environment... done`. Cold 27.99 s → warm
23.76 s on 32 cores.

---

## 3. Local verification — always first

CI costs 40–70 minutes per round. The old excuse "cloudViewer only exists in the
CI image" is **false**: this repository carries a build tree.

```bash
# one-time
python3 -m pip install plotly dash ipykernel
python3 -m ipykernel install --user --name python3

# every time before pushing
bash docs/automation/scripts/local_docs_check.sh            # all 28 notebooks
bash docs/automation/scripts/local_docs_check.sh <name>     # one
```

`local_docs_check.sh` finds `build_app/lib/python_package`, proves it imports,
executes notebooks with nbconvert exactly as `make_docs.py` does, counts error
outputs, and skips the six notebooks CI does not execute.

**`python3 -I` ignores `PYTHONPATH`.** Local docs commands must run without `-I`,
otherwise `cloudViewer` "cannot be imported" and the python_api guard fails for
the wrong reason.

### Full local build

```bash
cd docs
export PATH="$HOME/.local/lib/python3.11/site-packages/pypandoc/files:$PATH"  # pandoc 3.9
export PYTHONPATH="<repo>/build_app/lib/python_package"
export CI=true
python3 make_docs.py --sphinx --strict --doxygen \
    --execute-notebooks=never --py-api-rst=always --py-example-rst=always \
    --output-dir /tmp/fullbuild
```

pandoc must be **3.9**. `pip install pypandoc_binary` puts it under
`site-packages/pypandoc/files/`; the distro package is 2.9.2.1 and emits
different heading adornments, so its diagnostics disagree with CI.

### Guard ladder

| Command | Proves |
|---|---|
| `test_deploy_unified_docs.sh` | deploy contract, 37 assertions |
| `test_prepare_website.sh` | website contract, 18 assertions |
| `check_notebook_headings.py` | no illegal RST transitions, 34 notebooks |
| `check_workflow_limits.py` | no `run:` scalar near the 21000-char cliff |
| `verify_docs_site.py` | browser behaviour: switcher, deep links, 404 |
| `local_docs_check.sh` | all notebooks execute cleanly |
| `check_docstring_rst.py` | every pybind docstring parses (§9) |

CI is the final gate. Confirm the failing **step** changed, not just that the run
is red — a different failing step means progress.

---

## 4. Size budget

Two independent limits, easy to confuse:

| Limit | Applies to | Enforced by |
|---|---|---|
| 1 GB | **compressed artifact** | `upload-pages-artifact` warns; `deploy-pages` fails *after* the build |
| 1 GB | **published site** (uncompressed) | Pages, with a generic message |

`check_pages_size_budget` measures `tar -czf - . | wc -c`. That is gzip; Pages
uses zstd. gzip is the conservative upper bound, so the check over-reports
rather than under-reports — safe direction, do not "correct" it.

### Where the bytes go

| Change | Saving | Loss |
|---|---|---|
| `DOT_IMAGE_FORMAT=svg` | 1947→921 MB raw, 1177→145 MB archived | none; also fixes PNG's 32767 px truncation |
| `INCLUDED_BY_GRAPH=NO` | ~197 MB | near-zero |
| exclude `qSIBR/SIBR` | ~72 MB | none, INRIA code |
| `SOURCE_BROWSER=NO` | ~361 MB | keeps 2303 header pages via `VERBATIM_HEADERS` |
| `HIDE_UNDOC_MEMBERS=YES` | 15 MB | not worth it |

**SVG wins because it is text.** Thousands of graphs share one compression
dictionary; PNG pixels have no cross-file redundancy.

Per-version tree after all of the above: ~769 MB on disk / ~587 MB content /
~145 MB archived. The v3.9.4-era archive is 243 MB because it predates the
Doxyfile change. Four archives plus dev approaches the 1 GB limit; the answer
then is to re-archive old versions, not to drop them.

---

## 5. Versioning

| Archive | Stored where | Name |
|---|---|---|
| release docs | that release's assets | `acloudviewer-<tag>-docs.tar.gz` |
| dev docs | the `main-devel` release | `acloudviewer-<sha>-docs.tar.gz` |

Disjoint patterns, so neither recovery path can pick up the other's files.
`main-devel` keeps only the newest dev archive; older ones are deleted before
upload. Pages replaces the whole artifact per deploy, which is why dev is carried
forward from `main-devel` on release builds.

**`versions.json` shape** (exactly this; `docs/script.js` depends on it):

```json
{ "latest": "v3.9.4", "generated_at": "…",
  "versions": [ {"id":"dev","label":"Development (main)","kind":"dev"},
                {"id":"v3.9.4","label":"v3.9.4","kind":"latest"} ] }
```

`dev` first, then releases descending, `kind` ∈ `dev|latest|archive`.

### Directory matching is strict

`list_version_dirs` accepts only `^v[0-9]+\.[0-9]+\.[0-9]+$`. A tag like
`v3.9.5-rc1` is silently excluded from `versions.json`, the sitemap, and archive
restoration — a prerelease deploy writes the directory and the next deploy drops
it. If prereleases must publish, widen the regex in one place only.

### Version resolution

`@VERSION@` and `@LATEST_STABLE_VERSION@` come from
`prepare_website.sh::resolve_website_version`: release tag → newest
**published** release → `version.txt` → `dev`.

`version.txt` is last on purpose: it is bumped *before* a release is cut, so
preferring it advertises an undeployed version. Pinned by contract test Case 2.

`@DOC_LATEST@` is different: `deploy_unified_docs.sh` expands it, because only
that script knows what was actually deployed.

### Historical versions are not recoverable

Only releases whose assets contain a docs archive can be restored. Measured: of
18 releases, 2 had archives (`main-devel`, `v3.9.4`); `/documentation/v3.9.3/`
is a 404 and cannot be backfilled without rebuilding docs from that tag.
Guarantee archives from the next release onward.

---

## 6. Deployment mechanics

Inputs: `IS_RELEASE`, `RELEASE_VERSION`, `DOC_VERSION`, `SITE_URL`. CWD must hold
`website-deploy/`, `docs-output/`, `preserved-versions/`; it writes
`unified-deploy/`.

Rules that exist because each was violated:

- **Refuse `IS_RELEASE=true` with empty `DOC_VERSION`.** Otherwise the dev branch
  runs: release docs overwrite `/documentation/dev/`, no archive is created,
  nothing fails.
- **Refuse `docs-output/` without `index.html`.** Doxygen output survives a
  failed Sphinx run, so the directory can hold ~96k files and no entry page.
- **Extract into staging and require `index.html` before publishing.** Creating
  `vX.Y.Z/` first leaves an empty version directory on a corrupt archive, and
  `versions.json` then advertises pages that 404.
- **A release that has an archive but cannot restore it fails the job.** One that
  never shipped docs only warns. Continuing would delete a published version.
- **`cp -r src/. dst/` never deletes.** Re-deploying merges: deleted pages
  survive. `release: edited` re-runs everything, so this is reachable. Remove the
  target directory first.
- **Only probe `/documentation/dev/` when the artifact contains it.** The restore
  is best-effort by design.
- **Upload the resolved archive; do not re-derive its name.** Extraction has
  already renamed it to `acloudviewer-<tag>-docs.tar.gz`, so a `cp` onto that
  name fails (`GNU cp: are the same file`, exit 1) and killed every release
  build.

Archive storage details: see §5. Two size limits: see §4.

### Branch runs never deploy

`should_deploy` is true only for `refs/heads/main` or a `release` event. A
`workflow_dispatch` on a feature branch builds and uploads an artifact that goes
nowhere. **A green branch run says nothing about the deployment path**, and the
deploy scripts can only be fully exercised on `main` or a release.

---

## 7. Silent-failure catalogue

Every entry produced plausible output and a zero exit status.

| Symptom | Mechanism | Guard now |
|---|---|---|
| `python_api/` missing | per-module import errors printed only; `conf.py` suppresses `toc.not_readable`/`ref.doc` | zero generated pages → raise |
| `cpp_api/api/` missing | `DoxygenDocsBuilder.run` prints and returns | — (untested) |
| two Getting Started pages 404 | `process_rst_templates` swallowed per-file errors; `index.rst` still lists them | any template failure → raise |
| docs built from the wrong notebook | a download overwrote the staged copy with upstream Open3D's file (1670 lines, `####`) | download removed; staged-set check |
| notebook failure ignored | guard required `GITHUB_ACTIONS`; Docker sets only `CI` | either env → raise |
| historical version vanished | archive restore warned and continued | staging + `index.html` requirement |
| downloads page empty | `scan_releases.py` emits `download_data`; a workflow fallback wrote `download_links` | fallbacks emit the real shape |
| `update-downloads` cleanup died | `to_entries` applied to `version_metadata`, an **array** → `Cannot use number (0) as object key` | `map()` over the array |
| 13 versions missing from the page | `fetch_releases(limit=5)` against 18 releases | `DEFAULT_RELEASE_LIMIT = 100` |
| everything "green" with 0 notebooks run | no guard on execution count; cloudViewer was never built | `local_docs_check.sh` + count check |
| whole workflow silently invalid | `run:` scalar over 21000 chars → zero jobs, no logs | `check_workflow_limits.py` |

### Never fetch documentation content from another repository

A download used to install `t_icp_registration.ipynb` from
`github.com/isl-org/Open3D/files/8243984`, overwriting the copy staged from
`docs/jupyter/`. Every notebook lives there and is git-tracked; if one is
missing, fail. A fallback that substitutes another project's file is worse than
a missing file, because it produces plausible wrong output.

`docker/Dockerfile.docs` still clones CloudViewer-ML on purpose — a *dependency*
checkout (31 `cloudViewer.ml.*` modules in `documented_modules.txt`, plus
`docs/tensorboard.md`). Do not remove it while removing content downloads.

### Swallowing an exit status is the standard way this breaks

`build_docs()` runs under `set +u`, not `set -e`. Wrap every command:

```bash
if ! python make_docs.py ...; then
    set +x
    echo "❌ ... failed" >&2
    return 1
fi
```

---

## 8. Notebook rules

nbsphinx converts each Markdown cell with pandoc and concatenates the result.
Three traps:

**A standalone `---` becomes an RST transition.** docutils rejects a document or
section beginning with one (`ERROR/3`), and `-W` makes it fatal. Editing blank
lines only moves the failure between "overline title" and "illegal transition".
**Delete the rule.** Verify through the real pipeline:

```python
import nbsphinx
rst = '\n'.join(nbsphinx.markdown2rst(''.join(c['source']))
                for c in nb['cells'] if c.get('cell_type') == 'markdown')
# then publish_doctree(rst), count system_message with level >= 2
```

**Every notebook needs a toctree entry.** `-W` rejects `document isn't included
in any toctree`, and the page is invisible until something forces rendering.

**Headless rendering needs a real GL context, not a guard.** This cost two CI
rounds and a wrong fix. The mechanism:

- The docs image has no display. GLFW then falls back to
  `glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL)` (OSMesa software).
- That platform is handed no GL version — the log says
  `GLEW initialization returned GLEW_ERROR_NO_GL_VERSION in headless mode,
  continuing anyway`, and the code continues anyway.
- Rendering then runs against an invalid context. Locally this reproduces as a
  **segmentation fault** with `DISPLAY` unset; in CI it surfaces through pybind
  as `IndexError: vector::_M_range_check: __n (which is N) >= this->size()
  (which is N)`, where N is the mesh vertex count.
- The failure moves as you patch it: guarding `poll_events` lets execution
  reach the next cell (`remove low density vertices`) and fail there. Guarding
  cell by cell is the wrong layer.

**Fix: give it a display.** `docker/Dockerfile.docs` installs `xvfb` +
`libgl1-mesa-dri` + `libglx-mesa0`, and the docs build runs under
`xvfb-run -a --server-args="-screen 0 1920x1080x24" bash -c
"source util/ci_utils.sh && build_docs …"`. The child shell sources
`ci_utils.sh` itself so `build_docs` keeps its helpers; `export -f build_docs`
alone does not carry them.

**Proof, not inference.** Same notebook, no `DISPLAY`: without Xvfb it segfaults
(exit 139); under `xvfb-run` it reports `ok`. Reproduce with
`env -u DISPLAY -u WAYLAND_DISPLAY bash docs/automation/scripts/local_docs_check.sh <nb>`.

Open3D solves the same problem with an EGL offscreen context
(`EGLOffscreenContext`). This repository has never had EGL, so Xvfb is the
matching fix rather than a port.

Do **not** add a `try/except` in `cloudViewer_tutorial.py` to swallow this. It
hides a broken context and only postpones the crash to the next cell.

**Notebooks only execute when cloudViewer was actually built.** A run reporting
`Successfully executed: 0 notebook(s)` is not a pass — nothing ran.

---

## 9. pybind docstring warnings

`-W` fails on malformed RST that pybind11 generates from C++ signatures. There is
no source file to fix.

**Do not suppress these.** Fix the markup in `process_docstring`. The text is
correct; only its incidental markup is wrong, and a suppressed warning still
renders a mangled page.

### Collect them all first

The real build stops at the first failure, costing a full round per warning. Use
the checker, which walks every reachable docstring and prints the full set:

```bash
PYTHONPATH=<repo>/build_app/lib/python_package \
  python3 docs/automation/scripts/check_docstring_rst.py
```

It must reproduce autodoc's pipeline — **in order** — or it reports phantom
errors:

1. `prepare_docstring()` — removes the common indent (ignoring line 1)
2. `conf.py::process_docstring` (the transform, event priority **400**)
3. **Napoleon** — converts Google/NumPy sections (`Args:`) to RST (default
   priority **500**)
4. docutils parses

The order of 2 and 3 is not a detail. Running the transform *after* Napoleon
reports phantom errors in both directions:

- Skipping Napoleon produced 106 phantom `Unexpected indentation` errors,
  because `Args:` followed by deeper indentation reads as a broken definition
  list to plain docutils.
- The checker originally ran the transform **after** Napoleon and therefore
  never exercised the state the real build sees. It reported 0 problems while
  the build failed; the two diverged at the math blocks, which Napoleon
  re-indents. Call the transform on `prepare_docstring()` output *before*
  wrapping in `GoogleDocstring`/`NumpyDocstring`.

### The checker must cover what the build covers

Two independent bugs made the checker report **0 problems while the real build
failed**:

- The root list was hardcoded to five modules. `cloudViewer.reconstruction`
  was not in it, so `SequentialMatchingOptions.vocab_tree_path` was never
  checked. Read `docs/documented_modules.txt` — the same file
  `PyAPIDocsBuilder` uses — instead of keeping a second copy of the list.
- Walking those modules pulls in **re-exports**: `cloudViewer.visualization`
  re-exports torch, tensorboard, matplotlib and numpy. Their docstrings are
  Sphinx-flavoured and produce ~9,000 "Unknown role/directive" errors that the
  real build resolves through its role registry. Filter by `__module__`: only
  objects whose `__module__` starts with `cloudViewer` are ours to check.
  pybind members report `cloudViewer.cuda.pybind.<...>`, so the prefix test
  catches them; `torch.zeros_like` reports `torch` and is skipped.

The correct scope is ~2,300 docstrings, not 2,800 (five roots) and not 14,900
(unfiltered).

### Measured distribution and the fix for each

Starting point: **197 diagnostics / 2,830 docstrings**. Every family below has a
proven, non-suppressing fix applied in `process_docstring`.

| Count | Message | Fix |
|---|---|---|
| 65 | Unexpected indentation | `_separate_indent_changes` |
| 60 | Inline substitution_reference start-string | `_promote_pipe_art_blocks` |
| 36 | Block quote ends without a blank line | `_separate_indent_changes` |
| 14 | Definition list ends without a blank line | `_separate_indent_changes` |
| 8 | Inline strong start-string | `_escape_bare_stars` |
| 2 | Inline emphasis start-string | `_escape_bare_stars` |
| 1 | Inline literal start-string | `_escape_unbalanced_backticks` |
| 4 | Unknown role "class" | none — resolved by Sphinx |
| 2 | Unknown role "eq" | none — resolved by Sphinx |
| 2 | Unknown directive "todo" | none — `sphinx.ext.todo` |

The last three are checker artifacts. Prove it before chasing them: `:eq:`,
`:file:` and the `todo` directive all report `CLEAN` when the same text is
parsed through Sphinx's real environment. Do not "fix" them.

### The structural transforms

**`_separate_indent_changes`** — pybind interleaves a paragraph with
continuation-aligned code:

```
mesh.vertex.positions = cv3d.core.Tensor([[0, 0, 0],
                                         [0, 0, 1]], dtype_f, device)
```

The second line is indented only to align under the bracket. docutils has no
paragraph open there, so it reports "Unexpected indentation" and then a
"Block quote ends" when the indent drops back. Inserting a blank line wherever
consecutive text lines change indent makes each run its own block. **101 of the
197 disappear from this one transform.**

**`_promote_pipe_art_blocks`** — embedded directory trees:

```
    RedwoodIndoorOffice2
    |-- colors
    |   |-- 00000.jpg
```

docutils reads `|--` as a substitution-reference opener. Emit `::` above the run
and indent it: the art becomes a real literal block and renders verbatim.

Two traps in implementing it, both real bugs on the first attempt:

- Detect on the **tree markers** (`|`, `'--`, `` `-- ``), not on indentation.
  This runs after `_dedent_docstring`, which has already flattened the block to
  column 0; an indentation test skips every case.
- Advance the loop index by at least one. A flush-left pipe line makes
  `start == end == i`, and `i = end` spins forever — it hung the checker, not
  the build.

### `.. math::` blocks must be fenced before any line-wise fix

The blanket per-line transforms (star escaping, trailing-`::` trimming,
`_separate_indent_changes`) would corrupt LaTeX. Protect math first:

1. Scan for a line whose stripped form starts with `.. math::`.
2. Find the block end. The reliable terminator is **`\end{...}`**, not blank
   lines: pybind emits flush-left math bodies, and `x^{*}` lines carry no
   backslash at all, so a "stops at the first non-indented line" rule cuts the
   block in half. After seeing `\end{...}`, keep consuming `:`-prefixed option
   lines (they may be separated by blank lines).
3. Replace the block in place with a single sentinel line, run the normal
   `_reflow` over the rest, then splice the original block back.
4. **Indent the body** if its first content line is flush-left, and **lift
   option lines** (`:label:`, `:nowrap:`) directly under the directive with a
   blank line before the body. A `:label:` that stayed *below* the LaTeX cost
   two separate failures: `invalid option block` (options must precede the
   body) and then `equation not found` (the label never registered).

Steps 3 and 4 happen on the extracted block, not on the live list — the
sentinel must round-trip to a byte-identical directive.

### Equation labels collide across docstrings

Two distinct `RobustKernel` classes in different modules embed the same literal
`:label: robust_loss`, so the second registration fails `-W` with "duplicate
label". docutils labels are global to the document set. Namespace them per
docstring — `_namespace_equation_labels` rewrites both the `:label:` value and
every `:eq:` reference in the same docstring with a `name`-derived suffix. Do
this **after** the blocks are restored; renaming inside the fenced copies is
fine, renaming the sentinel is not.

### Empty-body directives

`.. math::` with no body after option relocation produces "Content block
expected". Check the restored block still has a non-empty, non-option body
before emitting it.

### Field lists need a blank line above them

`-W` reports "Field list ends without a blank line" for a flush-left `:param x:`
that follows text directly. This one cannot be fixed in the first pass —
Napoleon *creates* the field list, and it runs at 500, after the transform at
400. Connect a **second** handler at priority **600** (after Napoleon) that
only inserts the separating blank line. It must be an adapter with the full
`(app, what, name, obj, options, lines)` signature; connecting a one-argument
helper directly throws `takes 1 positional argument but 6 were given` and
aborts the whole build.

### Two-phase ordering summary

| Priority | Handler | Sees |
|---|---|---|
| 400 | `process_docstring` | raw pybind text; fencings, indent fixes, star escapes |
| 500 | napoleon (default) | normalised text; converts `Args:` to field lists |
| 600 | `_second_pass_docstring` | post-napoleon RST; blank line before field lists |

Anything that depends on Napoleon's output belongs in the 600 pass; anything
that Napoleon must see un-mangled belongs in the 400 pass.

### Why `suppress_warnings` cannot help

Sphinx forwards docutils messages via
`WarningStream.write` → `logger.log('WARNING', text)` **with no `type` attribute**.
`is_suppressed_warning(type, subtype, ...)` matches on that type, so no key in
`suppress_warnings` can ever match these. Verified: adding `'docutils'` changed
nothing.

`autodoc_warningiserror` does not help either — it is threaded into
`import_object()` only, not into docstring parsing.

A `logging.Filter` on `location` (`"docstring of" in location`) also does not
work as written: `WarningIsErrorFilter` is inserted before any filter added
later, and it raises on the record first. Do not reach for either.

- **`conf.py::process_docstring`** — the correct and only layer. Runs at
  `priority=400`, before napoleon's default 500, so napoleon sees normalised
  input and converts `Args:` properly.
- **`suppress_warnings`** — only for warnings that carry a type (e.g.
  `py:duplicate`, `ref.*`, `myst.*`). Note `py:duplicate` (Python domain) is a
  different key from `ref.duplicate`; using the wrong one suppressed nothing for
  65 warnings.

### Duplicate object descriptions are a separate problem

`:inherited-members:` documents a base class's **nested classes** again on every
subclass page. `ccObject.Type` is reached from every geometry type, so it
registered once per subclass and failed `-W` with "duplicate object
description". This is not a docstring issue and `process_docstring` cannot touch
it — the fix is in the generator (`_generate_class_doc` in `make_docs.py`):
emit `:exclude-members:` for nested classes the class only inherits, computed
from the MRO so it stays correct when the binding adds another enum.

### Directive vs literal-block rule

`.. note::` and `text::` both end in `::`, but only the latter opens a literal
block. Trimming every trailing `::` turned `.. note::` into plain text and broke
the indentation fix. Guard with `startswith(".. ")`.

### Backtick balance is not a parity check

An even count does not mean balanced: ```` `` `` ```` is four backticks, which
docutils reads as one opener with no closer (`str: (Default ````)` occurs in
`SequentialMatchingOptions.vocab_tree_path`). Scan runs in order and treat only
a run of exactly two as an inline-literal delimiter.

---

## 10. Workflow-level traps

- A `run:` scalar over **21000 chars** silently invalidates the whole workflow —
  zero jobs, no logs, only an annotation in the run's HTML page.
- **`apt` retry count defaults to 0.** The Qt layer once fetched 116 MB in 65
  minutes and died on one dropped connection. The Dockerfile sets
  `Acquire::Retries`/timeout, retries three times with `--fix-missing`, then
  asserts `dpkg -s qtbase5-dev` — a loop `exit 0` only proves apt returned
  success.
- **A `gh` log blob does not exist until the job completes.** `BlobNotFound`
  means "still running", not "no log".
- **A Docker job log interleaves multi-process stderr one character per line.**
  Reconstruct by stripping the `#<n> <time> ` prefix and rejoining.

## Diagnosing a red run

### A red run is not always your change

Check the failing Docker **layer** before reading it as a code defect. The Qt
install layer has failed on its own:

```
Err:199 ...qtattributionsscanner-qt5... Connection timed out
Fetched 116 MB in 1h 5min 43s (29.5 kB/s)
```

`apt`'s retry count is 0 by default, so one dropped connection aborts a job that
already spent an hour. Fixed in the Dockerfile (§10).

### Pre-push syntax gates

1. `python3 -m py_compile docs/make_docs.py`, `bash -n` on every touched script,
   YAML-parse the workflow.
2. Then the guard ladder (§3).
3. Local Sphinx build must reach `build succeeded` before pushing.

```bash
JOB=$(gh api repos/Asher-1/ACloudViewer/actions/runs/<run>/jobs \
      --jq '.jobs[]|select(.name=="build-docs")|.id')
gh api "repos/Asher-1/ACloudViewer/actions/jobs/$JOB/logs" > /tmp/build.log
```

Read the **first** error, not the last: `sphinx-build -W` reports one warning and
exits; later lines are the Python traceback unwinding through
`subprocess.check_call`.

Check `head_sha` before anything else — verifying the wrong commit looks exactly
like a fix that did not work.

---

## 11. Commit hygiene

This work sits beside an unrelated in-progress workstream. `git add <files>` does
**not** produce a focused commit when the index already holds other staged paths:
two docs commits shipped with 126 icon SVGs and 39 plugin images riding along.

```bash
git reset                       # clear the index, keep the worktree
git add <exactly the files>
git diff --cached --name-only   # read this before committing
```

---

## 12. Anti-patterns seen in this repository

- **Fixing a warning without reproducing the pipeline.** Two commits adjusted
  blank lines around a `---`. Neither could work: the file was being overwritten
  by a download, and the construct itself was invalid.
- **Editing a file the build does not read.** The cheap tell: the CI diagnostic
  quoted a `^^^^` underline, which pandoc only emits for level-4 headings, while
  the checked-in notebook uses `###`.
- **Treating a red run as one bug.** Three defects were live at once (archive
  re-copy, dev probe, notebook overwrite). Confirm the failing step moved after
  each fix.
- **Assuming CI covers deployment.** Branch runs skip `deploy-docs`.
- **Adding `|| true` / `|| echo` to get green.** That is how "success" stopped
  meaning anything.
- **Widening the blast radius.** "Stop downloading one notebook" does not mean
  "remove the CloudViewer-ML checkout": it supplies dependencies for 31 ML
  modules.
- **Fixing generated markup one warning at a time.** Collect the whole set with
  the checker (§9), then fix by class.
- **Reading a `⚠️` in a log as success.** Several guards printed a warning and
  continued; each of those was a real defect.

---

## 13. When something is wrong

1. **Read the failing step name**, not just the run status. `Build documentation
   Docker image` and `Verify live deployment` are entirely different problems.
2. **Check `head_sha`.**
3. **Classify: code or infrastructure.** A Qt `Connection timed out` or a runner
   OOM is not your diff.
4. **Reproduce locally** (§3). If it cannot be reproduced, say so and name what
   blocks it — do not push to find out.
5. **Only then** push a single focused commit.
