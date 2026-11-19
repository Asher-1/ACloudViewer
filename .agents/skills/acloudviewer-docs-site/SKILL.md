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

**Headless screenshots can be degenerate.** `capture_screen_float_buffer()`
returns a scalar when the window never produced a frame; `np.asarray` gives a 0-d
array and `PIL.Image.fromarray` raises `IndexError: tuple index out of range`.
`surface_reconstruction.ipynb` triggers this with 565k points / 1.13M triangles.
`cloudViewer_tutorial.py` guards on `ndim < 3` and skips the inline image; all 28
notebooks share that helper.

**Notebooks only execute when cloudViewer was actually built.** A run reporting
`Successfully executed: 0 notebook(s)` is not a pass — nothing ran.

---

## 9. pybind docstring warnings

`-W` fails on malformed RST that pybind11 generates from C++ signatures. There is
no source file to fix.

### Collect them all first

The real build stops at the first failure, costing a full round per warning. Use
the checker, which walks every reachable docstring and prints the full set:

```bash
PYTHONPATH=<repo>/build_app/lib/python_package \
  python3 docs/automation/scripts/check_docstring_rst.py
```

It must reproduce autodoc's pipeline or it reports phantom errors:

1. `prepare_docstring()` — removes the common indent (ignoring line 1)
2. **Napoleon** — converts Google/NumPy sections (`Args:`) to RST
3. `autodoc-process-docstring` → `conf.py::process_docstring`
4. docutils parses

Skipping step 2 alone produced 106 phantom `Unexpected indentation` errors here,
because `Args:` followed by deeper indentation looks like a broken definition
list.

### Measured distribution (2831 docstrings)

| Count | Message |
|---|---|
| 81 | Unexpected indentation |
| 60 | Inline substitution_reference start-string without end-string |
| 24 | Block quote ends without a blank line |
| 13 | Definition list ends without a blank line |
| 8 | Inline strong start-string without end-string |
| 4 | Unknown interpreted text role "class" |
| 2 | Unknown interpreted text role "eq" |
| 2 | Inline emphasis start-string without end-string |

### Why `suppress_warnings` cannot help

Sphinx forwards docutils messages via
`WarningStream.write` → `logger.log('WARNING', text)` **with no `type` attribute**.
`is_suppressed_warning(type, subtype, ...)` matches on that type, so no key in
`suppress_warnings` can ever match these. Verified: adding `'docutils'` changed
nothing.

`autodoc_warningiserror` does not help either — it is threaded into
`import_object()` only, not into docstring parsing.

The working levers are therefore:

- **`conf.py::process_docstring`** — fix the text before docutils sees it. This
  is the correct layer: the content is right, only its markup is incidental.
  Handles: namespace rewriting, unbalanced `` ` ``/`*`, `text::` that promises a
  literal block that never arrives, and `.. note::` bodies that are not indented
  and blank-line separated.
- **`suppress_warnings`** — only for warnings that carry a type (e.g.
  `py:duplicate`, `ref.*`, `myst.*`). Note `py:duplicate` (Python domain) is a
  different key from `ref.duplicate`; using the wrong one suppressed nothing for
  65 warnings.

Prefer fixing in `process_docstring` over suppression: suppressed warnings still
render malformed markup on the page.

### Directive vs literal-block rule

`.. note::` and `text::` both end in `::`, but only the latter opens a literal
block. Trimming every trailing `::` turned `.. note::` into plain text and broke
the indentation fix. Guard with `startswith(".. ")`.

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
