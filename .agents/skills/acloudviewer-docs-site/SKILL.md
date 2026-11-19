---
name: acloudviewer-docs-site
description: Build, size, version and publish the ACloudViewer website plus all documentation versions. Use when changing docs build/deploy scripts, adding or fixing docs pages or notebooks, cutting a doc version, debugging the live site, or reducing artifact size. Covers the whole surface end to end; acloudviewer-docs-pipeline holds the shorter operational checklist.
---

# ACloudViewer Docs Site — Full Reference

Everything here was learned by breaking this pipeline and measuring the result.
Where a number appears, it was measured; where a rule appears, it came from an
incident.

---

## 1. What the site is

Two products ship from one Pages artifact:

| Path | Content | Owner |
|---|---|---|
| `/` | marketing website (project info, downloads, quick links) | `docs/` minus build output |
| `/documentation/` | redirect stub → newest `vX.Y.Z` | `docs/automation/scripts/deploy_unified_docs.sh` |
| `/documentation/dev/` | main-branch docs | CI on `main` |
| `/documentation/vX.Y.Z/` | immutable per-release docs | release assets |
| `/documentation/versions.json` | version manifest for the switcher | generated per deploy |
| `/404.html` | legacy deep-link rewriter | calibrated per deploy |
| `/sitemap.xml` | regenerated per deploy | generated per deploy |

**The version manifest is derived from directories on disk, not from a release
list.** That is deliberate: it is the only answer that cannot disagree with what
is actually served. The cost is that a missing directory removes the version
from the site *and* from every probe that iterates the manifest — see §7.

---

## 2. Build chain

```
workflow documentation.yml
  └─ job docs-guards            seconds; blocks the expensive job
  └─ job build-docs
       └─ docker/Dockerfile.docs  (ubuntu 22.04, sets CI=true, NOT GITHUB_ACTIONS)
            └─ util/ci_utils.sh::build_docs
                 build 1  headless (BUILD_GUI=OFF, ENABLE_HEADLESS_RENDERING=ON)
                          → executes notebooks, writes source/tutorial/**
                 build 2  GUI (BUILD_GUI=ON, ENABLE_HEADLESS_RENDERING=OFF)
                          → renders everything into docs/_out/html
            └─ docs/make_docs.py --sphinx --doxygen --strict
       └─ archive acloudviewer-<tag|sha>-docs.tar.gz
  └─ job deploy-docs            main or release only
       └─ prepare_website.sh   + restore archives + deploy_unified_docs.sh
```

### Why two builds

`ENABLE_HEADLESS_RENDERING` and `BUILD_GUI` are mutually exclusive in
`CMakeLists.txt`. Notebooks need an offscreen context to run; autodoc for
`visualization.{gui,rendering}` needs the real GUI module. One build cannot do
both.

**Measured cost:** build 1 is ~1985 s (33 min) of which the C++ compile is the
whole thing; notebook execution adds ~772 s (13 min). Build 2 compiles the same
tree again with four flags flipped. This is known and accepted — do not "fix"
it by reworking the build strategy without measuring a replacement.

### make_docs.py builder order

```
process_doxyfile_template → Doxygen → cpp_api/api/
process_rst_templates     → source/**.rst from *.in.rst
PyAPIDocsBuilder          → source/python_api/   (needs importable cloudViewer)
PyExampleDocsBuilder      → source/python_example/
JupyterDocsBuilder        → source/tutorial/
SphinxDocsBuilder         → _out/html/
copy_plugin_showcase_assets
```

Order matters: Doxygen writes into `_out/html/cpp_api/api/`, which must not
collide with Sphinx-generated `cpp_api/{overview,plugins,quickstart}.html`.

### Incremental builds

`main()` deletes the output directory whenever `--sphinx` or `--doxygen` is
passed — not only with `--clean`:

```python
if args.clean or (args.sphinx or args.doxygen):
    _create_or_clear_dir(html_output_dir)
```

Sphinx's doctree cache defaults to `<build>/.doctrees`, inside that directory,
so without an explicit `-d` every build re-parses every source. Pass a
**sibling** directory; putting it inside is deleted along with everything else.
Measured: inside → second build logs `[new config] 155 added`;
sibling → `loading pickled environment... done`. Cold 27.99 s → warm 23.76 s on
32 cores; the win is larger on a slow runner.

---

## 3. Local verification — do this first, always

CI costs 40–70 minutes per round. Anything answerable locally must be answered
locally. The old excuse "cloudViewer only exists in the CI image" is false: this
repository carries a build tree.

```bash
# one-time
python3 -m pip install plotly dash ipykernel
python3 -m ipykernel install --user --name python3

# every time before pushing
bash docs/automation/scripts/local_docs_check.sh            # all 28 notebooks
bash docs/automation/scripts/local_docs_check.sh <name>     # one
```

The script finds `build_app/lib/python_package`, proves it imports, executes
notebooks with nbconvert exactly as `make_docs.py` does, counts error outputs,
and skips the six notebooks CI does not execute.

**`python3 -I` ignores `PYTHONPATH`.** Every local docs command must run without
`-I` or with `sys.path` manipulation, otherwise `cloudViewer` "cannot be
imported" and the python_api guard fails for the wrong reason.

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

pandoc must be **3.9** to match CI. `pip install pypandoc_binary` puts it under
`site-packages/pypandoc/files/`; the distro package is 2.9.2.1 and emits
different heading adornments, which makes diagnostics disagree with CI.

### Guard ladder

| Command | Proves |
|---|---|
| `test_deploy_unified_docs.sh` | deploy contract, 37 assertions |
| `test_prepare_website.sh` | website contract, 18 assertions |
| `check_notebook_headings.py` | no illegal RST transitions, 34 notebooks |
| `check_workflow_limits.py` | no `run:` scalar near the 21000-char cliff |
| `verify_docs_site.py` | browser behaviour: switcher, deep links, 404 |
| `local_docs_check.sh` | all notebooks execute cleanly |

---

## 4. Size budget

Two independent limits exist and they are easy to confuse:

| Limit | Applies to | Enforced by |
|---|---|---|
| 1 GB | **compressed artifact** | `upload-pages-artifact` warns; `deploy-pages` fails *after* the build |
| 1 GB | **published site** (uncompressed) | Pages itself, message is generic |

`deploy_unified_docs.sh::check_pages_size_budget` measures `tar -czf - . | wc -c`.
That is gzip; Pages compresses with zstd. gzip is the conservative upper bound,
so the check over-reports rather than under-reports — safe direction, do not
"correct" it.

### Where the bytes go (measured on a full tree)

Doxygen output dominates. Levers, in measured order:

| Change | Saving | Loss |
|---|---|---|
| `DOT_IMAGE_FORMAT=svg` (from png) | 1947→921 MB raw, 1177→145 MB archived | none; also fixes PNG's 32767 px truncation |
| `INCLUDED_BY_GRAPH=NO` | ~197 MB | near-zero; nobody navigates "who includes this" |
| exclude third-party (`qSIBR/SIBR`) | ~72 MB | none, it is INRIA code |
| `SOURCE_BROWSER=NO` | ~361 MB | keeps 2303 header pages via `VERBATIM_HEADERS`; drops only .cpp |
| `HIDE_UNDOC_MEMBERS=YES` | 15 MB | not worth it |

**SVG wins because it is text.** Thousands of graphs share one compression
dictionary; PNG pixels have no cross-file redundancy (zstd ratio 0.913).

Per-version tree after all of the above: ~769 MB on disk / ~587 MB content /
~145 MB archived. A v3.9.4-era archive is 243 MB because it predates the
Doxyfile change.

### Multi-version growth

Each release adds one archive. Four archives plus dev approaches the 1 GB
artifact limit. When that happens the answer is to re-archive old versions with
the current Doxyfile, not to drop them.

---

## 5. Versioning

| Archive | Stored where | Name |
|---|---|---|
| release docs | that release's assets | `acloudviewer-<tag>-docs.tar.gz` |
| dev docs | the `main-devel` release | `acloudviewer-<sha>-docs.tar.gz` |

The two patterns are disjoint, so neither recovery path can pick up the other's
files. `main-devel` keeps only the newest dev archive; older ones are deleted
before upload.

**`versions.json` shape** (this exact shape; `docs/script.js` depends on it):

```json
{ "latest": "v3.9.4",
  "generated_at": "…",
  "versions": [ {"id":"dev","label":"Development (main)","kind":"dev"},
                {"id":"v3.9.4","label":"v3.9.4","kind":"latest"} ] }
```

`dev` first, then releases descending, `kind` ∈ `dev|latest|archive`.

### Directory matching is strict

`list_version_dirs` accepts only `^v[0-9]+\.[0-9]+\.[0-9]+$`. A tag like
`v3.9.5-rc1` is silently excluded from `versions.json`, from the sitemap, and
from archive restoration — a prerelease deploy writes the directory and the
next deploy drops it. If prereleases must be published, widen the regex in one
place only.

### Version resolution

The website's `@VERSION@` and `404.html`'s `@LATEST_STABLE_VERSION@` come from
`prepare_website.sh::resolve_website_version`: release tag → newest **published**
release → `version.txt` → `dev`.

`version.txt` is last on purpose. It is bumped *before* a release is cut, so
preferring it advertises a version that is not deployed. This ordering is
pinned by contract test Case 2.

`@DOC_LATEST@` is different: it is expanded by `deploy_unified_docs.sh` because
only that script knows which version was actually deployed.

### Historical versions are not recoverable

Only releases whose assets contain a docs archive can be restored. Measured:
of 18 releases, 2 had archives (`main-devel`, `v3.9.4`). `/documentation/v3.9.3/`
is a 404 and cannot be backfilled without rebuilding docs from that tag.

Guarantee archives from the next release onward; treat older ones as absent.

---

## 6. Deployment mechanics

`deploy_unified_docs.sh` inputs: `IS_RELEASE`, `RELEASE_VERSION`, `DOC_VERSION`,
`SITE_URL`. CWD must contain `website-deploy/`, `docs-output/`,
`preserved-versions/`; it writes `unified-deploy/`.

Rules that exist because each was violated:

- **Refuse `IS_RELEASE=true` with an empty `DOC_VERSION`.** Otherwise the dev
  branch runs: release docs overwrite `/documentation/dev/`, no archive is
  created, nothing fails.
- **Refuse `docs-output/` without `index.html`.** Doxygen output survives a
  failed Sphinx run, so the directory can hold ~96k files and no entry page.
- **Extract archives into a staging dir and require `index.html` before
  publishing.** Creating `vX.Y.Z/` first leaves an empty version directory on a
  corrupt archive, and `versions.json` then advertises pages that 404.
- **A release that has an archive but cannot restore it fails the job.** A
  release that never shipped docs only warns. Continuing would delete a
  published version.
- **`cp -r src/. dst/` never deletes.** Re-deploying the same version merges:
  deleted pages survive. `release: edited` re-runs everything, so this is
  reachable. Remove the target directory before extracting.
- **Only probe `/documentation/dev/` when the artifact contains it.** The
  restore is best-effort by design; failing the deploy for a state the
  pipeline tolerates is a false alarm.

### Branch runs never deploy

`should_deploy` is true only for `refs/heads/main` or a `release` event. A
`workflow_dispatch` on a feature branch builds and uploads an artifact that
goes nowhere, and `deploy-docs` is skipped entirely.

**A green branch run says nothing about the deployment path.** Two consequences
worth remembering:

- the deploy scripts can only be fully verified on `main` or a release;
- when reading a red run, check `head_sha` first — verifying the wrong commit
  looks exactly like a fix that did not work.

### Release builds used to fail on their own archive

The upload step recomputed the archive name that extraction had already chosen
and ran `cp` of a file onto itself (`GNU cp: are the same file`, exit 1). Upload
the resolved archive as-is; require exactly one match.

---

## 7. Silent-failure catalogue

Every entry produced plausible output and a zero exit status.

| Symptom | Mechanism | Guard now |
|---|---|---|
| `python_api/` missing | per-module import errors printed only; `conf.py` suppresses `toc.not_readable`/`ref.doc` | zero generated pages → raise |
| `cpp_api/api/` missing | `DoxygenDocsBuilder.run` prints and returns when Doxyfile/Doxygen output absent | — (untested) |
| two Getting Started pages 404 | `process_rst_templates` swallowed per-file errors; `index.rst` still lists them | any template failure → raise |
| docs built from the wrong notebook | a download overwrote the staged copy with upstream Open3D's file (1670 lines, `####`) | download removed; staged-set check |
| notebook failure ignored | guard required `GITHUB_ACTIONS`; Docker sets only `CI` | either env → raise |
| historical version vanished | archive restore warned and continued | staging + `index.html` requirement |
| downloads page empty | `scan_releases.py` emits `download_data`; a workflow fallback wrote `download_links` | fallbacks emit the real shape |
| `update-downloads` cleanup died | `to_entries` applied to `version_metadata`, which is an **array** → `Cannot use number (0) as object key` | `map()` over the array |
| 13 versions missing from the page | `fetch_releases(limit=5)` against 18 releases | `DEFAULT_RELEASE_LIMIT = 100` |
| everything "green" with 0 notebooks run | no guard on execution count; cloudViewer was never built | see §3 and the count check |

---

## 8. Notebook rules

nbsphinx converts each Markdown cell with pandoc and concatenates the result.
Three distinct traps:

**A standalone `---` becomes an RST transition.** docutils rejects a document
or section that begins with one (`ERROR/3 Document or section may not begin with
a transition`); `-W` makes it fatal. Editing blank lines around it only moves
the failure between "overline title" and "illegal transition". **Delete the
rule.** Verify through the real pipeline, never by feeding Markdown to docutils:

```python
import nbsphinx
rst = '\n'.join(nbsphinx.markdown2rst(''.join(c['source']))
                for c in nb['cells'] if c.get('cell_type') == 'markdown')
# then publish_doctree(rst) and count system_message with level >= 2
```

**Every notebook needs a toctree entry.** `-W` rejects
`document isn't included in any toctree`, and the page is invisible until
something forces Sphinx to render it.

**Headless screenshots can be degenerate.** `capture_screen_float_buffer()`
returns a scalar when the window never produced a frame; `np.asarray` then gives
a 0-d array and `PIL.Image.fromarray` raises `IndexError: tuple index out of
range`. `surface_reconstruction.ipynb` triggers this by rendering 565k points /
1.13M triangles. `cloudViewer_tutorial.py` guards on `ndim < 3` and skips the
inline image; all 28 notebooks share that helper.

---

## 9. pybind docstring warnings

`-W` fails on malformed RST that pybind11 generates from C++ signatures. It
cannot be fixed at the source, and fixing them one at a time wastes a full CI
round each.

Collect them all first — build **without** `-W`, then group by message:

```bash
python3 make_docs.py --sphinx --doxygen --execute-notebooks=never \
    --py-api-rst=always --py-example-rst=always --output-dir /tmp/warnscan
grep -oE 'WARNING: [^:]+' /tmp/warnscan.log | sed 's/WARNING: //' \
  | sort | uniq -c | sort -rn
```

Measured distribution on this tree (217 warnings):

| Count | Message |
|---|---|
| 74 | Inline interpreted text or phrase reference start-string without end-string |
| 11 | Block quote ends without a blank line; unexpected unindent |
| 9 | Literal block expected; none found |
| 7 + 7 | image file not readable / Could not obtain image size |
| 6 | Explicit markup ends without a blank line; unexpected unindent |
| 3 | Definition list ends without a blank line; unexpected unindent |
| ~14 | duplicate object description (same enum documented by several widgets) |
| 2 | Inline emphasis / Inline literal start-string without end-string |

Handled in `conf.py::process_docstring` (escape unbalanced `*`) and
`suppress_warnings`. Prefer suppression by exact type over rewriting generated
text: the text is machine output and any rewrite is lossy.

---

## 10. Workflow-level traps

- **A `run:` scalar over 21000 chars silently invalidates the whole workflow**
  — zero jobs, no logs, only an annotation in the run's HTML page.
  `check_workflow_limits.py` guards this; keep large logic in scripts.
- **`apt` retry count defaults to 0.** The Qt layer once fetched 116 MB in 65
  minutes and died on one dropped connection. The Dockerfile sets
  `Acquire::Retries`/timeout, retries three times with `--fix-missing`, then
  asserts `dpkg -s qtbase5-dev` — a loop `exit 0` only proves apt returned
  success.
- **`gh`'s log blob does not exist until the job completes.** A `BlobNotFound`
  from the logs API means "still running", not "no log".
- **A Docker job log interleaves multi-process stderr one character per line.**
  Reconstruct by stripping the `#<n> <time> ` prefix and rejoining.

---

## 11. Commit hygiene for this pipeline

The docs work sits beside an unrelated in-progress workstream in this
repository. `git add <files>` does **not** produce a focused commit when the
index already holds other staged paths: two docs commits shipped with 126 icon
SVGs and 39 plugin images riding along.

```bash
git reset                       # clear the index, keep the worktree
git add <exactly the files>
git diff --cached --name-only   # read this before committing
```

---

## 12. When something is wrong

1. **Read the failing step name**, not just the run status. `Build documentation
   Docker image` vs `Verify live deployment` are entirely different problems.
2. **Check `head_sha`.** Verifying the wrong commit looks like a fix that did
   not work.
3. **Classify: code or infrastructure.** A Qt `Connection timed out`, a 404 from
   archive.ubuntu.com, or a runner OOM is not your diff.
4. **Reproduce locally** (§3). If it cannot be reproduced locally, say so and
   name what blocks it — do not push to find out.
5. **Only then** push a single focused commit.
