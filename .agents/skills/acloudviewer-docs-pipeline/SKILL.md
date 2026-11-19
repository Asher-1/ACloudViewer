---
name: acloudviewer-docs-pipeline
description: Build, verify and deploy the ACloudViewer documentation site without breaking it. Use when touching docs/make_docs.py, docs/source/conf.py, docs/Doxyfile.in, docs/automation/scripts/*, or .github/workflows/documentation.yml; when a docs CI run is red; or when /documentation/dev/ or a version archive 404s on the live site.
---

# ACloudViewer Docs Pipeline Skill

This file is the engineering contract for the documentation build and the
GitHub Pages deployment. Every rule below comes from a defect that actually
reached CI or the live site. Where a rule names a file and line, treat that
path as the place to look first, not as a permanent guarantee.

## Layout contract

```
/                             static website (docs/, minus _out and _build)
/documentation/index.html     redirect stub -> ./<highest vX.Y.Z>/index.html
/documentation/dev/           main-branch build
/documentation/vX.Y.Z/        immutable per-release archive
/documentation/versions.json  {latest, generated_at, versions[{id,label,kind}]}
/404.html                     calibrated: var version = '<latest>'
/sitemap.xml                  regenerated per deploy
```

`versions.json` is derived from **version directories that exist on disk**, not
from a release list. That is deliberate: it is the only answer that cannot
disagree with what is served. The consequence is that a missing directory
removes the version from the site *and* from every probe that iterates the
manifest, so archiving failures are invisible unless explicitly checked.

## Build chain

```
workflow documentation.yml
  -> docker/Dockerfile.docs            (ubuntu 22.04, CI=true, NO GITHUB_ACTIONS)
     -> util/ci_utils.sh::build_docs
        build 1 (headless, BUILD_GUI=OFF)   execute notebooks
        build 2 (GUI, ENABLE_HEADLESS_RENDERING=OFF)  render everything
     -> docs/make_docs.py --sphinx --doxygen --strict
  -> archive acloudviewer-<tag|sha>-docs.tar.gz
```

The two builds exist because `ENABLE_HEADLESS_RENDERING` and `BUILD_GUI` are
mutually exclusive (`CMakeLists.txt`): notebooks need an offscreen context,
autodoc for `visualization.{gui,rendering}` needs the real GUI module.

`make_docs.py` builder order matters: Doxygen -> `python_api/` -> `python_example/`
-> notebook staging -> Sphinx -> plugin assets. Doxygen is copied to
`cpp_api/api/` so it does not collide with the Sphinx-generated
`cpp_api/{overview,plugins,quickstart}.html`.

## Hard rules

### Never fetch documentation content from another repository

A download used to install `t_icp_registration.ipynb` from
`github.com/isl-org/Open3D/files/8243984`. It wrote another project's file over
the copy staged from `docs/jupyter/`, so the build documented a library nobody
here maintains, and every edit to the checked-in notebook was discarded before
Sphinx read it. It also made the build depend on a third-party attachment
surviving.

Every notebook lives in `docs/jupyter/` and is git-tracked. If one is missing,
fail. Do not add a fallback; a fallback that substitutes another project's file
is worse than a missing file, because it produces plausible wrong output.

The build verifies the whole staged set: every `docs/jupyter/**/*.ipynb` must
exist under `docs/source/tutorial/` after the copy step.

`docker/Dockerfile.docs` still clones CloudViewer-ML on purpose -- that is a
*dependency* checkout (31 `cloudViewer.ml.*` modules in
`documented_modules.txt`, plus `docs/tensorboard.md`). Do not remove it while
removing content downloads; they are unrelated.

### A notebook that fails to execute must fail the build

The guard once required `GITHUB_ACTIONS`, but the build runs in Docker, which
sets only `CI=true`. The log printed `❌ Execution ... failed (this will cause CI
to fail)` and the build continued, shipping partial output. Check `CI` or
`GITHUB_ACTIONS`; keep the save-on-failure behaviour only when neither is set.

### Markdown rules become illegal RST transitions

nbsphinx converts each markdown cell with pandoc and concatenates the result.
A standalone `---` becomes an RST *transition*, and docutils rejects a document
or section that begins with one (`ERROR/3 Document or section may not begin with
a transition`), which `-W` makes fatal.

Editing blank lines around the rule does not fix it; it only moves the failure
between "overline title" and "illegal transition". Delete the rule.

Verify with the real pipeline, never by feeding markdown straight to docutils:

```python
import nbsphinx
rst = '\n'.join(nbsphinx.markdown2rst(''.join(c['source']))
                for c in nb['cells'] if c.get('cell_type') == 'markdown')
# then publish_doctree(rst), count system_message with level >= 2
```

CI uses pandoc 3.9 (via `pypandoc_binary`), not the distro pandoc 2.9.2.1.
Reproduce with the same pandoc or the diagnostics will not match.

Guard: `docs/automation/scripts/check_notebook_headings.py`, run by
`docs-guards` before the 30-minute Docker build.

### Every notebook must be in a toctree

`-W` rejects `document isn't included in any toctree`. Tracked notebooks with no
toctree entry are invisible until something forces Sphinx to render them, so add
each new notebook to its directory's `index.rst` in the same change.

### Swallowing an exit status is the standard way this pipeline breaks

`build_docs()` runs under `set +u`, not `set -e`. Every command there is a
candidate for a silent failure. When adding one, wrap it:

```bash
if ! python make_docs.py ...; then
    set +x
    echo "❌ ... failed" >&2
    return 1
fi
```

Known silent-failure sites to check when content goes missing:

| Symptom | First place to look |
|---|---|
| `python_api/` empty or absent | `PyAPIDocsBuilder.generate_rst` catches per-module exceptions and only prints; then `conf.py` suppresses `toc.not_readable` and `ref.doc` |
| `cpp_api/api/` absent | Doxygen missing/failed: `DoxygenDocsBuilder.run` prints and returns |
| tutorial pages missing | notebook execution or staging; the staged-set check exists for this |
| downloads page blank | schema mismatch: `scan_releases.py` emits `download_data`; a workflow fallback once wrote `download_links`, which the frontend does not read |

### Redeploying the same version must not merge

`cp -r src/. dst/` never deletes. A `release: edited` event re-runs the whole
pipeline, so new output is merged over the existing archive: deleted pages
survive, and the docs come from the default branch rather than the tag. Remove
the target directory before extracting or copying a version archive.

### Never publish an empty version directory

Creating `vX.Y.Z/` before knowing the archive is readable leaves an empty
directory on a corrupt archive. `list_version_dirs` counts directories, so
`versions.json` then advertises a version whose pages all 404, and the live
probe only checks the ids the manifest lists. Extract into a staging directory,
require `index.html`, then move into place.

A release that *has* an archive but cannot restore it is a regression (a
published version is about to vanish): fail. A release that never shipped docs
is not: warn.

## Deployment

Branch builds never deploy. `should_deploy` is true only for
`refs/heads/main` or a `release` event, so a `workflow_dispatch` on a feature
branch builds and uploads an artifact that goes nowhere. **A green run on a
branch says nothing about the deploy path.** Only main or a release exercises
`deploy-docs`.

Archive storage:

| Archive | Location | Naming |
|---|---|---|
| release docs | that release's assets | `acloudviewer-<tag>-docs.tar.gz` |
| dev docs | the `main-devel` release | `acloudviewer-<sha>-docs.tar.gz` |

`main-devel` keeps only the newest dev archive; older ones are deleted before
upload. Both patterns are disjoint, so neither recovery path can pick up the
other's files.

GitHub Pages replaces the whole artifact on each deploy, which is why dev is
carried forward from `main-devel` on release builds.

Two independent limits exist: the **uploaded artifact** (compressed, 1 GB) and
the **published site** (uncompressed). `upload-pages-artifact` only warns when
the artifact is oversized; `deploy-pages` then fails after the build, leaving
the previous deployment live. Measure the compressed artifact
(`tar -czf - . | wc -c`), not `du`.

## Verification ladder

1. `python3 -m py_compile docs/make_docs.py`, `bash -n` on every touched script,
   YAML-parse the workflow.
2. `bash docs/automation/scripts/test_deploy_unified_docs.sh` -- the deploy
   contract (currently 37 assertions across 9 cases).
3. `python3 docs/automation/scripts/check_notebook_headings.py`.
4. `python3 docs/automation/scripts/check_workflow_limits.py` -- a `run:` scalar
   over 21000 characters silently invalidates the whole workflow file (zero
   jobs, no logs).
5. Local Sphinx build from `docs/`:
   `python3 make_docs.py --sphinx --strict --execute-notebooks=never --py-api-rst=always --py-example-rst=always --output-dir /tmp/docs-out`
   Expect `python_api/` warnings locally: the `cloudViewer` module only exists
   inside the CI image.
6. CI is the final gate. Confirm the failure **step** changed, not just that the
   run is red -- a different failing step means progress.

## Diagnosing a red run

Fetch the job log with the `gh` CLI (the log blob does not exist until the job
completes, so a 404 means "still running", not "no log"):

```bash
JOB=$(gh api repos/Asher-1/ACloudViewer/actions/runs/<run>/jobs \
      --jq '.jobs[]|select(.name=="build-docs")|.id')
gh api "repos/Asher-1/ACloudViewer/actions/jobs/$JOB/logs" > /tmp/build.log
```

Read the **first** error, not the last. `sphinx-build -W` reports one warning and
exits; later lines are the Python traceback that unwinds through
`subprocess.check_call`.

## Anti-patterns seen in this repository

- **Fixing a warning without reproducing the pipeline.** Two consecutive commits
  adjusted blank lines around a `---` in a notebook. Neither could work: the
  file they edited was being overwritten by a download, and the construct itself
  was invalid.
- **Editing a file the build does not read.** Before changing a notebook or a
  template, confirm the build actually consumes *that* file. The tell here was
  cheap: the CI diagnostic quoted a `^^^^` underline, which pandoc only emits
  for level-4 headings, while the checked-in notebook uses `###`. Different
  input, different output.
- **Treating a red run as one bug.** Three independent defects were live at once
  (archive re-copy, dev probe, notebook overwrite). Fixing the first two changed
  nothing visible until the third was found. After each fix, confirm the
  **failing step moved**; a run that is still red at the same step means the
  fix did not take effect, not that the problem is hard.
- **Assuming CI covers deployment.** Branch runs skip `deploy-docs` entirely.
- **Adding `|| true` or `|| echo` to get a red run green.** That is how the
  pipeline reached a state where "success" meant nothing.
- **Widening the blast radius of a request.** "Stop downloading one notebook"
  does not mean "remove the CloudViewer-ML checkout": that checkout supplies the
  dependencies for 31 documented ML modules. Remove one thing at a time and say
  what was removed.

## Verifying locally without the CI image

`cloudViewer` only exists inside the CI image, so a local build cannot produce
`python_api/`. Two consequences:

- `--py-api-rst=never` reproduces everything except the Python API.
- The Python API guard will fail a local `--py-api-rst=always` build. That is
  correct behaviour, not a local-workflow bug.

pandoc must match CI's version (3.9). `pip install pypandoc_binary` provides it
at `~/.local/lib/python3.11/site-packages/pypandoc/files/pandoc`; put that
directory on `PATH`. The distro package is 2.9.2.1 and produces different
heading adornments, which makes diagnostics disagree with CI in a way that is
easy to misread as a content problem.
