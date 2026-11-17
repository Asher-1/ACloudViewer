# ACloudViewer Documentation Deployment Guide

How the documentation site is built, versioned, and published.

## 📋 Overview

**https://asher-1.github.io/ACloudViewer/** hosts three things:

| Path | Content |
|------|---------|
| `/` | Main website (homepage, downloads, donation) |
| `/documentation/` | Versioned Sphinx + Doxygen documentation |
| `/downloads_data.json` | Download metadata, generated from the GitHub API |

Deployment uses **GitHub's artifact-based Pages pipeline**
(`actions/upload-pages-artifact` + `actions/deploy-pages`). There is **no
`gh-pages` branch** and no `peaceiris/actions-gh-pages` step. Everything is
uploaded as a single Pages artifact, which keeps git history free of build
output and means contributors never download documentation in a `git pull`.

> **Important consequence**: an artifact deployment **replaces the whole
> site**. It has no `keep_files` option. Every version that must survive a
> deploy has to be restored explicitly on each run — see
> [Version persistence](#-version-persistence).

## 🏗️ Deployment Architecture

```
Repository (main)
├── docs/                     website + Sphinx source + deploy scripts
│   ├── index.html            main website
│   ├── 404.html              site-wide 404 with version-aware redirect
│   ├── source/               Sphinx documentation source
│   └── automation/scripts/   deploy + guard scripts
└── .github/workflows/
    └── documentation.yml     build → merge → deploy → verify

                ↓ GitHub Actions

Pages artifact (assembled in ./unified-deploy)
├── index.html                    main website
├── downloads_data.json
├── 404.html                      calibrated version redirect
├── sitemap.xml                   regenerated per deploy
└── documentation/
    ├── index.html                redirect → newest vX.Y.Z (or dev)
    ├── versions.json             generated manifest (single source of truth)
    ├── dev/                      development build (main branch)
    └── v3.9.4/ , v3.9.5/ ...     permanent release archives
```

## 🚀 Triggers

| Event | Build | Deploy |
|-------|-------|--------|
| `push` to `main` | ✅ | ✅ |
| `release` (released, edited) | ✅ | ✅ |
| `pull_request` (opened, synchronize) | ✅ | ❌ artifact only |
| `workflow_dispatch` | ✅ | ❌ unless run on `main` |

Deployment is gated on `should_deploy`, which is true only for `main` and
release events. Pull requests build and upload an artifact so a reviewer can
download the built site, but never touch the live site.

## 📦 Documentation Versioning

### Directory semantics

| Path | Meaning |
|------|---------|
| `/documentation/` | Redirect page → newest `vX.Y.Z`, or `/dev` if no release exists yet |
| `/documentation/dev/` | Development build from `main`, titled `main-<hash>` |
| `/documentation/vX.Y.Z/` | Permanent archive, created from a release event |
| `/documentation/versions.json` | Manifest of versions that actually exist |

### Single source of truth

`versions.json` is **generated at deploy time** from the version directories
that really exist in the artifact:

```json
{
  "latest": "v3.9.4",
  "generated_at": "2026-11-17T04:49:31Z",
  "versions": [
    {"id": "dev",    "label": "Development (main)", "kind": "dev"},
    {"id": "v3.9.4", "label": "v3.9.4", "kind": "latest"}
  ]
}
```

Everything downstream reads it: the sidebar version table, the landing-page
dropdown, the 404 redirect calibration, and `sitemap.xml`. A version cannot be
advertised unless its directory was deployed, because the list is derived from
the directories rather than maintained by hand.

This is a deliberate difference from Open3D, whose `versions.js` is a
hand-edited HTML table that must be updated on every release.

### Version persistence

Because an artifact deploy replaces the entire site, archives are restored on
every run before the artifact is uploaded:

1. **Release archives** — `acloudviewer-v<version>-docs.tar.gz` is downloaded
   from each matching GitHub Release.
2. **Development docs** — the latest dev archive is downloaded from the
   `main-devel` release, so a release deploy does not drop `/documentation/dev/`.

Both steps live in `.github/workflows/documentation.yml` under *Preserve
existing version archives* and *Preserving development documentation*.

**Archives only exist for v3.9.4 and later.** Earlier releases were published
before documentation archiving existed and have no `-docs.tar.gz` asset, so
`/documentation/v3.9.3/` and older cannot be backfilled. Version history is
guaranteed **from v3.9.4 onward**.

## ⚙️ GitHub Pages Configuration

1. **Settings → Pages**
2. **Source**: `GitHub Actions` (not "Deploy from a branch")
3. The `github-pages` environment is used by the deploy job; its URL is
   reported as `steps.deployment.outputs.page_url`.

If the source is left as "Deploy from a branch", artifact deployments are
ignored and the site will keep serving whatever that branch contains.

## 🔧 Local verification

The deploy pipeline is testable without CI:

```bash
# Deploy-script contracts: manifest shape, latest resolution, redirects,
# 404 calibration, archive restore, switcher propagation
bash docs/automation/scripts/test_deploy_unified_docs.sh

# Browser end-to-end: table/dropdown population, path-preserving version
# switch, missing-page fallback, 404 rewriting
python3 docs/automation/scripts/verify_docs_site.py

# Workflow files GitHub would reject before running any job
python3 docs/automation/scripts/check_workflow_limits.py

# Version copies agree with libs/cloudViewer/version.txt
python3 util/sync_version.py --check
```

All four run in the `docs-guards` job before the documentation build starts,
so a structural defect fails in seconds instead of after a Docker build.

### Manual dry run of the merge step

```bash
mkdir -p /tmp/dryrun && cd /tmp/dryrun
mkdir -p website-deploy docs-output
echo '<html>site</html>' > website-deploy/index.html
echo '<html>docs</html>' > docs-output/index.html
cp /path/to/ACloudViewer/docs/404.html website-deploy/404.html

IS_RELEASE=false RELEASE_VERSION= DOC_VERSION= \
  bash /path/to/ACloudViewer/docs/automation/scripts/deploy_unified_docs.sh

cat unified-deploy/documentation/versions.json
```

## 🔍 Verification

The deploy job ends with a **live smoke test** that probes the published site
and fails the run when it is wrong:

| Probe | Expectation |
|-------|-------------|
| `/` | 200, contains `ACloudViewer` |
| `/documentation/` | 200, contains the redirect |
| `/documentation/versions.json` | 200, valid manifest |
| `/documentation/dev/index.html` | 200 |
| `/documentation/dev/cpp_api.html` | 200 |
| Every `id` in `versions.json` | its `/documentation/<id>/index.html` is 200 |
| `/404.html` | carries a calibrated `var version = 'vX.Y.Z'` |

Each probe retries for up to ~100 s, because Pages serves a new artifact
asynchronously. A failure here fails the workflow — a deploy that cannot be
reached is not a successful deploy.

### Manual checks

```bash
curl -sI  https://asher-1.github.io/ACloudViewer/documentation/
curl -s   https://asher-1.github.io/ACloudViewer/documentation/versions.json | python3 -m json.tool
curl -sI  https://asher-1.github.io/ACloudViewer/documentation/dev/index.html

# Every deployed version must answer
for v in $(curl -s https://asher-1.github.io/ACloudViewer/documentation/versions.json \
           | grep -oE '"id": *"[^"]+"' | sed 's/.*: *"//;s/"$//'); do
  printf '%-10s ' "$v"
  curl -s -o /dev/null -w '%{http_code}\n' \
    "https://asher-1.github.io/ACloudViewer/documentation/$v/index.html"
done
```

## ↩️ Rollback

Pages keeps a **deployment history**, and the artifact for each deployment is
retained. Rolling back does not require a commit.

### Fastest: re-deploy a previous artifact

1. Open **Actions → Documentation → the run you want to restore**
2. **Re-run all jobs**

The run rebuilds from the same commit and re-deploys that artifact.

### Restore a previous site version from Pages

1. **Settings → Environments → github-pages**
2. Open **Deployment history**
3. Pick the last known-good deployment → **Re-deploy**

This restores the exact artifact that was live at that time, including its
version archives.

### If a bad docs build was merged

1. Revert the offending commit on `main` (`git revert <sha>`) and push. The
   push triggers a normal build + deploy.
2. If the revert is not yet ready, use one of the two options above to restore
   service first, then fix forward.

**Not** a rollback path: there is no `gh-pages` branch to reset. Any procedure
based on `git push --force origin gh-pages` does not apply to this setup.

### Version archives after a rollback

`versions.json` is regenerated from the restored artifact's directories, so a
rolled-back site automatically advertises exactly the versions it contains.
No manual manifest edit is needed.

## 🐛 Troubleshooting

### Workflow is red with zero jobs and no logs

A run whose **name is the file path** (`.github/workflows/documentation.yml`
instead of `Documentation`) and that has **no jobs** was rejected at parse
time. GitHub validates a workflow before creating any job, so nothing runs and
no logs exist.

Common cause: a single `run:` scalar exceeding GitHub's 21000-character
expression limit.

```bash
python3 docs/automation/scripts/check_workflow_limits.py
```

### Live smoke test fails

- **All probes time out** → Pages has not picked up the artifact yet (wait and
  re-run), or the Pages source is set to a branch instead of GitHub Actions.
- **`versions.json` 404** → the artifact was assembled without the deploy
  script. Check the *Merge website and documentation* step's log.
- **`404.html` not calibrated** → no version directory was present in the
  artifact; inspect the *Preserving development documentation* step.
- **A version listed but 404** → `versions.json` was generated from a directory
  that did not survive into the artifact. Compare the merge step's directory
  listing with the manifest.

### Documentation Not Updated

```bash
curl -sI https://asher-1.github.io/ACloudViewer/documentation/dev/index.html | grep -i last-modified
```

A stale `Last-Modified` means the last deploy did not run (check the workflow
run for that commit) rather than a caching problem.

### Version selector empty

The selector reads `documentation/versions.json`. Check it first:

```bash
curl -s https://asher-1.github.io/ACloudViewer/documentation/versions.json
```

If it is valid but the UI is empty, look at the browser console for a
`version_switch.js` error. If it is 404, the merge step did not run.

### Missing plugin images

The docs build runs with `-W` in CI, so a broken image reference fails the
build rather than shipping a 404. To reproduce locally:

```bash
cd docs
python make_docs.py --sphinx --strict
```

Images referenced from a plugin README must live under
`plugins/core/<Category>/<Plugin>/images/` — only that directory is synced into
the Sphinx tree.

## 🔒 Assets that must not be processed

**`.nojekyll`** (empty file at the deploy root) tells Pages to skip Jekyll.
Without it, underscore-prefixed paths such as `_static/` are stripped and every
stylesheet and script 404s.

## ✅ Pre-merge checklist

- [ ] `python3 docs/automation/scripts/check_workflow_limits.py`
- [ ] `python3 util/sync_version.py --check`
- [ ] `bash docs/automation/scripts/test_deploy_unified_docs.sh`
- [ ] `python3 docs/automation/scripts/verify_docs_site.py`
- [ ] `cd docs && python make_docs.py --sphinx --strict` (0 warnings)
- [ ] After merge, the run's *Verify live deployment* step is green

## 📚 Related

- **Build guide**: [BUILDING_DOCS.md](BUILDING_DOCS.md)
- **Scripts**: [scripts/README.md](scripts/README.md)
- **Version management background**: [../guides/VERSION_MANAGEMENT.md](../guides/VERSION_MANAGEMENT.md)

---

**Last Updated**: November 2026
**Maintainer**: ACloudViewer Team
