# Publishing @asher-1/acloudviewer

Release path: `.github/workflows/npm-publish.yml` in **this** repository.
It publishes the installer bootstrap in `npm/` to both registries in one run:

1. **npmjs.org** — primary channel, end users install with zero setup.
2. **GitHub Packages** — mirror at
   https://github.com/Asher-1/ACloudViewer/packages (uses `GITHUB_TOKEN`,
   no extra secret; publishes even when `NPM_TOKEN` is absent).

**Publishing is fully automated.** The package version is the **channel
router**, not just metadata: a stable version pins the GitHub release
`v<version>`; a prerelease (`X.Y.Z-dev.<stamp>`, contains `-`) tracks the
rolling `main-devel` assets. `--release <tag>` overrides. Do not hand-edit
`npm/package.json` to release — CI reconciles it.

---

## One-time setup (per npmjs account)

1. **Account**: register at https://www.npmjs.com as **`asher-1`**
   (lowercase — must match the `@asher-1` scope; GitHub's `Asher-1` case
   does not carry over). No organization needed: a personal account whose
   username equals the scope can publish `@asher-1/*` public packages.
   Enable 2FA — npm now requires **security-key / passkey (WebAuthn)**;
   authenticator-app TOTP was retired. Signup rate-limits ("Too many
   requests") are per exit IP: switch network / incognito, wait ~1 h.
2. **Granular publish token** (classic tokens are retired): npmjs.com →
   Access Tokens → Generate New Token, with exactly these settings:
   - **Bypass two-factor authentication: CHECKED** (else CI hits `EOTP`);
   - Permissions: **Read and write (publish and stage)** — not read-only,
     not stage-only (staged versions need a human to promote);
   - Packages: All Packages (or the `@asher-1` scope once it exists);
   - Expiration: max **90 days** — renewal is mandatory (below).
3. **GitHub secret**: repo Settings → Secrets → Actions → `NPM_TOKEN`.

**Trusted Publishing (OIDC)** is the no-token alternative: package page →
Settings → Trusted publishing → GitHub Actions, owner `Asher-1`, repo
`ACloudViewer`, workflow `npm-publish.yml`, **allow `npm publish`**. The
workflow already has `id-token: write` and npm ≥ 11.21 (Node 24 + explicit
upgrade, needed for OIDC dist-tag operations). "Signed provenance
statement" in the run log proves OIDC was used; `NPM_TOKEN` remains the
automatic fallback.

**Token renewal (~every 90 days)**: symptom is the npmjs step failing with
`ENEEDAUTH` / `401` / `EOTP` while version resolution still passes.
Generate a fresh token with the same four settings → update the secret →
re-run the failed job (reconcile is idempotent, never re-publishes). The
mirror keeps working meanwhile.

---

## Automated release (default path)

Two channel invariants, enforced by **reconcile** on every trigger (the app
version is the one carried by the `main-devel` release assets):

- **STABLE — `latest` == app version.** Publish is gated on the pinned
  release `v<version>` existing **with assets**; without it the publish is
  deferred, never faked.
- **DEV — every complete `main-devel` asset wave → one
  `X.Y.Z-dev.<stamp>` under `next`.** Completeness = `.run` + `.exe` +
  `.dmg` all present; early uploader completions skip, the last one
  publishes. The stamp derives from the release `updated_at`, so the five
  completion events converge to one version (E409 treated as success).
  A `next` publish NEVER moves `latest`.

| Stable registry state | Reconcile action |
|----------------|------------------|
| `latest == app version` | skip |
| `latest < app version` | publish `@<app version>` under `latest` — only when the pinned release `v<app version>` exists with assets |
| `latest > app version` | re-point `latest` back to `<app version>` on both registries (extra versions stay; npm numbers are immutable) |

| Trigger | Behaviour |
|---------|-----------|
| Platform build finishes (`workflow_run`: Windows / MacOS / Ubuntu / Ubuntu CUDA / Ubuntu Wheel — the `main-devel` uploaders) | reconcile both channels |
| Push to `main` touching `npm/**` or the workflow | reconcile both channels |
| **Release published** (any release, incl. a new pinned `v<version>`) | reconcile — closes the gap where the pinned release is created *after* the last build wave: publishing it with assets immediately enables the deferred stable publish |
| Tag `npm-vX.Y.Z` / manual dispatch | emergency paths (below) |

**Timing model**: event-driven, not polling. No run "waits" for a release;
each event re-reads the current release state and acts only when the gates
pass. A release created but still uploading assets fails the asset gate and
is retried on the next event (build completion, npm push, or the release
event itself). `concurrency: npm-publish` serializes near-simultaneous runs.

Steady state — what you actually do:

- **App release**: bump `libs/cloudViewer/version.txt`, merge to `main`,
  publish the GitHub release `v<new version>` **with the platform
  installers attached**. The release event (or next build completion)
  publishes `npm@<new version>` under `latest` automatically.
- **Daily commits, same app version**: nothing. Stable SKIPs (aligned);
  each complete asset wave refreshes `next`. Zero npm errors either way.
- Outcomes are in the run's step summary (per-channel stable/dev rows,
  derived from actual step results).

### Version scheme facts

- **Burned numbers**: a published npm version can never be republished,
  even after unpublish (409). 3.9.6 / 3.9.7 were consumed by early
  mistakes and are permanently skipped. **Never delete ALL versions**:
  that drops the package + its Trusted Publisher config and triggers a
  24 h publish ban.
- **Transitional 3.9.5 (current)**: the published `npm@3.9.5` tarball
  predates channel routing and is immutable, so it keeps fetching rolling
  `main-devel` assets (there is no pinned `v3.9.5` either). True stable
  semantics begin with the next release: bump `version.txt` to **3.9.8**,
  create the pinned `v3.9.8` release with installers → reconcile publishes
  `npm@3.9.8` whose bootstrap pins it. Until then `latest=3.9.5` is a
  rolling carrier by necessity; `@next` is the explicit test channel.
- **Testers**: `npm i -g @asher-1/acloudviewer@next` then
  `acloudviewer install` — always the current rolling build. Dev
  prerelease stamps are disposable, so dev publishes cost no stable slot.

### Zero-waste rehearsal (no burned versions)

npmjs has **no sandbox**. To rehearse the dual-channel flow for a future
release (e.g. 3.9.8) without spending real version numbers, publish to a
local [verdaccio](https://verdaccio.org/) registry — it implements the npm
registry HTTP API, so publish / view / dist-tags behave identically:

```bash
mkdir /tmp/rehearsal && cd /tmp/rehearsal && npm install verdaccio
# config.yaml packages: '@asher-1/*' -> access: $all, publish: $authenticated
#   (gotcha: $anonymous means UNAUTHENTICATED only; npm publish needs a login)
./node_modules/.bin/verdaccio --config config.yaml --listen 127.0.0.1:4873 &
# register a throwaway user: PUT /-/user/org.couchdb.user:<name> -> token;
# point a copy of npm/ at the local registry via .npmrc, then:
npm version 3.9.8 --no-git-tag-version --allow-same-version
npm publish --access public --tag latest --registry http://127.0.0.1:4873/
npm version 3.9.8-dev.<stamp> --no-git-tag-version --allow-same-version
npm publish --access public --tag next   --registry http://127.0.0.1:4873/
npm view @asher-1/acloudviewer dist-tags --registry http://127.0.0.1:4873/
# -> { latest: '3.9.8', next: '3.9.8-dev.<stamp>' }   (dev never moves latest)
```

Validated end-to-end (6/6 assertions + routing): stable→`latest`,
dev→`next`, coexistence, dev re-publish leaves `latest` unchanged,
duplicate stable→E409, and the installed bootstrap routes stable→`v3.9.8`
vs `@next`→`main-devel`.

---

## Emergency manual paths

Both assume `NPM_TOKEN` is configured. Reconcile re-points `latest` back to
the app version on its next run regardless.

```bash
# Dispatch (no commit needed; publishes exactly the given version):
gh workflow run npm-publish.yml -R Asher-1/ACloudViewer -f version=3.9.8
# optional: -f dist_tag=next  (land on a dedicated tag instead of latest)
# UI equivalent: Actions -> npm publish -> Run workflow.

# Tag push (strict contract: tag MUST equal the main-devel app version):
git tag npm-v3.9.8 && git push origin npm-v3.9.8
```

Dispatch applies `npm version` in the CI workspace without committing back,
so `npm/package.json` on `main` can lag the registry — reconcile resolves
against the registry, so drift never publishes a stale/duplicate version.
Tag runs use the workflow file **at the tagged commit**.

| Failure | Meaning / fix |
|---------|---------------|
| `409` on tag/dispatch | Version already taken — pick another (reconcile never 409s; it resolves first and treats E409 as convergence) |
| `403` npmjs | Token lacks `@asher-1` write, or scope owned by someone else |
| `tag npm-vX.Y.Z does not match` | Version drift — the check is the contract; fix and re-tag |
| `npm registry query failed (not a 404)` | Transient network; re-run (the workflow refuses to guess a 0.0.0 fallback) |
| `422` mirror | Scope/repo-owner mismatch — package must stay `@asher-1` on `Asher-1/ACloudViewer` |
| Run never started | Tag's commit (or dispatch branch) must contain the workflow file; `workflow_run` names must match the uploader workflows exactly |

---

## Post-publish verification

```bash
npm view @asher-1/acloudviewer version dist-tags   # npmjs (may lag ~1 min)
cd "$(mktemp -d)" && npm init -y >/dev/null
npm install -g @asher-1/acloudviewer
acloudviewer version                 # the published version
acloudviewer list                    # current release assets
acloudviewer install --dry-run       # matches an asset for this platform
acloudviewer install --silent --prefix ~/.local/share/ACloudViewer
~/.local/share/ACloudViewer/ACloudViewer.sh --version   # run via LAUNCHER
acloudviewer uninstall --yes         # QtIFW maintenancetool purge
```

Run via the launcher (`ACloudViewer.sh` / `.bat`), never the bare binary —
shared libs live in `lib/` (bare binary exits 127). The installer ships
three apps: `ACloudViewer.sh --version` and `Colmap.sh help` are
headless-safe (exit 0); `CloudViewer.sh` is a Filament GUI with no headless
exit — probe it under `timeout` (exit 124 + "FEngine created" = healthy).
Offline re-test: `install --from <local .run> --silent --prefix <dir>`.
Mirror install needs a PAT with `read:packages` (README "Channel 2").

Operational facts (verified 2026-11-16):

- **Coexistence with a manual `.run` install**: different default prefixes
  (`~/.local/share/ACloudViewer` vs `~/ACloudViewer`), no file conflict;
  same-directory reinstall = in-place upgrade. **Desktop entries are
  shared, last-install-wins**; uninstall removes the entries it owns and
  does not restore a previous owner — re-run the surviving installer to
  fix `Exec=` if needed.
- **2FA / web-auth**: sensitive CLI ops (`npm login`, `unpublish`) print a
  single-use `https://www.npmjs.com/auth/cli/<id>` link, valid ~5 min,
  confirmed in a browser with the passkey — that link IS the OTP
  replacement; expired links are permanently 404, always regenerate.
  Bypass-2FA tokens **cannot** unpublish (403 since 2026-07-31); recovery-
  code login triggers a 72 h security freeze. Emergency unpublish uses
  `workflow_dispatch` `unpublish_versions` (interactive web-auth per
  version) — delete the version `latest` points at **last**, keep the
  aligned one, never delete all.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| npmjs step `EOTP` / `401` / `ENEEDAUTH` | Token expired or wrong type — renew granular token (four settings above), update secret, re-run |
| `skipped (aligned)` but you expected a publish | `latest` already equals the app version — expected; publish happens only after the app version bumps |
| Stable deferred: `pinned release ... missing or asset-less` | Create/publish `v<version>` **with installers**; the release event re-triggers reconcile |
| `workflow_run` did not fire | Uploader renamed, or run on a non-default branch; use dispatch meanwhile |
| `install`: `segment N/M: retry ... stalled` | Expected resilience: 60 s idle abort + Range resume; persistent = network outage, retry or `--parallel 1` |
| `install`: `server does not support range requests` | Informational single-stream fallback, still works |
| Mirror install `404` / `401` | `.npmrc` scope mapping missing / PAT lacks `read:packages` or expired — see README "Channel 2" |
| Bare `ACloudViewer` exits 127 | Use the launcher (sets `LD_LIBRARY_PATH`) — not a packaging defect |

## Sync contract

Asset matching in `npm/bin/acloudviewer.js` mirrors
`cli_anything/acloudviewer/utils/installer.py` (`find_matching_app` /
`find_matching_wheel` / QtIFW console invocations). Linux selection is
**closest-compatible**: candidate tags = exact distro tag first, then
published Ubuntu tags whose build-host glibc floor ≤ the host glibc,
newest-first — a host never installs a build tagged newer than its glibc
allows, and a missing exact tag falls back to the newest older build
(both files keep the same floor table; update both together). Flavor
selection is **explicit opt-in**: CPU is the default; CUDA (desktop **and
wheel**) is downloaded only with `--cuda`, and only after a pre-download
driver gate — the
`nvidia-smi` banner's max supported CUDA **major** must be ≥ the build's
CUDA major (`REQUIRED_CUDA`, currently 12.6, sourced from
`docker/docker_build.sh` and `windows.yml`; bump both mirrors when CI
bumps the toolkit). Minor versions within a major are compatible (CUDA
minor-version compatibility), so a 12.0 driver passes a 12.6 build.
Incompatible interactive hosts get re-check / abort / CPU choices; on a
**major mismatch only**, a fourth choice hands over to a source build:
the bootstrap ensures `cli-anything-acloudviewer` is installed (pip) and
runs its one-click `install source`; installer.py mirrors this via
`sys.executable -m cli_anything.acloudviewer install source`. There is
no *automatic* source-build fallback (full builds are too slow and
toolchain-heavy); source compilation is only ever an explicit user
choice. Non-interactive shells refuse with guidance (`--force-cuda`
bypasses). The download
engine (ranged segments, 60 s stall abort, Range resume) and `uninstall`
are **bootstrap-only** — installer.py has neither; do not "re-sync" them
away. If release asset names or QtIFW console syntax change, update
**both** files; validate the silent path on Linux
(`--silent --prefix <dir>`).