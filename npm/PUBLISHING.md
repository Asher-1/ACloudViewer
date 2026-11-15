# Publishing @asher-1/acloudviewer

Release path: `.github/workflows/npm-publish.yml` in **this** repository
(ACloudViewer). It publishes the installer bootstrap in `npm/` to both
registries in one run:

1. **npmjs.org** — primary channel, end users install with zero setup.
2. **GitHub Packages** — mirror, appears at
   https://github.com/Asher-1/ACloudViewer/packages.

**Publishing is fully automated** — no manual `npm publish`, no manual tag:
platform builds trigger version alignment, `npm/**` source changes trigger a
patch bump, and explicit tags/dispatches remain available. The npm package
version is **metadata only** — the bootstrap always installs whatever assets
are current on the `main-devel` release.

---

## One-time setup (per npmjs account, cannot be automated)

### Step 1 — npmjs.org account

1. Register / log in at https://www.npmjs.com with 2FA enabled (npm requires
   2FA on all publishers; you will be forced to enable it before publishing).
   Registration is the website form only (username / email / password +
   email verification) — there is no GitHub SSO signup. **npm usernames are
   lowercase**, so register as `asher-1` (exactly matching the `@asher-1`
   scope); GitHub's `Asher-1` display case does not carry over.
2. **You do NOT need to create an organization.** The package name
   `@asher-1/acloudviewer` is a *scoped* package, and a plain personal
   account whose **username** is `asher-1` can publish `@asher-1/*` public
   packages directly — skip https://www.npmjs.com/org/create entirely.
   Only if your npmjs username differs from `asher-1` and you cannot rename
   it, create an org named `asher-1` (npm scopes are lowercase) at
   https://www.npmjs.com/org/create.
3. Verify the scope is yours: https://www.npmjs.com/search?q=%40asher-1
   (a scope owned by someone else means publishing will 403).

#### If signup shows "Too many requests" / 操作频繁

npmjs rate-limits by **network exit IP**, not by your identity. Fix from most
to least likely:

1. **Change the exit IP.** Corporate networks, VPNs and shared proxy nodes
   are the usual trigger. Switch to a direct home broadband connection, or a
   different VPN region, and retry.
2. **Reset the browser context.** Use a private/incognito window, clear
   cookies for npmjs.com, and disable script/ad blockers for the domain —
   blocking extensions routinely break the Cloudflare verification and look
   like abuse to the rate limiter.
3. **Wait out the window.** The limit typically lifts within an hour;
   repeated attempts while throttled extend it (up to ~24h). Do not retry
   every few minutes.
4. **Complete the account fully on the first success**: verify the
   confirmation email, enable 2FA (authenticator app), and confirm your
   username — a half-created account still counts against the IP limit.

Fallback while the npmjs account is blocked: the **GitHub Packages mirror
publishes on its own** with zero external setup (the workflow's npmjs step
skips itself when `NPM_TOKEN` is absent). Users can install from the mirror
via the PAT instructions in README "Channel 2"; switch them to
`npm install -g @asher-1/acloudviewer` once npmjs goes live.

### Step 2 — Create the publish token (granular, bypass 2FA)

npm removed classic-token creation under the 2025 security policy (classic
tokens are fully revoked on 2026-12-09), so the **granular** page is the only
path. On npmjs.com → avatar → **Access Tokens** → **Generate New Token**:

1. **Bypass two-factor authentication** — **CHECK THIS BOX** (unchecked by
   default). Without it every `npm publish` demands an OTP no CI can enter —
   this exact mistake produced the first run's `EOTP` failure.
2. **Packages and scopes → Permissions**: select
   **Read and write (publish and stage)**.
   Do NOT pick `Read-only` (publish is a write) or
   `Read and write (stage only)` — staged versions wait for a human
   maintainer to promote them, so unattended CI can never publish.
3. **Select Packages**: `All Packages` (a fresh account holds only this
   package; restrict to the `@asher-1` scope later once it exists).
4. **Expiration**: the policy maximum is 90 days — tokens now always expire
   and must be renewed (see *Token renewal* below). Copy the token
   immediately (shown once, starts with `npm_`).

> A long-term no-token alternative is npm **Trusted Publishing** (OIDC). Now
> that the package exists, it can be configured: package page → **Settings →
> Trusted publishing → Add trusted publisher** → GitHub Actions →
> Organization/user `Asher-1`, repository `ACloudViewer`, workflow filename
> `npm-publish.yml`, and **allow `npm publish`** (post-Sep-2026 configs default
> to stage-only). The workflow already carries `id-token: write` and runs npm
> >= 11.5.1 (Node 24), so the npmjs step then authenticates via short-lived
> OIDC and falls back to `NPM_TOKEN` automatically. Caveats (as of
> 2026-11): the configuration must complete its first successful publish
> within 2 days, and scoped-package OIDC publishing has a history of E404
> bugs (npm/cli#8976) — if the OIDC path misbehaves the token fallback keeps
> the channel alive, so keep renewing `NPM_TOKEN` until an OIDC publish is
> observed in the logs ("Signed provenance statement" proves OIDC).

### Step 3 — Add the secret to GitHub

1. GitHub repo `Asher-1/ACloudViewer` → **Settings** → **Secrets and
   variables** → **Actions** → **New repository secret**.
2. Name: `NPM_TOKEN`, Value: the token from Step 2.
3. No secret is needed for the GitHub Packages mirror — the workflow uses the
   built-in `GITHUB_TOKEN` with `packages: write`.

Without `NPM_TOKEN` the workflow still succeeds and publishes only the
mirror; the npmjs step is skipped and marked as such in the run summary.

### Token renewal (granular tokens expire every 90 days)

npm's security policy caps token lifetime; the publish token **will** expire
and the npmjs channel stops until it is renewed (the GitHub Packages mirror
keeps working — it uses `GITHUB_TOKEN`). Symptoms and the renewal routine:

- **Symptom in the run**: the *Publish to npmjs.org* step fails with
  `ENEEDAUTH`, `401 Unauthorized`, or (on partially-scoped tokens) `EOTP`.
  The version resolution step still passes — `npm view` runs anonymously.
- **Renewal** (about once per 90 days): generate a fresh granular token
  (same four settings as Step 2) → update the `NPM_TOKEN` secret value →
  **Re-run failed jobs** on the failed run, or push/dispatch any trigger.
  The align/patch-bump modes re-resolve against the registry, so re-running
  after any gap is always safe and never re-publishes an existing version.
- **Renewal rhythm**: set a recurring calendar reminder; aligning renewal
  with the start of each odd month works well for a 90-day token.

---

## Automated release (default path, no manual action)

Once `NPM_TOKEN` is in place, publishing happens on its own. One workflow,
four triggers — each with its own version strategy:

| Trigger | Mode | Version behaviour |
|---------|------|-------------------|
| A platform build finishes (`workflow_run` on Windows/MacOS/Ubuntu/Ubuntu CUDA/Ubuntu Wheel — the same uploaders that feed `main-devel`) | **align** | Read the app version from the current `main-devel` assets (e.g. `3.9.5`). If npmjs `latest` already equals it → **skip** (idempotent: the five completion events converge to one publish). Otherwise publish `@asher-1/acloudviewer@<app version>` — npm versions track app releases automatically. |
| A push to `main` touching `npm/**` or the workflow itself | **patch-bump** | The bootstrap source changed; bump a fresh patch strictly above npmjs `latest` and publish. |
| Tag `npm-vX.Y.Z` pushed | **tag** | Strict check: tag must equal `npm/package.json` version; publish that exact version. |
| Manual **Run workflow** with a version input | **dispatch** | Force the given version and publish. |

What you actually do in the steady state:

- **App release → nothing.** The next platform build completion publishes
  `npm@<app-version>` automatically.
- **Bootstrap change → nothing.** Pushing `npm/**` changes bumps and publishes
  a patch automatically.
- All outcomes are visible in the run's step summary (mode, skipped/published
  per channel).

Note the git-side version drift: the **patch-bump** and **dispatch** modes
apply `npm version` inside the CI workspace without committing back, so
`npm/package.json` on `main` can lag the registry by a few patches. The
align/tag modes treat `npm/package.json` as the contract; the resolve step
reconciles against the registry (`sort -V` comparison), so drift never
publishes a stale or duplicate version.

---

## Manual release paths (tag npm-v3.9.5 worked example)

Two manual paths exist: a **tag push** (strict version contract, fires the
same workflow via `push`) and a **workflow_dispatch** (no commit needed).
Both assume `NPM_TOKEN` is already configured in repo secrets.

### Pre-flight checks (run all before touching git or GitHub)

```bash
# 1. Local version matches what you intend to publish
node -p "require('./npm/package.json').version"        # -> 3.9.5

# 2. The version is free on npmjs (anything other than E404 output means taken)
npm view @asher-1/acloudviewer version                  # 404 -> free; 3.9.5 -> taken
#    (also inspect what exists: https://www.npmjs.com/package/@asher-1/acloudviewer)

# 3. NPM_TOKEN secret exists on the repo (CLI or web):
gh secret list -R Asher-1/ACloudViewer                  # must list NPM_TOKEN
#    web: Settings -> Secrets and variables -> Actions

# 4. Working tree clean and up to date (tag mode publishes the code AT the tag)
git -C /path/to/ACloudViewer status --short
git -C /path/to/ACloudViewer fetch origin && git -C /path/to/ACloudViewer status | head -2

# 5. The workflow file exists on the branch you will tag (tag runs use the
#    workflow file from the TAGGED commit, not from main HEAD)
git show origin/main:.github/workflows/npm-publish.yml >/dev/null && echo "workflow present"
```

### Path A — tag push (CLI, version contract)

```bash
# If package.json is not yet at the target version, bump and commit first:
node -p "require('./npm/package.json').version"          # is it 3.9.5?
# if not:
#   (cd npm && npm version 3.9.5 --no-git-tag-version)
#   git add npm/package.json && git commit -m "npm: bootstrap 3.9.5" && git push origin main

# Tag and push — the tag MUST be npm-vX.Y.Z and equal package.json
git tag npm-v3.9.5
git push origin npm-v3.9.5

# Watch:
gh run watch -R Asher-1/ACloudViewer "$(gh run list -R Asher-1/ACloudViewer -w 'npm publish' -L 1 --json databaseId -q '.[0].databaseId')"
```

The workflow refuses to run if `npm-v3.9.5` != `npm/package.json` version
(resolve step exits with `tag npm-vX.Y.Z does not match`) — that check is the
contract, not a bug.

### Path B — workflow_dispatch via gh CLI (no commit needed)

```bash
gh workflow run npm-publish.yml -R Asher-1/ACloudViewer -f version=3.9.5
sleep 5 && gh run watch -R Asher-1/ACloudViewer "$(gh run list -R Asher-1/ACloudViewer -w 'npm publish' -L 1 --json databaseId -q '.[0].databaseId')"
```

The dispatch mode applies the version with `npm version` inside the CI
workspace and publishes exactly it. It does **not** commit the bump — sync
`npm/package.json` in a follow-up commit if you want `main` to match.

### Path C — workflow_dispatch via GitHub UI

1. Open https://github.com/Asher-1/ACloudViewer/actions/workflows/npm-publish.yml
2. **Run workflow** → branch: `main` → input `version`: `3.9.5` → **Run workflow**.
3. Watch the run page; the step summary table at the end shows per-channel
   status.

### Verify the publish

```bash
npm view @asher-1/acloudviewer version          # -> 3.9.5 (npmjs, may lag ~1 min)
npm view @asher-1/acloudviewer --registry https://npm.pkg.github.com version 2>/dev/null \
  || echo "mirror check needs a PAT (see README Channel 2)"
```

Then run the post-publish checklist below.

### If the manual run fails

| Symptom | Where | Meaning / fix |
|---------|-------|---------------|
| `409 Cannot publish over previously published version` | npmjs step | Version already exists there. Tag mode: retag (see below). Dispatch: run again with a different version. |
| `403 Forbidden` | npmjs step | Token lacks write on `@asher-1`, or the scope belongs to someone else. Recreate the token. |
| `npm registry query failed (not a 404 ...)` | resolve step | npmjs was unreachable from the runner — transient; re-run the job. (The workflow deliberately refuses to guess here: a wrong 0.0.0 fallback would end in a guaranteed 409.) |
| `tag npm-v3.9.5 does not match npm/package.json version` | resolve step | Version drift. Fix `package.json`, commit, delete and re-create the tag (`git tag -d npm-v3.9.5 && git push origin :refs/tags/npm-v3.9.5`, then re-tag). |
| `422 Unprocessable Entity` | mirror step | Scope/repo-owner mismatch — package must stay `@asher-1` on `Asher-1/ACloudViewer`. |
| Run did not start at all | — | Tag push: the tag's commit must contain `.github/workflows/npm-publish.yml` (pre-flight check 5). Dispatch: the file must exist on the selected branch. |

### Watch the run

Actions → npm publish → the run shows a mode resolution step, up to two
publish steps, and a step summary table (mode / skipped / per-channel status
derived from actual step outcomes). All-or-nothing is not guaranteed: if the
npmjs step fails, the mirror step still runs. Re-publishing a version that
already exists in a registry is intentionally rejected (409) — the **align**
and **patch-bump** modes resolve against the registry first, so they never
409; only the explicit **tag**/**dispatch** paths can hit it, which means the
version was already taken and you should pick another.

---

## Post-publish verification checklist

```bash
# 1. Registry pages (should show the new version)
#    https://www.npmjs.com/package/@asher-1/acloudviewer
#    https://github.com/Asher-1/ACloudViewer/packages

# 2. Primary-channel install in a scratch dir (no auth, no .npmrc)
cd "$(mktemp -d)" && npm init -y >/dev/null
npm install -g @asher-1/acloudviewer
acloudviewer version                 # must print the version you published
acloudviewer list                    # must list current main-devel assets
acloudviewer install --dry-run       # must match an asset for your platform

# 3. Mirror-channel install (PAT with read:packages required)
npm config set @asher-1:registry https://npm.pkg.github.com/
npm config set //npm.pkg.github.com/:_authToken=<YOUR_PAT>
npm uninstall -g @asher-1/acloudviewer
npm install -g @asher-1/acloudviewer
acloudviewer version

# 4. Real install (Linux) — the download line should read "4 parallel segments"
acloudviewer install --silent --prefix ~/.local/share/ACloudViewer
ls ~/.local/share/ACloudViewer            # ACloudViewer binary sits at the target root (verified QtIFW layout)

# 5. Uninstall via the CLI (wraps QtIFW maintenancetool purge, verified working)
acloudviewer uninstall --yes              # refuses without --yes in non-interactive shells
# equivalent manual form:
#   ~/.local/share/ACloudViewer/maintenancetool purge --confirm-command

# 6. Restore the registry if you ran the mirror test in step 3
npm config delete @asher-1:registry
npm config delete //npm.pkg.github.com/:_authToken
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| npmjs signup: `Too many requests` / 操作频繁 | Shared exit IP (VPN/office network) throttled by npmjs | See Step 1 "If signup shows Too many requests": switch exit IP, reset browser context, wait out the window; meanwhile the mirror publishes on its own |
| npmjs step: `EOTP` / "requires a one-time password" | Token is a classic **Publish** token — 2FA-protected tokens demand an OTP on publish, which CI cannot provide (verified first run) | Regenerate the token as classic **Automation** type (or a granular token with `@asher-1` package write) and update the `NPM_TOKEN` secret; then re-run the failed job |
| npmjs step: `403 Forbidden` | Token lacks write on the scope, or scope owned by someone else | Recreate token with package write for `@asher-1`; verify scope ownership on npmjs |
| npmjs step: `409 Cannot publish over previously published version` | Explicit tag/dispatch reused a version that exists | Bump the version; align/patch-bump modes never hit this (they resolve against the registry first) |
| GitHub step: `422 Unprocessable Entity` | Package scope does not match the repo owner | Scope must be `@asher-1` and the repo owner `Asher-1` — do not rename the package |
| Tag run: `tag npm-vX.Y.Z does not match package.json` | Tag/version drift | Re-tag or fix `npm/package.json` — the check is the contract |
| Align run: `skipped (version aligned)` but you expected a publish | npmjs `latest` already equals the app version carried by `main-devel` assets | Expected behaviour — publish again only after the app version bumps, or use tag/dispatch for an out-of-band bootstrap fix |
| `workflow_run` trigger did not fire | Uploader workflow renamed, or the run belongs to a non-default branch | Workflow names in `on.workflow_run.workflows` must match exactly (Windows/MacOS/Ubuntu/Ubuntu CUDA/Ubuntu Wheel); use dispatch to publish manually in the meantime |
| `install` prints `segment N/M: retry ... stalled (60s without data)` | Connection stall mid-transfer | Expected resilience: idle streams abort after 60s and retries resume via HTTP `Range` from the bytes on disk; persistent failures mean a network outage — retry later or use `--parallel 1` |
| `install` prints `server does not support range requests; single-stream download` | Mirror/proxy stripped Range support | Informational fallback, download still works (slower) |
| Mirror install: `404 Not Found` | `.npmrc` missing scope mapping or PAT lacks `read:packages` | Follow README "Channel 2" steps; regenerate PAT with the scope |
| Mirror install: `401 Unauthorized` | Stale/expired PAT in `//npm.pkg.github.com/:_authToken` | `npm config set //npm.pkg.github.com/:_authToken=<NEW_PAT>` |

## Sync contract

Asset-matching rules in `npm/bin/acloudviewer.js` mirror
`cli_anything/acloudviewer/utils/installer.py` (find_matching_app /
find_matching_wheel / QtIFW console invocations). The download engine
(ranged-segment parallelism, 60s stall abort, Range resume) and the
`uninstall` command are **bootstrap-only capabilities** — installer.py has
neither; do not "re-sync" them away, and re-verify them on Linux whenever
the engine changes. If release asset names or the QtIFW console syntax
change, update **both** files; the JS file header notes this contract.
Changes to the silent path should be validated on Linux
(`--silent --prefix <dir>`) — Windows/macOS console syntax follows
installer.py's production-verified invocations.
