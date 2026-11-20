---
name: deepseek-harness-deploy
description: Deploy and operate DeepSeek Harness (DSH) local clients — web UI and dsh-TUI — on Linux dev machines against the deeproute token-wormhole gateway. Use when installing or upgrading dsh, booting the web UI or terminal client, configuring model providers and API keys (cordis.patch.yml / credentials store), adding international models (GPT/Claude/Grok) with reasoning-effort levels, installing profile plugins, or troubleshooting DSH startup, plugin, and credential issues. Also bootstraps a full remote dsh web over SSH with ZERO human interaction (VSCode Remote-SSH semantics — agent native execution context on the remote host, GPU passthrough, reboot self-heal) via scripts/bootstrap-remote.sh, including on managed platforms with old glibc and forced seccomp.
---

# DeepSeek Harness (DSH) Local Deployment & Operation

Verified recipe (2026-11, dsh 0.2.2-rc.5 / dsh-tui 0.20.2) for internal Linux
dev machines with access to `token-wormhole.deeproute.cn`. Consolidates the
colleague quick-start doc (`~/Downloads/dsh-quick-start-for-colleague.md` if
present) plus hands-on findings the doc does not cover.

## Sources of Truth

| Topic | Reference |
|---|---|
| Colleague quick start (web profile, gateway, plugins) | `~/Downloads/dsh-quick-start-for-colleague.md` (this machine) |
| Launcher/profile/CLI contract | `~/.npm-global/lib/node_modules/@deepseek-ai/dsh/README.zh.md` |
| User guide (providers, reasoning levels) | `.../@deepseek-ai/dsh/docs/user/guide/providers.md` |
| Credential seam | `.../@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-credentials-local/src/index.ts` |
| Troubleshooting table | [references/troubleshooting.md](references/troubleshooting.md) |
| Gateway models, dual keys, reasoning levels | [references/models-and-reasoning.md](references/models-and-reasoning.md) |
| Remote zero-human bootstrap (Path A): decision tree, platform-seccomp boundary, permission mode, GPU bind, self-heal | [references/remote-bootstrap.md](references/remote-bootstrap.md) |
| One-click remote scripts (bootstrap / verify / local-forward) | `scripts/bootstrap-remote.sh`, `scripts/verify-remote.sh`, `scripts/local-forward.sh` |

## Three Principles (from the colleague doc)

1. **Never install with sudo** — it breaks npx cache ownership and forces
   `sudo node $(which dsh)` afterwards. npm global prefix is user-owned
   (`~/.npm-global`); nothing needs root.
2. **Node >= 24.9** — dsh `engines.node >= 24.9.0` (any newer version passes;
   no nvm needed just to match a specific 24.x).
3. **npm registry → npmmirror** — `npm config set registry
   https://registry.npmmirror.com`. Caveat: the mirror's `latest` tag can lag
   npmjs.org; pin exact versions when a peer check rejects (see troubleshooting).

## Client Form Routing (pick before deploying)

| Platform | Local client (non-web) | Web |
|---|---|---|
| Linux | **dsh-TUI** only — the official Electron desktop and the third-party DSH Desktop (dataelement) both have no Linux builds | `dsh web` |
| Windows / macOS | official desktop app | `dsh web` |

Both forms share one `$DSH_HOME` (`~/.dsh`): credentials, sessions, and the
config layering. Profiles are independent (`~/.dsh/profiles/<name>/`).

## Phase A — Environment (one-time)

```bash
node -v        # >= 24.9.0 (engines)
pnpm --version # >= 11 (dsh-TUI requirement); 11.x works, upgrading optional
npm config set registry https://registry.npmmirror.com
```

Gateway key: apply for **your own** key (never reuse someone else's; see the
doc's red lines) and supply it via the credentials store or an rc-file export
(Phase C). Put the export in `~/.bashrc` on bash shells, `~/.zshrc` on zsh.

## Phase B — Install & Boot

```bash
npm install -g @deepseek-ai/dsh @deepseek-harness-tui/dsh-tui   # web+CLI+TUI launcher
dsh --version          # e.g. 0.2.2-rc.5

# Web (startup dir = default workspace root)
cd <workspace> && dsh web --no-open
# → copy the FULL printed URL incl. ?token=... (regenerated every boot)

# Terminal client (profile name is `dsh-tui`, not `tui`)
dsh plugin --profile dsh-tui add @deepseek-harness-tui/dsh-tui@<exact-version>
dsh-tui                # full-screen TUI; `dsh-tui --resume` continues a session
```

Pin the exact dsh-tui version whose `peerDependencies` covers your `dsh
--version` (check on registry.npmjs.org; npmmirror `latest` may lag behind).
Version mismatch → dsh rejects with `incompatible-version` and auto-restores
package.json/node_modules (safe to retry).

## Updating Safely

`~/.local/bin/dsh-safe-update` (check mode by default; `--apply` to upgrade):
backs up profile `package.json` / `cordis.patch.yml` / `pnpm-workspace.yaml`
plus skin & wallpaper configs to `~/.dsh-backup/<ts>/`, upgrades the dsh core
and web plugins, verifies both profiles still compose (`--dump-config`), and
rolls back the core + profile files on failure. Restart `dsh web` after apply
(bundle changes need a restart); dsh-TUI self-updates via its `/update`
command. Version truth comes from registry.npmjs.org because npmmirror
`latest` can lag — and a just-published version can 404 on its tarball (re-run
later). To pull a specific version ahead of the mirror:
`dsh plugin --profile web add <pkg>@<exact> --registry=https://registry.npmjs.org`.

## Phase C — Model Gateway Config

Edit `~/.dsh/profiles/<profile>/cordis.patch.yml` (top-level YAML **array**,
entries located by `id`, spaces not tabs):

```yaml
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      deeproute-gateway:                    # OpenAI-protocol models
        apiKeyEnv: DEEPROUTE_GATEWAY_API_KEY
        api: openai-responses
        baseURL: https://token-wormhole.deeproute.cn
        models: [ ... ]                     # id/name pairs; contextWindow & reasoningEfforts optional
      deeproute-claude:                     # Claude via the Anthropic entry (optional, see reference)
        apiKeyEnv: DEEPROUTE_ANTHROPIC_API_KEY
        api: anthropic-messages
        baseURL: https://token-wormhole.deeproute.cn
        models: [ ... ]
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: deeproute-gateway
    model: deepseek-v4.1-flash
```

Protocol choices: `openai-responses`, `openai-completions`,
`anthropic-messages`, `azure-openai-responses`, `openai-codex-responses`.

Keys never go into the patch. Put them in the credentials store
`~/.dsh/.credentials.yaml` (v1 layout: `version: 1` + `refs: {ENV_NAME: value}`
+ `records:`; keep `records` entries intact) — chmod 600, enforced at startup.
Resolution order: store > `<cwd>/.env` > `$DSH_HOME/.env` > process env; the
file is watched and hot-reloads. An env-var `export` in the shell rc also works
but depends on the server process inheriting it — prefer the store.

The web UI Settings→Models page writes this same patch file; never edit both
sides at once. `dsh --profile <name> --dump-config` (exit 0) validates YAML and
shows the composed tree.

### Rotating a gateway key — align the model list or everything 404s

The gateway authorizes **per-key model groups**. A new key's group can be
completely different from the old one (2026-11-19 real case: old key =
DeepSeek/Qwen/Kimi/GLM/GPT group; new key = 58 models, **all Gemini**). dsh
does NOT read `~/.codex/auth.json` or `~/.claude/settings.json` — those are
Codex CLI / Claude Code stores, unrelated to dsh. The web UI key editor
**writes the credentials store directly** (verify via store mtime; the green
hint "使用已保存的 API key" means "empty input keeps the stored value" — an
empty save is a no-op, but entering a value does persist). So after a key
rotation the auth succeeds but every configured model 404s, which looks like
"the key didn't take effect".

Rotation checklist:
1. Probe the new key's group: `curl -H "Authorization: Bearer <key>"
   https://token-wormhole.deeproute.cn/v1/models` — the returned ids ARE the
   authorized set; pick models only from it.
2. Rewrite the provider's `models:` list + `agent-default-model.model` to
   group members (patch hot-reloads the model catalog; no restart needed).
3. Validate with `dsh --profile web --dump-config` (exit 0) — NOT python
   yaml.safe_load, which silently accepts duplicated mapping keys that dsh's
   parser rejects; and when replacing a block anchored on `models:` do not
   repeat the `models:` line inside the replacement text.
4. Smoke-test one model via `/v1/responses` (200) before handing back.
5. A 502 `upstream_error` on `/v1/messages` is a gateway-side upstream
   outage (auth already passed) — not a local config problem; retry later.
6. **Model availability is decided solely by the key's group** — the same
   gateway key returns 404 `model_not_found` on any model outside its group,
   on every path (`/responses`, `/v1/responses`; a missing gateway route like
   `/codex/responses` falls back to the SPA HTML page with HTTP 200 — check
   the body, not just the status). The `openai-codex-responses` protocol is
   NOT a gateway path: its default base is `https://chatgpt.com/backend-api`
   (ChatGPT OAuth backend), unrelated to token-wormhole. When a user reports
   "openai/gpt models disappeared" after a key rotation, the new key simply
   has no GPT models in its group — no dsh config can fix that; ask the
   gateway admin for a GPT-authorized key, then re-add the gpt entries to
   the provider's models list and set `agent-default-model` back.
7. **Every model entry MUST declare `reasoningEfforts`** (off/low/medium/high,
   GPT families also xhigh) — the Effort menu is generated per-model from
   this declaration. Entries written as bare `{id, name}` compose fine
   (dump-config passes) but silently drop the thinking-effort selector in
   the model picker; add the block back and the menu hot-reloads.

## Phase D — Web Profile Plugins (optional, doc-recommended)

```bash
dsh plugin --profile web add @linxin666/dsh-web-all@latest   # 19 sub-plugins
dsh plugin --profile web add github:AiLi1337/dsh-context-length
dsh plugin --profile web add dsh-plugin-wallpaper-engine
```

- First add may fail with `ERR_PNPM_IGNORED_BUILDS`: pnpm 11 writes an
  `allowBuilds:` scaffold into the profile's `pnpm-workspace.yaml` — replace
  each `set this to true or false` with `true`, then re-run the add. This is
  the non-interactive equivalent of the doc's `approve-builds`.
- If the first add was interrupted, the dependency may exist while the bundle
  is NOT registered in `package.json` `dsh.profile.bundles` → fix with
  `dsh plugin --profile web remove <pkg>` then re-add.
- Native builds (ssh2/cpu-features) may fail on very new Node — optional deps,
  pure-JS fallback, ignore.
- Peer-dependency warnings are ignorable; doc fallback
  `PNPM_PACKAGE_IMPORT_METHOD=copy dsh plugin ...` for import failures.

## Phase E — Verify

```bash
dsh --profile web --dump-config          # exit 0 = YAML + schema compose OK
dsh web --no-open --port 0 &             # temp boot on a free port; then pkill
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3080/          # 401 expected
curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:3080/?token=<token>"  # 303/200
python3 - <<'EOF'                        # gateway reachability + entitlement
import json, os, urllib.request
key = os.environ.get("DEEPROUTE_GATEWAY_API_KEY", "")
req = urllib.request.Request("https://token-wormhole.deeproute.cn/v1/models",
                             headers={"Authorization": f"Bearer {key}"})
print(urllib.request.urlopen(req, timeout=15).status)   # 200
EOF
```

In the UI: pick the workspace once (per profile), then chat; Effort menu and
model catalog come from the patch (hot-reloaded via dsh-hmr — a running server
picks up patch edits without restart).

## Pets & Ambient Status

Two web-profile pet renderers coexist and can look duplicated:
- `dsh-plugin-whale-pet` (Yifffan) — an **in-window** floating pet (self-declared
  "not a separate operating-system desktop overlay"); toggled by the `web-ui-pet`
  patch row (`visible: false` hides it) or the plugin manager.
- `dsh-claude-style` ships its own mascot **Deepy** on the composer edge
  (8 states incl. sleeping) — part of the theme; no documented off-switch.

For out-of-browser task status install `dsh-desktop-pet` (sereinmono): a normal
dsh bundle whose host half opens an **independent transparent overlay window**
(Neutralinojs runtime; idle/thinking/tools-running/awaiting/finished states are
driven deterministically by session events, zero added LLM cost). Linux
(X11/Wayland) needs `libwebkit2gtk` and a running compositor (GNOME/KDE ship
both; Ubuntu 20.04 has libwebkit2gtk-4.0). User pets drop into
`~/.dsh/desktop-pet/pets/` (runtime-discovered, no rebuild).

**The bundled pet is a placeholder.** `assets/pets/text` renders a green label
card ("idle / frame 5/6") for binding verification — seeing a green square
means the pet picker is still on the test pet, NOT a rendering failure. Real
pets are `pet.json` + `spritesheet.webp` folders (1536×1872, 8×9 grid of
192×208 transparent cells; rows fixed in order: idle 6, running-right 8,
running-left 8, waving 4, jumping 5, failed 8, waiting 6, running 6, review 6).
Community pets: `npx petdex list` / `npx petdex install <slug>` (thousands of
pets; installs into `~/.petdex/pets`), then copy the folders into
`~/.dsh/desktop-pet/pets/`. Select the active pet with a user patch row
`- id: desktop-pet, config: { ...all keys..., petId: <directory-name> }` —
the user-layer config **replaces** the bundle config, so copy every key from
the bundle's own `cordis.patch.yml`. Patch hot-reload switches the pet live.

Web in-window pets: hide `dsh-plugin-whale-pet` via `visible: false` on the
`web-ui-pet` patch row (or its UI card's 隐藏 button — the UI writes the patch
back); `dsh-claude-style`'s Deepy has no documented off-switch.

**Fully removing the web in-window pets** (2026-11-19): the dsh-pet settings
card writes its WHOLE config back to the patch on any interaction (visible
flips back to `true`, petId changes) — hiding via the card is NOT durable.
Definitive removal: `dsh plugin --profile web remove dsh-plugin-whale-pet`
+ delete the `web-ui-pet` patch row (uninstalled bundle can never come back).
Deepy (claude-style) is hidden via its DOM classes
(`.deepy-anchor/.deepy-strip/.deepy-sprite/.deepy-hit`) using `dsh-custom-css`:
it watches plain `.css` files in `$DSH_HOME/custom-css/` (hot reload,
`custom.css` = the always-on sheet, applied before the shell paints).

**Reopening a closed pet window**: closing the Neutralino window (X button)
exits the process and persists `~/.dsh/desktop-pet/position.json`, but nothing
respawns it — patch hot-reload does NOT reopen a closed window (verified:
a real petId/animationSpeed diff does not respawn). Recover by restarting
`dsh web` (settings `enabled` stays `true` in the patch). Community pets also
via `npx petdex install <slug>`; the official FEATURED gallery entry (e.g.
`boba`, highest downloads) is the safe default pick.

Pet state machine (deterministic, zero LLM cost): `STARTING IDLE THINKING
WORKING CODING RUNNING_COMMAND WAITING_FOR_USER SUCCESS ERROR SLEEPING` — IDLE
= no task, THINKING = model reasoning, WORKING/CODING/RUNNING_COMMAND = tools
running, WAITING_FOR_USER = awaiting your input/approval, SUCCESS/ERROR = turn
finished (celebrate/frown), SLEEPING = long inactivity (`hideWhenIdle: true`
auto-hides). In a session type `/pet <state>` (e.g. `/pet thinking`) to
preview any state without an LLM. **Clicking the pet always opens a NEW
browser tab** at the WebUI URL (xdg-open; no tab-focus, no setting to change
it) — don't click when a tab is already open.

**Multi-instance policy**: never run two instances of the SAME profile
(task-board ledger lock contention). One instance serves both local and
remote development (Path B SSH tools switch hosts per session). If isolation
is really needed, use a second PROFILE (`dsh web --profile dev2 --port 0`) —
separate state/sessions/pets at the cost of duplicated config and memory.

**Size & DeepSeek-mascot pets**: `petScale` (step 0.25, **min 0.5**, max 4)
adjusts size via the user patch row — a value below 0.5 fails the row's
schema validation (`ValidationError: invalid config` at startup) and the pet
window silently never opens. A live patch diff also kills the running window
without respawn, so restart `dsh web` after petScale/petId changes. Petdex
hosts a DeepSeek-mascot family: `aoi-shark-maid` (DeepSeek酱 maid form,
POPULAR/LOVED on petdex.dev), `deepseek-goldwhale-v1` (绣金鲸 refined whale),
`deepseek-2`, `deepseek-drool`, `spritesheet-64` (Whale-Chan). Installed
2026-11-19: 9 pets in `~/.dsh/desktop-pet/pets/`, active =
`deepseek-goldwhale-v1` @ petScale 0.5.

**State-distinguishability is measurable** (PIL): open each
`spritesheet.webp`, slice the 9 state rows, mean pairwise pixel distance of
row centers — higher = states visually distinct at small sizes. Measured
ranking (0–255): goldwhale-v1 57.7 > boba 54.0 > aoi-shark-maid 52.2 >
mecha-xiaobai 47.9 > shuangsheng-linger 45.3 > golden-empress 44.7 >
dream-girl 30.0 > lulu-capybara 24.0 > jill-stingray 21.1. Portrait/character
pets (立绘类) score low — prefer action-pose pets when the window is small.

**Idle-pose look-alike trap**: `deepseek-goldwhale-v1` (绣金鲸) and
`aoi-shark-maid` (DeepSeek酱) are both blue-haired maid-form DeepSeek girls;
their IDLE poses are nearly identical at petScale 0.5, so "it still looks like
the maid" does NOT mean a petId switch failed. Distinguish via the action rows
(run/celebrate/frown) or verify with the cache evidence below.

**Verify which pet a window actually renders** (authoritative, no guessing):
the Neutralino WebView disk cache records the spritesheet URL actually fetched
— `grep -rl <petId> ~/.cache/neutralino-linux_x64/WebKitCache/` with per-file
`stat -c %y` mtimes; the URL written AFTER the current `dsh web` start time is
what is on screen. Secondary: `dsh --profile web --dump-config` shows the
composed `desktop-pet` config (proves the patch applied), and
`~/.dsh/desktop-pet/position.json` gives the window position for PIL-crop
screenshot comparison against the spritesheet's first idle cell.

Install pitfall: the first add blocked by `ERR_PNPM_IGNORED_BUILDS` leaves
`dsh-desktop-pet: set this to true or false` in the profile's
`pnpm-workspace.yaml` `allowBuilds:` — replace the placeholder with `true`,
re-add, and if the bundle is still missing from `package.json`
`dsh.profile.bundles`, `remove` + `add` (dependency present, bundle
unregistered).

## Remote Development over SSH

Two orthogonal paths; both keep the web UI in the local browser:

- **Path B — local dsh + SSH tool plugin (recommended; remote needs nothing but
  sshd).** `dsh-ssh-remote` (kkutysllb/dsh-kylin-ssh-tunnel) routes
  run/read/write/edit/glob/grep/push/pull through SSH with a ControlMaster
  layer that consumes `~/.ssh/config` hosts as-is, plus a status panel and a
  settings-page host manager. The local `~/.dsh` (models, theme, wallpapers,
  pets) is reused unchanged.
- **Path A — full dsh web on the remote host.** Requires node >= 24.9,
  pnpm >= 11, gateway reachability, and ~1.5-2 GB free disk; then
  `ssh -L 3080:127.0.0.1:3080 <host>` and open the local forwarded URL (see
  Security). Probe a candidate host first:
  `ssh <host> 'node -v; pnpm -v; curl -s -o /dev/null -w "%{http_code}" https://token-wormhole.deeproute.cn/v1/models'`
  (401 = gateway reachable, no key). Only Path A survives local shutdown for
  long-running remote sessions.

Verified 2026-11-19: `sem2d-dev-code` (970 GB free) fits Path A;
`huoshan_8` has only ~4.5 GB free (Path A tight, Path B fine); both reach the
gateway.

### Path A zero-human bootstrap (scripts)

`scripts/bootstrap-remote.sh <ssh-alias>` converges a full remote dsh web with
**zero human steps** and reboot self-heal; `--audit-only` probes first.
Decision tree: glibc >= 2.28 → native user-space install; else docker →
ubuntu:24.04 container substrate; else hard fail. It transfers the user's own
gateway refs + model patch as raw text (no YAML round-trip), passes GPUs
through via a directory bind + `.so.1` symlinks (works without
nvidia-container-toolkit), probes the netns and adds a container `ssh -R` hop
only when the host loopback cannot reach dsh, and commits `dsh-base:latest`
with `CMD boot.sh` + `--restart unless-stopped` so a host reboot self-heals.
Re-runs are convergent no-ops (healthy + patched + CMD-aligned → no restart).

Zero-human permission: managed ML platforms **reject `--cap-add SYS_ADMIN`
and force-inject seccomp/apparmor** (probed on huoshan_8: `unshare` EPERM,
overrides ignored) — do not fight them. The default
`--permission-mode danger-full-access` makes dsh-bash-sandbox skip
confine()/bwrap (no userns needed) and sets approval `policy: never`, so no
per-session "full permissions" click remains; the container itself is the
isolation boundary (VSCode Remote-SSH trust model). `workspace-write` is
available where userns works. After any remote reboot run
`scripts/local-forward.sh <ssh-alias>` to reprint the fresh token URL (the
token regenerates every boot); `scripts/verify-remote.sh <ssh-alias>` gates
native context / GPU / zero-human permission / activate failures / forward.
Full evidence, security trade-off, and pitfall table:
[references/remote-bootstrap.md](references/remote-bootstrap.md).

Static hosts (restart to apply) via a user patch row `- id: ssh-remote,
config: {hosts: [...], commandTimeoutMs}` — fields per host: `id/name/host/
user/port/auth` (`agent` = defer to `~/.ssh/config` / agent / default keys) and
optional `jump` (references another host id) + `defaultCwd`. Dynamic hosts go
to `~/.dsh/ssh-remote/hosts.json` (hot-applied). Passwords only in
`credentials.enc` (AES-256-GCM), never in `hosts.json`. In Liangshen mode the
`ssh_*` tool family is paged away by default — call `tool_activate` first.

## Liangshen Mode (minimal agent preset)

`@linxin666/dsh-liangshen` (bundled with web-all; also built into dsh-TUI as
an official agent preset). Purpose-built for **deepseek-v4.1-flash**, whose
post-training aligns with its native minimal tool surface (official harness
benchmarks: minimal native beats programmatic — DeepSWE v1.1 72.6% vs 67.6%,
Terminal-Bench 2.1 90.6% vs 85.8%). Mechanics: minimal persona system prompt;
wire collapses to a single `run_code` transport tool (`ptc` presentation, all
other tools invoked via the generated SDK; `native` is a one-line fallback);
high-fanout tool families (`mcp__*`, `browser_*`, `task_board_*`, `team_*`,
`squad*`, `ui_*`, `ssh_*`) are paged away behind one-line namespace summaries
until `tool_activate({namespace})` (max 3 active namespaces, LRU eviction);
a fact ledger (`fact_register`) fights long-session forgetting. **Use for**:
long deepseek-flash coding sessions (context/token savings). **Avoid for**:
sessions leaning on MCP/browser/task-board/SSH tools (must activate per
namespace) and for GPT/Claude models (the alignment argument is
DeepSeek-specific). Toggle: the slot-machine lever next to the model picker,
or the preset selector.

## Security (from the doc, non-negotiable)

- Keep the default 127.0.0.1 binding; remote access only via `ssh -L
  3080:127.0.0.1:3080 <host>`.
- Keys live only in `~/.dsh/.credentials.yaml` (600) or an rc-file `export`
  (600) — never in files that can reach git, never in shell history (leading
  space + `HIST_IGNORE_SPACE` for sensitive commands).
- The `?token=` URL grants full agent access; treat it like a password.

## Reference Files

- Failures and fixes: [references/troubleshooting.md](references/troubleshooting.md)
- Gateway entitlement, model probing, reasoning-effort declaration and default-level semantics:
  [references/models-and-reasoning.md](references/models-and-reasoning.md)
- Remote zero-human bootstrap (Path A): substrate decision tree, platform-seccomp boundary proof,
  danger-full-access permission model, GPU directory-bind recipe, reboot self-heal, pitfalls:
  [references/remote-bootstrap.md](references/remote-bootstrap.md)
