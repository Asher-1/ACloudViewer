# DSH Troubleshooting Quick Reference

Symptom → cause → fix, merged from the colleague quick-start doc (§5) and
hands-on findings (2026-11). Validate after any fix with
`dsh --profile <name> --dump-config`.

## Startup & Access

| Symptom | Cause | Fix |
|---|---|---|
| `EACCES: permission denied ... ~/.dsh/...` | sudo was used before; files owned by root | `sudo chown -R $(id -u):$(id -g) ~/.dsh` |
| `EACCES` under `~/.npm/_npx` | npx cache ownership | `sudo chown -R $(id -u):$(id -g) ~/.npm` |
| Browser 401 / blank page | URL missing `?token=` | use the FULL URL printed by the current boot; the token is regenerated on every start |
| Port 3080 busy | another dsh instance | `dsh web --port 0` (random free port) or `--port 8080` |
| `EADDRINUSE` on smoke-test boot | user's own instance already running | don't kill it; test on `--port 0` instead |
| Two instances: `task-board ledger is already owned by process N` / `EEXIST ledger-v2.lock` | second instance on same profile | expected contention, chat still works; keep one instance |
| Chat errors in UI | gateway not configured / key ref missing | Settings→Models: provider card key ref must be green; check credentials store/env |
| Image uploads fine but send fails: `pi-ai model "<id>" does not support image input` (model is plainly vision-capable) | the model entry has no `input:` declaration; pi-ai falls back to `defaultInput(["text"])` for ids not in its built-in catalog | add `input: [text, image]` to that model in the patch (or tick `image` in Settings→Models `inputModalities`); hot-reloads, no restart. Bulk audit command + ordering traps in [models-and-reasoning.md §5](models-and-reasoning.md) |
| Profile crashes at boot | `cordis.patch.yml` YAML error (tabs, bad indent) | `dsh --profile <name> --dump-config` names the line; spaces only |
| `EMFILE: too many open files, watch '~/.dsh/...'` + `N entries did not activate` (hmr, ssh-remote) | **inotify-instance limit**, not fd limit: per-user `fs.inotify.max_user_instances` (128) hit at boot instant by co-running IDE/browser/chat watchers (Qoder+trae+Chrome alone hold ~150); failed bundles never retry. (ENOSPC "file watchers reached" would instead mean `max_user_watches`.) | restart `dsh web` once the transient peak passes (verify: no `did not activate`, dsh proc holds 2 inotify fds); raise the limit: `sudo sysctl -w fs.inotify.max_user_instances=1024` + persist via `/etc/sysctl.d/99-inotify.conf` |
| Patch edits not applied | server started before edit | dsh-hmr hot-reloads patch files; if the UI still shows old state, restart `dsh web` |
| Image upload in the chat box always fails / Enter blocked; all API calls fail but the browser tab still shows the UI | the dsh server process is GONE (curl 3080 → 000). A dsh launched through an agent/IDE **background shell** is killed when that tool's observer lifetime expires (~24 h) — the SPA stays cached in the tab so it looks alive, but uploads/APIs have no backend | restart detached so it outlives the tool host: `setsid nohup <ABS-PATH>/dsh web --no-open --port 3080 > /tmp/dsh-web.log 2>&1 < /dev/null &` (use the ABSOLUTE dsh path — the detached child may not inherit the interactive PATH; verify `ppid=1` and curl 401). Re-open the freshly printed token URL |

## Plugins

| Symptom | Cause | Fix |
|---|---|---|
| `ERR_PNPM_IGNORED_BUILDS` (cloudflared/ssh2/cpu-features) | pnpm 11 requires declarative build approval | fill `allowBuilds:` in the profile's `pnpm-workspace.yaml` with `true` per package, re-run add |
| Re-add still rejected by `ERR_PNPM_IGNORED_BUILDS` | the failed first add left a placeholder row `pkg: set this to true or false` in `allowBuilds:` — a "package already listed" check skips it | replace the placeholder with `true` (not merely check the name exists), re-run add |
| Plugin dep installed but bundle missing from `dsh.profile.bundles` | first add was interrupted mid-flow | `dsh plugin --profile <p> remove <pkg>` then add again (bundle registers on a clean cycle) |
| `Issues with peer dependencies found` warning | informational | ignore unless functionality breaks |
| `incompatible-version` rejection | plugin peer range excludes current dsh | prefer exact plugin version whose peers cover `dsh --version` (npmmirror `latest` can lag npmjs); or `dsh plugin --profile <p> allow-version <pkg>@<ver> --dsh-version <v> --accept-risk` (revoke with `revoke-version`) |
| `ERR_PNPM_...` import failures | store/linking quirk | `PNPM_PACKAGE_IMPORT_METHOD=copy dsh plugin ... add <pkg>` |
| ssh2/cpu-features gyp build failure on new Node | optional native bindings | ignore; pure-JS fallback |
| Plugin installed but UI unchanged | bundle not enabled / server not restarted | check `dsh.profile.bundles`, restart DSH |

## Credentials

| Symptom | Cause | Fix |
|---|---|---|
| Provider key ref red dot in UI | referenced env var not visible to the server process | put the value in `~/.dsh/.credentials.yaml` `refs:` (hot-reloaded), not only an rc-file export opened before the server started |
| `credentials-local: ... readable beyond its owner` | file perms > 600 | `chmod 600 ~/.dsh/.credentials.yaml` |
| `pre-release flat layout` error on credentials file | old flat layout | add `version: 1` and nest entries under `refs:` |
| Claude models error `not available for this group` | key's gateway entitlement group lacks Claude | use the Claude-entitled key (e.g. the Claude Code token) via a separate provider + `apiKeyEnv`; never mix keys silently |

## Version & Environment

| Symptom | Cause | Fix |
|---|---|---|
| dsh refuses to start / engine error | Node < 24.9 | install newer Node (nvm or system); no need to match a specific 24.x |
| Third-party plugin won't resolve | registry not npmmirror | `npm config set registry https://registry.npmmirror.com` |
| Everything broken after bad edits | — | profile rebuild (below); don't hand-repair |

## Profile Rebuild (last resort)

```bash
# 1. stop DSH (Ctrl+C)
cp ~/.dsh/profiles/<p>/cordis.patch.yml ~/cordis.patch.yml.bak
mv ~/.dsh/profiles/<p> ~/.dsh/profiles/<p>.broken
# 2. re-init
cd <workspace> && dsh web --no-open     # Ctrl+C after URL prints
# 3. re-add llm-pi-ai + agent-default-model patches AND plugins — they do NOT come back automatically
```

## Remote Container (Path A zero-human bootstrap)

See [remote-bootstrap.md](remote-bootstrap.md) for the full recipe and probe
evidence. Scripts: `scripts/bootstrap-remote.sh` / `verify-remote.sh` /
`local-forward.sh`.

| Symptom | Cause | Fix |
|---|---|---|
| Per-session "full permissions" human click; `unshare --user` EPERM in container | managed platform force-injects seccomp denying `unshare(CLONE_NEWUSER)`; bwrap cannot confine | `--permission-mode danger-full-access` (skips bwrap) + approval `policy: never`; do NOT try to win userns |
| `docker run --cap-add SYS_ADMIN` → `Capability is rejected:SYS_ADMIN` | platform docker wrapper hard-rejects the cap | plain `docker run`; zero-human comes from the dsh permission mode, not container caps; never destroy/recreate a working container for caps |
| `--security-opt seccomp=unconfined`/`apparmor=unconfined` has no effect (inspect still shows platform policy, `Seccomp: 2`) | overrides silently ignored / force-reinjected | stop fighting; use danger-full-access (boundary proof in remote-bootstrap.md) |
| `nvidia-smi` missing in agent session / terminal tab | driver userspace not passed through | directory bind `/root/gpu-pass:/opt/gpu-pass:ro` + container `.so.1` symlinks + `ldconfig` + `--device /dev/nvidia*`; single-file binds are rejected by fuse-overlayfs |
| Host `curl 127.0.0.1:3080` refused (exit 7) while dsh serves in container | nested-docker netns isolation from sshd | container `ssh -fN -R 3080:127.0.0.1:3080 user@host:port` (bootstrap probes and pins it); curl exit 1 = non-HTTP/protocol, 28 = timeout — not the same as refused |
| Remote patch has `false:` where local had `off:` | YAML 1.1 round-trip renamed the bare key | raw-text awk block extraction (`^- id:` … next `^- id:`), never yaml.dump; breaks dsh reasoningEfforts schema otherwise |
| After host reboot: old URL 401 | token regenerated on every dsh boot; boot.sh self-heals dsh but the URL changed | `scripts/local-forward.sh <alias>` greps `/var/log/dsh-web.log` and prints the fresh full URL |
| After host reboot: dsh up but local forward dead | restart script lacked the tunnel target (first boot generated with empty target) | bootstrap pins `gen_restart "$TUNNEL"` + push at the END; boot.sh re-establishes the tunnel |
| Bootstrap's own ssh/docker-exec killed by its pkill | pkill pattern matched the caller's cmdline | restart script lives inside the container (own cmdline `bash /root/restart-dsh.sh`); bracket patterns `[s]sh` / `[R] 3080:` |
| Permission patch appended but UI still gated | cordis config is NOT hot-reloaded (unlike the model catalog) | restart dsh once (bootstrap does it only when the patch was newly added; healthy+patched = no-op) |
| `--dump-config` "shows" a permission value that isn't effective | dump prints unevaluated `!!js` expressions (runtime-evaluated) | verify via patch grep (`mode: danger-full-access` + `policy: never`) and runtime behavior, not dump output |
| Container create fails on small docker root | image + node + dsh need ~2.5 GiB on docker root fs | `--data-disk` moves only the workspace; image storage still needs root disk — expand it platform-side or no zero-human path |
| `docker cp` → `remount-ro .../merged/opt/gpu-pass ... operation not permitted` | on fuse-overlayfs platform hosts the daemon remount-ro's the merged tree for cp; the ro gpu-pass bind inside it makes that remount EPERM under platform seccomp. `docker commit` is NOT affected | never use docker cp there; stream files via `ssh host "docker exec -i C sh -c 'cat > /path'" < localfile` (same channel as the converge heredoc). bootstrap push_files does this |
| Reverse tunnel never comes up; container log shows ssh `usage:` or `Could not resolve hostname 10.x.x.x:2222` | ssh does not accept `user@host:port` as a destination; and boot.sh / docker exec call restart-dsh.sh with NO args, so the target must be baked at generation time, not parsed from runtime `$1` | gen_restart parses `user@host:port` into `-p PORT USER@HOST` while generating and bakes it into the script (verify: `grep 'ssh -fN' /root/restart-dsh.sh` shows `-p 2222 ... root@IP`) |
| `[: 0\n0: integer expression expected` on the permission-presence check | `grep -c` prints `0` AND exits 1 on zero matches, so `$(grep -c ... \|\| echo 0)` yields two lines | `$(... \|\| true)` then `head -1` + `${var:-0}` |
| After container rebuild, forward prints a token that 401s / local-forward finds no URL | the committed image carries the OLD `/var/log/dsh-web.log`; wait_url can match the stale token before boot.sh truncates it | wait past the restart script's own sleeps (2+12+2) + dsh boot (~20 s) after `docker run` before reading the token log |
| In a LOCAL dsh session with an ssh-remote "remote workspace", asking the agent `whoami` returns the LOCAL user, not remote root | ssh-remote mode registers a LOCAL mirror dir (`~/.dsh/remote-workspaces/<id>/`) as the workspace; only the explicit `remote_ssh_*` / `ssh_*` tools run on the remote host (as the SSH user). The agent's NATIVE bash runs where the dsh server runs — locally. Session ownership is visible in `~/.dsh/storages/workspace.json` + `~/.dsh/sessions/--<mirror-path>--/` | Not a bug; NO restart changes it — native context follows the server process, not the workspace label. For native root@remote (VSCode Remote-SSH semantics) open the REMOTE dsh instance via the Path A local-forward URL; there the server itself runs as root in the container. In the local session, remote work must go through `remote_ssh_exec` explicitly |
| First tool calls in a danger-full-access session fail with `invalid justification: expected a non-empty sentence` / `invalid escalation: sandbox_permissions requires a justification` (`SANDBOX_ESCALATION_INVALID_ARGUMENT`), then a retry succeeds | Argument validation, NOT a permission denial: the bash/fs tool schema still advertises the escalation pair `sandbox_permissions` + `justification` (must travel together, justification a non-empty sentence — `dsh-sandbox/lib/index.js` `validateEscalationArgs`). The model sometimes emits a malformed/empty escalation ask even though full access needs none; the call is rejected before execution | Self-heals in-turn, zero human clicks: the error returns to the model, which retries the plain call and it executes under danger-full-access + approval=never (observed 2026-11-20 remote UI e2e: errored retries then success, 19 s total). No config/restart change. Intervene ONLY if retries loop or a real denial marker `[sandbox: file access denied under <mode> mode]` appears — that means the SESSION's effective mode was switched narrower than the composition default; check the session mode, not the patch |

## Debug Logging

```bash
DEBUG='dsh:*' dsh web --no-open 2>&1 | tee /tmp/dsh-startup.log
```

## Web UI Lag & Workspace-Switching Lag

Verified on a local instance (2026-11-20, port 3080, workspace = a repo root
with a 39 GB `build_app/`). HTTP itself was fast (single-digit ms) — the lag
was event-loop churn, not the server or the file service.

| Symptom | Cause | Fix |
|---|---|---|
| Whole web UI sluggish; host log flooded with `[pet-link diag]` lines reporting multi-MB `meme` / `guide-shot` fetches | wallpaper-engine **pet-link** bridge — a third pet layer on top of the desktop-pet overlay and web-ui-pet. `routes/official-pet.js` runs `setInterval(coordinator.sweep, 1000)` and the browser half repeatedly re-fetches multi-MB images; every UI re-render (including workspace switches) bursts this activity | Disable the bridge in the user patch (the wallpaper-engine bundle inserts it with no config, so the override is safe): `- id: wallpaper-engine-pet-link` / `name: dsh-plugin-wallpaper-engine-pet-link` / `config: {linkEnabled: false}`. Server gate: `index.js` `isLinkEnabled: () => config.linkEnabled !== false`; when false every `/pet-link/official` route returns `bridge-disabled` and sweep has no targets. Restart dsh (cordis config is NOT hot-reloaded). Evidence: before 48/56 log lines were pet-link diag (2.4 MB memes); after 0 |
| Workspace switching sluggish | Same pet-link churn amplified by the full UI re-render — NOT the file service: `workspace-files list` reads only direct children, `search` runs `rg --files` honoring `.gitignore` / `.ignore` (`build*/` already ignored), and `git status` on the 39 GB-build repo took ~80 ms | Fix pet-link first. Only if switching is still slow add a root `.ignore` or literal `excludedPaths` in the search request. Do not blame build dirs without measuring |
| Boot shows `N entries did not activate`; UI features missing | inotify-instance exhaustion at the boot instant (see the EMFILE row in Startup & Access) | Restart dsh once the transient IDE/browser peak passes; raise limits host-side (`sudo sysctl -w fs.inotify.max_user_instances=1024 fs.inotify.max_user_watches=524288`, persist in `/etc/sysctl.d/99-inotify.conf`). Caveat: counting `anon_inode:inotify` across `/proc/*/fd` OVER-counts — fork-inherited fds are the same kernel instance — so a raw count above the limit does not prove exhaustion; trust the actual boot error |

Diagnosis commands:

```bash
grep -c 'pet-link' <dsh-log>                                  # flood?
curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' http://127.0.0.1:<port>/   # 401 w/o token is normal; expect ms
time git status --porcelain | wc -l                           # workspace git path cost
```
