# Remote Zero-Human Bootstrap (Path A)

One-click full `dsh web` on a remote SSH host whose **agent native execution
context is that host** — VSCode Remote-SSH semantics: every run/read/write/edit
tool and every terminal tab executes on the remote machine (including
`nvidia-smi`), while the browser stays local via `ssh -L`. Goal: a brand-new
machine converges with **zero human steps**, and survives host reboot without
anyone touching it.

This is the hardened successor of the manual Path A recipe in SKILL.md. All
platform claims below were **probed on huoshan_8 (2026-11-20)**, not assumed.

## Scripts

| Script | Role |
|---|---|
| `scripts/bootstrap-remote.sh <ssh-alias>` | audit → substrate decision → install → credentials/config → permission mode → netns/tunnel → commit + reboot self-heal → local forward → verify |
| `scripts/verify-remote.sh <ssh-alias>` | native-context / GPU / zero-human-permission / activate-failures / token-URL / forward-401 / forward-token checks |
| `scripts/local-forward.sh <ssh-alias>` | idempotent local `ssh -L` + print the FULL fresh token URL (run after any remote reboot) |

```bash
bash scripts/bootstrap-remote.sh <ssh-alias> --audit-only  # probe + decision, change nothing
bash scripts/bootstrap-remote.sh <ssh-alias>               # converge; default --permission-mode danger-full-access
bash scripts/verify-remote.sh <ssh-alias>                  # all PASS = done
bash scripts/local-forward.sh <ssh-alias>                  # reprint URL (token regenerates every boot)
```

Useful flags: `--local-port` (default 3180), `--remote-port` (default 3080),
`--container` (default dsh-remote), `--workspace` (default /root/dsh-ws),
`--node-version` (v24.11.0), `--dsh-version` (0.2.2-rc.5), `--data-disk PATH`
(big-disk hosts: extra bind at /data), `--with-plugins` (dsh-web-all,
allowBuilds auto-approved), `--tunnel-target user@host:port` (override the
auto-probed reverse-tunnel target), `--permission-mode` (see below).

Inputs reused from the **local** machine (the user's own, never someone
else's): `~/.dsh/.credentials.yaml` `refs:` (`DEEPROUTE_*_API_KEY` only) and
`~/.dsh/profiles/web/cordis.patch.yml` (`llm-pi-ai` + `agent-default-model`
blocks). **Boundary: the gateway key must be the user's own** — pushing a
colleague's key to a remote host violates the doc's red lines.

## Substrate decision tree (first principles, probed not assumed)

```
glibc >= 2.28            -> native user-space install (no sudo, ~/.npm-global + ~/.local/node24)
else docker available    -> ubuntu:24.04 container substrate (node/dsh live inside)
else                     -> HARD FAIL: no zero-human substrate exists
```

Why glibc 2.28: modern Node (>=24) binaries require it; huoshan_8's host is
Ubuntu 18.04 (glibc 2.27) so native install is impossible there — the container
is not a preference, it is the only substrate. The audit phase reads
`ldd --version`, `command -v docker`, `df /`, `/dev/nvidia0` over SSH and prints
the decision before changing anything (`--audit-only`).

Disk: image + node + dsh need **~2.5 GiB on the docker root filesystem**.
`--data-disk` moves the *workspace* to a big disk but cannot move image
storage; if docker root is smaller than ~2.5 GiB there is no zero-human path
without platform-side disk changes.

## Container substrate (what bootstrap converges)

- Base `ubuntu:24.04` (pull fallback: platform mirror
  `ml-platform-hub-cn-shanghai.cr.volces.com/library/ubuntu:24.04` retagged).
- Node tarball from npmmirror binary mirror (fallback nodejs.org) → `/opt/node`,
  symlinks into `/usr/local/bin`; npm registry → npmmirror; pnpm; pinned dsh.
- apt: ca-certificates curl xz-utils openssh-client procps git bubblewrap
  iproute2 libatomic1 libstdc++6 (bubblewrap installed for completeness even
  though danger-full-access skips it).
- Container ssh keypair (`/root/.ssh/id_ed25519`) generated once; pubkey
  appended to the **host's** `authorized_keys` (idempotent `grep -qF`) so the
  reverse tunnel works with BatchMode.
- `--restart unless-stopped` + committed `dsh-base:latest` with
  `CMD ["/bin/bash","/root/boot.sh"]` (see Self-heal).

### GPU passthrough (directory bind, NOT single files)

Managed platforms use fuse-overlayfs, which **rejects single-file bind
mounts**; a directory bind always works. Recipe (host side):

1. `mkdir /root/gpu-pass`; copy `nvidia-smi` + `libnvidia-ml.so*`,
   `libcuda.so*`, `libnvidia-cfg.so*` (resolved via `ldconfig -p`) into it.
2. `docker run --device /dev/nvidia0..7 --device /dev/nvidiactl
   --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
   --device /dev/nvidia-modeset -v /root/gpu-pass:/opt/gpu-pass:ro ...`
   (bootstrap enumerates whatever `/dev/nvidia*` exist).

Container side: for each `/opt/gpu-pass/lib*.so.*` create
`/usr/lib/x86_64-linux-gnu/<base>.so.1` symlink (+ `libcuda.so` dev symlink),
`ldconfig`, and `ln -sf /opt/gpu-pass/nvidia-smi /usr/local/bin/nvidia-smi`.
Result: agent sessions **and terminal tabs** run `nvidia-smi` and see all GPUs
(verified 8×L20 on huoshan_8). No `--gpus all` / nvidia-container-toolkit
needed (often absent on managed platforms).

### netns probe → reverse tunnel (only when needed)

Nested-docker platforms put the container in a **different network namespace**
from sshd: the host's `curl 127.0.0.1:3080` is refused even though dsh serves
inside the container. Bootstrap probes exactly this:

```
host: curl -s -m 3 http://127.0.0.1:<remote-port>/
  success -> no tunnel needed (sshd netns sees dsh loopback)
  refused -> container must ssh -R <port>:127.0.0.1:<port> back to the sshd netns
```

Tunnel target candidates are probed with BatchMode ssh from inside the
container: `ssh -G <alias>` facts (user@hostname:port) first, then the
container's default gateway. `--tunnel-target` overrides. The chosen target is
**pinned into the on-disk `/root/restart-dsh.sh` at the end of bootstrap** —
otherwise a reboot would self-heal dsh but lose the tunnel (the first boot
script is generated with an empty target).

curl exit-code semantics used by the probes: **7 = connection refused**
(netns isolation or dsh down), **1 = non-HTTP/protocol error**, **28 =
timeout**. Distinguish before acting.

## Zero-human permission model (the core fix)

### Do NOT fight the platform seccomp — it is unwinnable

Probed on huoshan_8 with throwaway `--rm` containers (no residue):

| Attempt | Result |
|---|---|
| `docker run --cap-add SYS_ADMIN` | **hard rejected**: `Capability is rejected:SYS_ADMIN` |
| `--security-opt seccomp=unconfined` / `apparmor=unconfined` | **silently ignored**; `docker inspect` still shows the platform's `varmor-mlplatform` apparmor + a seccomp JSON denying `unshare(CLONE_NEWUSER=268435456)`/`bpf`; runtime `Seccomp: 2` |
| `unshare --user --pid --fork true` inside | **EPERM** |

Conclusion: on managed ML platforms you cannot get user namespaces, so
bwrap-based sandboxing (`workspace-write` mode) can never confine, and dsh
falls back to a **per-session human "full permissions" click**. Adding
SYS_ADMIN to an existing container also forces a destroy/recreate that the
platform rejects — an earlier version of the bootstrap did this and broke the
running container; that code path is removed.

### The zero-human path: `danger-full-access` + `approval=never`

Source-confirmed in `@deepseek-ai/dsh-bash-sandbox/src/index.ts` (~L91-95):
when the sandbox mode is `danger-full-access`, the plugin **skips
`confine()`/bwrap entirely** — no userns required. Combined with
`@deepseek-ai/dsh-user-approval` `policy: never`, no command ever waits for a
human click. Bootstrap persists both as patch entries (own idempotency guard):

```yaml
- id: sandbox-policy
  name: "@deepseek-ai/dsh-sandbox-policy"
  config:
    mode: danger-full-access
- id: approval
  name: "@deepseek-ai/dsh-user-approval"
  config:
    policy: never
```

plus `export DSH_PERMISSION_MODE=danger-full-access` in `restart-dsh.sh`
(belt and suspenders). These override the base profile's `!!js` defaults.
**cordis config is NOT hot-reloaded** — a newly added permission patch requires
one dsh restart (bootstrap does it only when `PERM_PRESENT=0`; an already
serving + patched instance is a convergent no-op, never restarted).

`--permission-mode workspace-write` is still available for platforms where
`unshare` works (verify-remote.sh then gates on userns instead). On
seccomp-forced platforms it leaves the human click — bootstrap prints a WARN.

### Security trade-off (explicit, not hidden)

`danger-full-access` + `approval=never` means the agent shell is unconfined and
ungated. **The isolation boundary is the container itself**: disposable,
workspace bind-mounted, no host access beyond the binds — exactly the trust
model of VSCode Remote-SSH (the remote terminal already has the logged-in
user's full privileges). On the *native* substrate the same mode grants the
SSH user's full privileges; acceptable only on the user's own dev machine.
Never deploy this mode on shared/multi-tenant hosts without the container
boundary. The `?token=` URL remains a password-equivalent secret; binding
stays 127.0.0.1, exposure only via `ssh -L`.

### Verification semantics

- `verify-remote.sh` in danger-full-access mode greps the persisted patch for
  `mode: danger-full-access` **and** `policy: never` → `zero-human-permission
  PASS`; `unshare` EPERM is reported as **informational**, not a failure.
- `dsh --profile web --dump-config` prints **unevaluated `!!js` expressions**
  (they are evaluated at runtime) — it validates composition only; never use
  its output to assert the effective permission value.
- Browser proof (≤3 steps): open the forwarded token URL → new session → run
  `id` in the default sandbox → expect UID 0 + the remote hostname, **with no
  "full permissions" click anywhere** (approval=never).

## Credentials & config transfer (raw text, never YAML round-trip)

YAML 1.1 parsers turn the bare key `off:` into `false:`, which breaks dsh's
`reasoningEfforts` schema. Bootstrap therefore extracts the
`llm-pi-ai` / `agent-default-model` blocks from the local patch as **raw
text** with awk (`^- id: <id>` up to the next top-level `^- id:`) and appends
them verbatim; credentials are rebuilt as `version: 1` + only the
`DEEPROUTE_*API_KEY` refs, chmod 600 on both ends. Append is guarded by
`grep -q '^- id: llm-pi-ai'` / `'^- id: sandbox-policy'` so re-runs never
duplicate blocks. Remote stores may legitimately contain fewer refs than
local (e.g. gateway-only); the incremental push is safe.

## Persistence & reboot self-heal

- `/root/restart-dsh.sh` (in container): pkill old dsh + tunnel, relaunch
  `dsh web --no-open --port <p>` with the permission env, re-establish the
  pinned `ssh -fN -R` tunnel, print ACTIVATE_FAILS / URL / TUNNEL_PROCS.
- `/root/boot.sh` = container CMD: run restart-dsh.sh → log to
  `/var/log/dsh-boot.log` → `exec sleep infinity`.
- `docker commit --change 'CMD ["/bin/bash","/root/boot.sh"]' dsh-remote
  dsh-base:latest`; if the live container's CMD is not boot.sh, bootstrap
  stops/rm/runs it **plain** (no SYS_ADMIN, no seccomp overrides) from the
  committed image with `--restart unless-stopped`. Host reboot → docker
  restarts the container → boot.sh brings dsh + tunnel back. Zero human steps.
- After any reboot the **token is regenerated**: run `local-forward.sh` to
  print the fresh URL (it greps `/var/log/dsh-web.log`).

Idempotency guards: healthy + permission-patch-present → skip restart;
CMD already boot.sh → skip rebuild; node/dsh at pinned versions → skip
install; pubkey already authorized → skip append. A second full run is a
convergent no-op.

## Pitfalls (all hit and fixed during this work)

| Pitfall | Root cause | Fix in scripts |
|---|---|---|
| `pkill -f 'ssh -fN.*-R 3080'` kills the bootstrap's own ssh/docker-exec | pattern matches the caller's cmdline | restart script lives **inside** the container (own cmdline = `bash /root/restart-dsh.sh`); tunnel pattern uses bracket trick `[s]sh` / `[R] 3080:` |
| host curl refused while dsh serves | nested-docker netns isolation | netns probe → container `ssh -R` hop (above) |
| `off:` became `false:` in remote patch | YAML 1.1 round-trip | raw-text awk block extraction, no yaml.dump |
| reboot loses the tunnel | first-boot restart script generated with empty target | `gen_restart "$TUNNEL"` + push **at the end** of bootstrap (pin) |
| single-file GPU bind mount fails | fuse-overlayfs rejects file binds | directory bind `/root/gpu-pass:/opt/gpu-pass:ro` + container-side symlinks |
| `--cap-add SYS_ADMIN` destroys a working container | platform hard-rejects the cap on (re)create | removed; plain `docker run`; zero-human comes from permission mode |
| permission patch added but UI still asks | cordis config not hot-reloaded | restart only when `PERM_PRESENT=0` |
| old token URL 401 after reboot | token regenerated every boot | grep `dsh-web.log` for the fresh URL; local-forward prints it |
| `--dump-config` "shows" wrong permission value | `!!js` unevaluated at dump time | verify via patch grep + runtime behavior |
| `docker exec ... curl` exit 1 vs 7 vs 28 confused | 1=protocol/non-HTTP, 7=refused, 28=timeout | probe interprets 7 as netns/down, not as HTTP failure |

## When zero-human is impossible (hard boundaries)

- glibc < 2.28 **and** no docker → bootstrap dies with an explicit message.
- docker root filesystem < ~2.5 GiB and no platform-side disk expansion.
- Platform rejects even plain `docker run` (no container substrate at all).
- Someone else's gateway key (policy boundary, not technical).

In these cases fall back to Path B (local dsh + dsh-ssh-remote), which is
**not** VSCode Remote-SSH semantics (native tools stay local) — say so
explicitly rather than silently downgrading.
