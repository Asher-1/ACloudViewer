#!/usr/bin/env bash
# bootstrap-remote.sh — zero-human Path A bootstrap: full dsh web on a remote
# SSH host whose agent NATIVE execution context is that host (VSCode
# Remote-SSH semantics), GPU passthrough included, browser stays local.
#
# Usage:
#   bootstrap-remote.sh <ssh-alias> [options]
#     --audit-only          print audit + decision + container flag gaps, change nothing
#     --local-port N        local forward port (default 3180)
#     --remote-port N       dsh web port on remote loopback (default 3080)
#     --container NAME      docker container name (default dsh-remote)
#     --workspace PATH      remote workspace dir, host-bind (default /root/dsh-ws)
#     --node-version V      node tarball version (default v24.11.0)
#     --dsh-version V       pinned dsh version (default 0.2.2-rc.5)
#     --tunnel-target U@H:P override the container->host reverse-tunnel target
#     --data-disk PATH      extra host dir bind-mounted at /data (big-disk hosts)
#     --with-plugins        also add @linxin666/dsh-web-all (allowBuilds auto-approved)
#     --permission-mode M   dsh sandbox/approval mode (default danger-full-access).
#                           danger-full-access => no bwrap/userns + approval=never
#                           => ZERO human click even when the platform seccomp
#                           blocks unshare(CLONE_NEWUSER). The container itself is
#                           the isolation boundary (VSCode Remote-SSH semantics).
#                           workspace-write => granular sandbox, but needs userns;
#                           on managed platforms that force seccomp this leaves
#                           the per-session "full permissions" human click.
#
# Inputs reused from the LOCAL machine (the user's own, never someone else's):
#   ~/.dsh/.credentials.yaml refs DEEPROUTE_*_API_KEY  -> remote store (600)
#   ~/.dsh/profiles/web/cordis.patch.yml               -> raw-text block copy
#     (llm-pi-ai + agent-default-model; NO YAML round-trip: YAML 1.1 turns
#      `off:` into `false:` and breaks the dsh reasoningEfforts schema)
#
# Decision tree (first principles, probed not assumed):
#   glibc >= 2.28            -> native user-space install (no sudo)
#   else docker available    -> ubuntu:24.04 container substrate
#   else                     -> hard fail (no zero-human path exists)
#   netns: host curl dsh port after boot; refused -> container ssh -R hop
#   sandbox: do NOT fight platform seccomp. Managed ML platforms REJECT
#     --cap-add SYS_ADMIN and FORCE-inject seccomp/apparmor (overrides ignored).
#     unshare(CLONE_NEWUSER) is therefore unavailable => set dsh permission mode
#     danger-full-access (skips bwrap, approval=never) for zero-human operation.
#
# Idempotent: healthy + CMD aligned + config present => convergent no-op
# (dsh is NOT restarted when already serving with the permission patch applied).
# Liveness: docker commit dsh-base:latest with CMD boot.sh + --restart
# unless-stopped => host reboot self-heals with zero human steps.
set -euo pipefail

ALIAS="" LOCAL_PORT=3180 REMOTE_PORT=3080 CONTAINER=dsh-remote
WS=/root/dsh-ws GPU_DIR=/root/gpu-pass NODE_VER=v24.11.0 DSH_VER=0.2.2-rc.5
AUDIT_ONLY=0 WITH_PLUGINS=0 DATA_DISK="" TUNNEL_TARGET="" PERMISSION_MODE=danger-full-access

while [ $# -gt 0 ]; do
  case "$1" in
    --audit-only) AUDIT_ONLY=1 ;;
    --local-port) LOCAL_PORT=$2; shift ;;
    --remote-port) REMOTE_PORT=$2; shift ;;
    --container) CONTAINER=$2; shift ;;
    --workspace) WS=$2; shift ;;
    --node-version) NODE_VER=$2; shift ;;
    --dsh-version) DSH_VER=$2; shift ;;
    --tunnel-target) TUNNEL_TARGET=$2; shift ;;
    --data-disk) DATA_DISK=$2; shift ;;
    --with-plugins) WITH_PLUGINS=1 ;;
    --permission-mode) PERMISSION_MODE=$2; shift ;;
    -h|--help) grep '^# ' "$0" | head -40; exit 0 ;;
    *) [ -z "$ALIAS" ] && ALIAS=$1 || { echo "unknown arg: $1" >&2; exit 2; } ;;
  esac
  shift
done
[ -n "$ALIAS" ] || { echo "usage: $0 <ssh-alias> [--audit-only] ..." >&2; exit 2; }

log() { printf '[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap] FAIL: %s\n' "$*" >&2; exit 1; }
r() { ssh "$ALIAS" "$@"; }

SKILL_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOCAL_PATCH="$HOME/.dsh/profiles/web/cordis.patch.yml"
LOCAL_CREDS="$HOME/.dsh/.credentials.yaml"
[ -f "$LOCAL_PATCH" ] || die "local profile patch missing: $LOCAL_PATCH"
[ -f "$LOCAL_CREDS" ] || die "local credentials store missing: $LOCAL_CREDS"

# ---------------- phase 0: audit ----------------
AUDIT=$(r 'echo GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE "[0-9]+\.[0-9]+$" || echo 0)
echo DOCKER=$(command -v docker >/dev/null 2>&1 && echo yes || echo no)
echo AVAIL=$(df -B1 --output=avail / 2>/dev/null | tail -1 | tr -d " ")
echo GPU=$([ -e /dev/nvidia0 ] && echo yes || echo no)
echo HOSTUSER=$(id -un)')
GLIBC=$(sed -n 's/^GLIBC=//p' <<<"$AUDIT")
DOCKER=$(sed -n 's/^DOCKER=//p' <<<"$AUDIT")
AVAIL=$(sed -n 's/^AVAIL=//p' <<<"$AUDIT")
GPU=$(sed -n 's/^GPU=//p' <<<"$AUDIT")
log "audit: glibc=$GLIBC docker=$DOCKER free=$((AVAIL/1024/1024))MiB gpu=$GPU"
[ "$AVAIL" -gt 2500000000 ] || log "WARN: <2.5GiB free on /; consider --data-disk for workspace (image storage still needs ~2.5GiB)"

MODE=container
if [ "$(printf '%s\n2.28\n' "$GLIBC" | sort -V | head -1)" = "2.28" ]; then
  MODE=native
elif [ "$DOCKER" != yes ]; then
  die "glibc $GLIBC < 2.28 and no docker: no zero-human substrate available"
fi
log "substrate decision: $MODE"

flag_gaps() {
  r "docker inspect $CONTAINER --format 'CAP={{json .HostConfig.CapAdd}} SEC={{json .HostConfig.SecurityOpt}} CMD={{json .Config.Cmd}}'" 2>/dev/null || echo "NO_CONTAINER"
}

if [ "$AUDIT_ONLY" = 1 ]; then
  log "current container flags: $(flag_gaps)"
  log "audit-only: would run mode=$MODE permission_mode=$PERMISSION_MODE tunnel=${TUNNEL_TARGET:-auto}"
  exit 0
fi

# ---------------- local config extraction (raw text, no YAML round-trip) ----------------
LT=$(mktemp -d); trap 'rm -rf "$LT"' EXIT
extract_block() { # $1 = entry id; raw lines from "- id: X" up to next top-level "- id:"
  awk -v id="$1" '$0 ~ "^- id: "id"$" {f=1; print; next} /^- id:/ {f=0} f' "$LOCAL_PATCH"
}
{ extract_block llm-pi-ai; extract_block agent-default-model; } > "$LT/patch-blocks.yml"
[ -s "$LT/patch-blocks.yml" ] || die "could not extract llm-pi-ai/agent-default-model blocks from $LOCAL_PATCH"
# permission overrides (separate file + own idempotency guard). danger-full-access
# makes dsh-bash-sandbox skip confine()/bwrap (no userns) and approval policy=never.
if [ "$PERMISSION_MODE" = danger-full-access ]; then
  cat > "$LT/permission-blocks.yml" <<'YML'
- id: sandbox-policy
  name: "@deepseek-ai/dsh-sandbox-policy"
  config:
    mode: danger-full-access
- id: approval
  name: "@deepseek-ai/dsh-user-approval"
  config:
    policy: never
YML
else
  cat > "$LT/permission-blocks.yml" <<YML
- id: sandbox-policy
  name: "@deepseek-ai/dsh-sandbox-policy"
  config:
    mode: $PERMISSION_MODE
- id: approval
  name: "@deepseek-ai/dsh-user-approval"
  config:
    policy: ask
YML
fi
{
  echo "version: 1"
  echo "refs:"
  awk '/^refs:/{f=1;next} /^[^ ]/{f=0} f && /^  DEEPROUTE_[A-Z_]*API_KEY:/{print}' "$LOCAL_CREDS"
} > "$LT/credentials.yaml"
grep -q DEEPROUTE_GATEWAY_API_KEY "$LT/credentials.yaml" || die "no DEEPROUTE_GATEWAY_API_KEY in local refs"
chmod 600 "$LT/credentials.yaml"

# ssh -G facts for tunnel candidates (container must reach the sshd netns)
SSH_HOST=$(ssh -G "$ALIAS" | awk '/^hostname /{print $2}')
SSH_PORT=$(ssh -G "$ALIAS" | awk '/^port /{print $2}')
SSH_USER=$(ssh -G "$ALIAS" | awk '/^user /{print $2}')

# ---------------- container-side generated scripts ----------------
gen_restart() { # $1 = tunnel target or empty; parsed HERE and baked into the
  # script (boot.sh / docker exec call it with no args, so runtime $1 is empty)
  local TU="" TH="" TP=22 THP
  if [ -n "$1" ]; then
    TU=${1%@*}; THP=${1#*@}
    case "$THP" in *:*) TH=${THP%:*}; TP=${THP##*:} ;; *) TH=$THP ;; esac
  fi
  cat > "$LT/restart-dsh.sh" <<EOF
#!/bin/bash
# idempotent dsh web + (optional) reverse tunnel; lives in the container.
# pkill patterns are safe here: this file's own cmdline is "bash /root/restart-dsh.sh".
pkill -f 'dsh web --no-open' 2>/dev/null
pkill -f '[s]sh -fN.*-R $REMOTE_PORT' 2>/dev/null
sleep 2
cd $WS
export DSH_PERMISSION_MODE=$PERMISSION_MODE
nohup dsh web --no-open --port $REMOTE_PORT > /var/log/dsh-web.log 2>&1 &
sleep 12
if [ -n "$TH" ]; then
  ssh -fN -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $TP -R $REMOTE_PORT:127.0.0.1:$REMOTE_PORT $TU@$TH
fi
sleep 2
echo ACTIVATE_FAILS=\$(grep -c 'did not activate' /var/log/dsh-web.log || true)
echo URL=\$(grep -o 'http://127.0.0.1:$REMOTE_PORT/?token=[A-Za-z0-9_-]*' /var/log/dsh-web.log | head -1)
echo TUNNEL_PROCS=\$(ps aux | grep -c '[R] $REMOTE_PORT:127.0.0.1:$REMOTE_PORT' || true)
EOF
  cat > "$LT/boot.sh" <<'EOF'
#!/bin/bash
# container CMD: self-heal dsh + tunnel on every (re)start, then stay alive.
/bin/bash /root/restart-dsh.sh > /var/log/dsh-boot.log 2>&1
exec sleep infinity
EOF
}

push_files() { # $@ = local names under $LT; stream via docker exec -i stdin.
  # docker cp is unusable on fuse-overlayfs platform hosts: the daemon
  # remount-ro's the merged tree and the ro gpu-pass bind inside it makes
  # that remount EPERM under the platform seccomp (verified 2026-11-20).
  # exec+stdin never touches the daemon mount path (same channel as CONVERGE).
  for f in "$@"; do
    case "$f" in
      credentials.yaml)
        r "docker exec -i $CONTAINER sh -c 'mkdir -p /root/.dsh && cat > /root/.dsh/.credentials.yaml && chmod 600 /root/.dsh/.credentials.yaml'" < "$LT/$f" ;;
      *)
        r "docker exec -i $CONTAINER sh -c 'cat > /root/$f && chmod +x /root/$f'" < "$LT/$f" ;;
    esac
  done
}

wait_url() {
  local i url=""
  for i in $(seq 1 30); do
    url=$(r "docker exec $CONTAINER grep -o 'http://127.0.0.1:$REMOTE_PORT/?token=[A-Za-z0-9_-]*' /var/log/dsh-web.log 2>/dev/null | head -1" || true)
    [ -n "$url" ] && { echo "$url"; return 0; }
    sleep 3
  done
  return 1
}

# ================= native mode (glibc >= 2.28) =================
if [ "$MODE" = native ]; then
  log "native install (user-owned, no sudo)"
  scp -q "$LT/credentials.yaml" "$ALIAS:/tmp/.dsh-creds.$$"
  scp -q "$LT/patch-blocks.yml" "$ALIAS:/tmp/.dsh-patch.$$"
  scp -q "$LT/permission-blocks.yml" "$ALIAS:/tmp/.dsh-perm.$$"
  r "bash -s" <<EOF
set -euo pipefail
export PATH=\$HOME/.npm-global/bin:\$HOME/.local/node24/bin:\$PATH
if [ "\$(node -v 2>/dev/null | cut -d. -f1 | tr -d v)" -lt 24 ] 2>/dev/null || ! command -v node >/dev/null; then
  mkdir -p \$HOME/.local
  curl -fsSL https://registry.npmmirror.com/-/binary/node/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz -o /tmp/node.tar.xz \
    || curl -fsSL https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz -o /tmp/node.tar.xz
  tar -xJf /tmp/node.tar.xz -C \$HOME/.local
  ln -sfn \$HOME/.local/node-$NODE_VER-linux-x64 \$HOME/.local/node24
fi
npm config set registry https://registry.npmmirror.com
npm config set prefix \$HOME/.npm-global
[ "\$(dsh --version 2>/dev/null)" = "$DSH_VER" ] || npm install -g @deepseek-ai/dsh@$DSH_VER
mkdir -p \$HOME/.dsh
install -m 600 /tmp/.dsh-creds.$$ \$HOME/.dsh/.credentials.yaml
rm -f /tmp/.dsh-creds.$$
pkill -f 'dsh web --no-open' 2>/dev/null || true
sleep 2
export DSH_PERMISSION_MODE=$PERMISSION_MODE
cd \$HOME && nohup dsh web --no-open --port $REMOTE_PORT > \$HOME/dsh-web.log 2>&1 &
sleep 12
grep -q '^- id: llm-pi-ai' \$HOME/.dsh/profiles/web/cordis.patch.yml 2>/dev/null || cat /tmp/.dsh-patch.$$ >> \$HOME/.dsh/profiles/web/cordis.patch.yml
grep -q '^- id: sandbox-policy' \$HOME/.dsh/profiles/web/cordis.patch.yml 2>/dev/null || cat /tmp/.dsh-perm.$$ >> \$HOME/.dsh/profiles/web/cordis.patch.yml
rm -f /tmp/.dsh-patch.$$ /tmp/.dsh-perm.$$
dsh --profile web --dump-config > /dev/null && echo COMPOSE_OK
EOF
  log "native boot done; local forward next"
  bash "$SKILL_DIR/scripts/local-forward.sh" "$ALIAS" --local-port "$LOCAL_PORT" --remote-port "$REMOTE_PORT" --native
  bash "$SKILL_DIR/scripts/verify-remote.sh" "$ALIAS" --local-port "$LOCAL_PORT" --remote-port "$REMOTE_PORT" --native --permission-mode "$PERMISSION_MODE"
  exit 0
fi

# ================= container mode =================
# ---- host-side prep: workspace + GPU driver directory (single-file binds are
# ---- rejected by fuse-overlayfs; a directory bind always works) ----
r "WS=$WS GPU_DIR=$GPU_DIR bash -s" <<'EOS'
set -e
mkdir -p "$WS" "$GPU_DIR"
if [ -e /dev/nvidia0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  if [ ! -x "$GPU_DIR/nvidia-smi" ]; then
    cp -f "$(command -v nvidia-smi)" "$GPU_DIR/nvidia-smi"
    chmod +x "$GPU_DIR/nvidia-smi"
    for lib in libnvidia-ml libcuda libnvidia-cfg; do
      p=$(ldconfig -p | awk -v l="$lib.so" '$1 ~ l {print $NF; exit}')
      [ -n "$p" ] && cp -f "$p" "$GPU_DIR/"
    done
  fi
fi
EOS

DEV_ARGS=""
if [ "$GPU" = yes ]; then
  for d in $(r 'ls -d /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset 2>/dev/null || true'); do
    DEV_ARGS="$DEV_ARGS --device $d"
  done
fi
BIND_ARGS="-v $WS:$WS -v $GPU_DIR:/opt/gpu-pass:ro"
[ -n "$DATA_DISK" ] && BIND_ARGS="$BIND_ARGS -v $DATA_DISK:/data"

# ---- ensure working container ----
if ! r "docker ps -a --format '{{.Names}}' | grep -qx $CONTAINER"; then
  IMG=$(r 'docker image inspect dsh-base:latest >/dev/null 2>&1 && echo dsh-base:latest || echo ubuntu:24.04')
  if [ "$IMG" = ubuntu:24.04 ]; then
    r "docker image inspect ubuntu:24.04 >/dev/null 2>&1 || docker pull ubuntu:24.04 || { docker pull ml-platform-hub-cn-shanghai.cr.volces.com/library/ubuntu:24.04 && docker tag ml-platform-hub-cn-shanghai.cr.volces.com/library/ubuntu:24.04 ubuntu:24.04; }"
  fi
  log "creating container $CONTAINER from $IMG"
  # shellcheck disable=SC2086
  r "docker run -d --name $CONTAINER --restart unless-stopped $BIND_ARGS $DEV_ARGS $IMG sleep infinity"
else
  r "docker start $CONTAINER >/dev/null"
fi

# ---- converge runtime inside container ----
log "converging node/pnpm/dsh/GPU libs inside container"
r "docker exec -i -e NODE_VER=$NODE_VER -e DSH_VER=$DSH_VER $CONTAINER bash -s" <<'EOS'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl xz-utils openssh-client procps git bubblewrap iproute2 libatomic1 libstdc++6 >/dev/null
if [ "$(node -v 2>/dev/null)" != "$NODE_VER" ]; then
  curl -fsSL "https://registry.npmmirror.com/-/binary/node/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz" -o /tmp/node.tar.xz \
    || curl -fsSL "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz" -o /tmp/node.tar.xz
  tar -xJf /tmp/node.tar.xz -C /opt
  ln -sfn "/opt/node-$NODE_VER-linux-x64" /opt/node
  for b in node npm npx; do ln -sf /opt/node/bin/$b /usr/local/bin/$b; done
fi
export PATH=/opt/node/bin:$PATH
npm config set registry https://registry.npmmirror.com
command -v pnpm >/dev/null || npm install -g pnpm
ln -sf /opt/node/bin/pnpm /usr/local/bin/pnpm
if [ "$(dsh --version 2>/dev/null)" != "$DSH_VER" ]; then
  npm install -g "@deepseek-ai/dsh@$DSH_VER"
  ln -sf /opt/node/bin/dsh /usr/local/bin/dsh
fi
# GPU userspace: .so.1 symlinks to the ro-bind driver directory + ldconfig
if [ -d /opt/gpu-pass ]; then
  for f in /opt/gpu-pass/lib*.so.*; do
    [ -e "$f" ] || continue
    base=$(basename "$f"); base=${base%.so.*}
    ln -sf "$f" "/usr/lib/x86_64-linux-gnu/$base.so.1"
    [ "$base" = libcuda ] && ln -sf "$f" /usr/lib/x86_64-linux-gnu/libcuda.so
  done
  ldconfig
  [ -x /opt/gpu-pass/nvidia-smi ] && ln -sf /opt/gpu-pass/nvidia-smi /usr/local/bin/nvidia-smi
fi
[ -f /root/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519 -q
mkdir -p /root/.dsh
echo CONVERGE_OK
EOS

# ---- container ssh key -> host authorized_keys (for the reverse tunnel) ----
PUB=$(r "docker exec $CONTAINER cat /root/.ssh/id_ed25519.pub")
r "mkdir -p \$HOME/.ssh && chmod 700 \$HOME/.ssh && touch \$HOME/.ssh/authorized_keys && chmod 600 \$HOME/.ssh/authorized_keys && grep -qF '$PUB' \$HOME/.ssh/authorized_keys || echo '$PUB' >> \$HOME/.ssh/authorized_keys"

# ---- credentials + first boot (creates profile) ----
gen_restart ""
push_files credentials.yaml restart-dsh.sh boot.sh patch-blocks.yml permission-blocks.yml
PERM_PRESENT=$(r "docker exec $CONTAINER grep -c '^- id: sandbox-policy' /root/.dsh/profiles/web/cordis.patch.yml" 2>/dev/null || true)
PERM_PRESENT=$(head -1 <<<"$PERM_PRESENT"); PERM_PRESENT=${PERM_PRESENT:-0}
if [ "$PERM_PRESENT" -gt 0 ] && r "docker exec $CONTAINER curl -s -m 3 -o /dev/null http://127.0.0.1:$REMOTE_PORT/" 2>/dev/null; then
  log "dsh already serving + permission patch present; skipping restart (no-op)"
else
  log "first boot (creates profile; no tunnel yet)"
  r "docker exec $CONTAINER /bin/bash /root/restart-dsh.sh" || true
fi
URL=$(wait_url) || die "dsh web did not print URL; check: ssh $ALIAS docker exec $CONTAINER tail -50 /var/log/dsh-web.log"

# ---- netns probe: does the sshd netns see dsh loopback directly? ----
TUNNEL=""
if r "curl -s -m 3 -o /dev/null http://127.0.0.1:$REMOTE_PORT/" ; then
  log "netns: host loopback reaches dsh directly (no reverse tunnel needed)"
else
  log "netns: isolated (nested docker) -> container ssh -R hop required"
  if [ -n "$TUNNEL_TARGET" ]; then
    TUNNEL=$TUNNEL_TARGET
  else
    GW=$(r "docker exec $CONTAINER sh -c 'ip route 2>/dev/null | awk \"/default/{print \\\$3; exit}\"'" || true)
    for cand in "$SSH_USER@$SSH_HOST:$SSH_PORT" "$SSH_USER@$GW:$SSH_PORT"; do
      u=${cand%@*}; hp=${cand#*@}; h=${hp%:*}; p=${hp#*:}
      if r "docker exec $CONTAINER ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -p $p $u@$h true" 2>/dev/null; then
        TUNNEL=$cand; break
      fi
    done
  fi
  [ -n "$TUNNEL" ] || die "no reachable container->host ssh target; pass --tunnel-target user@host:port"
  log "tunnel target: $TUNNEL"
  gen_restart "$TUNNEL"
  push_files restart-dsh.sh
  r "docker exec $CONTAINER /bin/bash /root/restart-dsh.sh"
  URL=$(wait_url) || die "dsh web URL missing after tunnel restart"
fi

# ---- patch blocks (raw text) + compose validation ----
r "docker exec $CONTAINER bash -c 'grep -q \"^- id: llm-pi-ai\" /root/.dsh/profiles/web/cordis.patch.yml || cat /root/patch-blocks.yml >> /root/.dsh/profiles/web/cordis.patch.yml'"
r "docker exec $CONTAINER bash -c 'grep -q \"^- id: sandbox-policy\" /root/.dsh/profiles/web/cordis.patch.yml || cat /root/permission-blocks.yml >> /root/.dsh/profiles/web/cordis.patch.yml'"
r "docker exec $CONTAINER bash -lc 'dsh --profile web --dump-config >/dev/null && echo COMPOSE_OK'"
# permission patch newly added => restart to load it (cordis config is not hot-reloaded)
if [ "$PERM_PRESENT" = 0 ]; then
  log "permission patch newly applied; restarting dsh to load it"
  r "docker exec $CONTAINER /bin/bash /root/restart-dsh.sh" || true
  URL=$(wait_url) || die "dsh URL missing after permission restart"
fi
# pin the on-disk restart script to the determined tunnel so reboot self-heal works
gen_restart "$TUNNEL"
push_files restart-dsh.sh boot.sh

if [ "$WITH_PLUGINS" = 1 ]; then
  r "docker exec $CONTAINER bash -lc 'dsh plugin --profile web add @linxin666/dsh-web-all@latest || { sed -i \"s/set this to true or false/true/\" /root/.dsh/profiles/web/pnpm-workspace.yaml; dsh plugin --profile web add @linxin666/dsh-web-all@latest; }'" || log "WARN: plugin add failed (optional)"
fi

# ---- commit + rebuild with boot.sh CMD (zero-human reboot self-heal) ----
log "committing dsh-base:latest (CMD boot.sh)"
r "docker commit --change 'CMD [\"/bin/bash\",\"/root/boot.sh\"]' $CONTAINER dsh-base:latest >/dev/null"

INSPECT=$(flag_gaps)
NEED_REBUILD=0
grep -q 'boot.sh' <<<"$INSPECT" || NEED_REBUILD=1
if [ "$NEED_REBUILD" = 1 ]; then
  log "rebuilding container with boot.sh CMD (reboot self-heal)"
  # No --cap-add SYS_ADMIN / seccomp overrides: managed ML platforms REJECT
  # SYS_ADMIN and FORCE-inject seccomp/apparmor (probed on huoshan_8). Zero-human
  # sandbox comes from dsh permission mode, not container security flags.
  r "docker stop $CONTAINER >/dev/null && docker rm $CONTAINER >/dev/null"
  # shellcheck disable=SC2086
  r "docker run -d --name $CONTAINER --restart unless-stopped $BIND_ARGS $DEV_ARGS dsh-base:latest" \
    || die "docker run failed; inspect platform docker wrapper"
  # the committed image carries the OLD dsh-web.log; boot.sh truncates it on
  # start. Wait past the restart script's own sleeps (2+12+2) plus dsh boot
  # so wait_url cannot match the stale token from the image.
  sleep 20
  URL=$(wait_url) || die "boot.sh did not bring dsh up after rebuild"
else
  log "container CMD already boot.sh; keeping it (convergent no-op)"
fi

# ---- zero-human proof: permission mode (not userns) is the guarantee ----
if [ "$PERMISSION_MODE" = danger-full-access ]; then
  log "permission: danger-full-access => approval=never, bwrap skipped -> zero-human"
  if r "docker exec $CONTAINER unshare --user --pid --fork true" 2>/dev/null; then
    log "note: userns also available (workspace-write would work too)"
  else
    log "note: platform seccomp blocks unshare(CLONE_NEWUSER); danger-full-access is the zero-human path (boundary in references/remote-bootstrap.md)"
  fi
else
  if r "docker exec $CONTAINER unshare --user --pid --fork true" 2>/dev/null; then
    log "sandbox: unshare OK -> $PERMISSION_MODE available, no human click"
  else
    log "WARN: unshare EPERM (platform forces seccomp) + mode=$PERMISSION_MODE => per-session 'full permissions' click remains. Use --permission-mode danger-full-access for zero-human."
  fi
fi

# ---- local forward + verify ----
bash "$SKILL_DIR/scripts/local-forward.sh" "$ALIAS" --local-port "$LOCAL_PORT" --remote-port "$REMOTE_PORT" --container "$CONTAINER"
bash "$SKILL_DIR/scripts/verify-remote.sh" "$ALIAS" --local-port "$LOCAL_PORT" --remote-port "$REMOTE_PORT" --container "$CONTAINER" --permission-mode "$PERMISSION_MODE"
log "DONE: open the printed URL; agent native context = $ALIAS"