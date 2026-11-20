#!/usr/bin/env bash
# verify-remote.sh — the Path A verification triple + sandbox/liveness checks.
# Usage: verify-remote.sh <ssh-alias> [--local-port 3180] [--remote-port 3080]
#        [--container dsh-remote | --native]
# PASS criteria: agent native id=0/remote hostname; GPU visible in exec context;
# unshare user-ns OK (zero-human sandbox); 0 activate failures; tunnel alive
# (when needed); local forward serves the token URL.
set -euo pipefail
ALIAS=${1:?usage: verify-remote.sh <ssh-alias> [opts]}
shift || true
LOCAL_PORT=3180 REMOTE_PORT=3080 CONTAINER=dsh-remote NATIVE=0 PERMISSION_MODE=danger-full-access
while [ $# -gt 0 ]; do
  case "$1" in
    --local-port) LOCAL_PORT=$2; shift ;;
    --remote-port) REMOTE_PORT=$2; shift ;;
    --container) CONTAINER=$2; shift ;;
    --native) NATIVE=1 ;;
    --permission-mode) PERMISSION_MODE=$2; shift ;;
  esac
  shift
done
FAIL=0
check() { # $1=label $2=ok(0/1) $3=evidence
  if [ "$2" = 0 ]; then echo "PASS $1: $3"; else echo "FAIL $1: $3"; FAIL=1; fi
}
rexec() { if [ "$NATIVE" = 1 ]; then ssh "$ALIAS" "bash -lc '$1'"; else ssh "$ALIAS" "docker exec $CONTAINER bash -lc '$1'"; fi; }

IDOUT=$(rexec 'id -u; hostname' 2>/dev/null || true)
RHOST=$(ssh "$ALIAS" hostname 2>/dev/null || true)
check "native-context" "$([ -n "$IDOUT" ] && [ "$(tail -1 <<<"$IDOUT")" != "$(hostname)" ] && echo 0 || echo 1)" "$(tr '\n' ' ' <<<"$IDOUT") (remote=$RHOST)"

HOSTGPU=$(ssh "$ALIAS" '[ -e /dev/nvidia0 ] && echo yes || echo no' 2>/dev/null || echo no)
GPUOUT=$(rexec 'nvidia-smi -L 2>/dev/null | wc -l' || echo 0)
if [ "$HOSTGPU" = yes ]; then
  check "gpu" "$([ "$GPUOUT" -gt 0 ] && echo 0 || echo 1)" "$GPUOUT GPU(s) in exec context"
else
  check "gpu" 0 "host has no GPU; passthrough not required"
fi

UNS=$(rexec 'unshare --user --pid --fork true >/dev/null 2>&1 && echo ok || echo eperm' || echo eperm)
if [ "$PERMISSION_MODE" = danger-full-access ]; then
  # zero-human via full access: patch sets mode + approval=never (bwrap skipped).
  # userns is NOT required; report it as informational only.
  OK=$(rexec 'grep -q "mode: danger-full-access" ~/.dsh/profiles/web/cordis.patch.yml && grep -q "policy: never" ~/.dsh/profiles/web/cordis.patch.yml && echo 1 || echo 0' || echo 0)
  check "zero-human-permission" "$([ "$OK" = 1 ] && echo 0 || echo 1)" "danger-full-access + approval=never persisted (userns=$UNS, not required)"
else
  check "sandbox-zero-human" "$([ "$UNS" = ok ] && echo 0 || echo 1)" "mode=$PERMISSION_MODE needs unshare --user => $UNS"
fi

if [ "$NATIVE" = 1 ]; then
  LOG=$(ssh "$ALIAS" 'cat $HOME/dsh-web.log' 2>/dev/null || true)
else
  LOG=$(ssh "$ALIAS" "docker exec $CONTAINER cat /var/log/dsh-web.log" 2>/dev/null || true)
fi
AF=$(grep -c 'did not activate' <<<"$LOG" || true)
check "activate" "$([ "$AF" = 0 ] && echo 0 || echo 1)" "$AF bundle activate failures"
URL=$(grep -o "http://127.0.0.1:$REMOTE_PORT/?token=[A-Za-z0-9_-]*" <<<"$LOG" | head -1)
check "token-url" "$([ -n "$URL" ] && echo 0 || echo 1)" "${URL:-missing}"

CODE=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$LOCAL_PORT/" || echo 000)
check "forward-401" "$([ "$CODE" = 401 ] && echo 0 || echo 1)" "no-token HTTP $CODE (401 expected)"
if [ -n "$URL" ]; then
  TURL=${URL/127.0.0.1:$REMOTE_PORT/127.0.0.1:$LOCAL_PORT}
  CODE=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$TURL" || echo 000)
  check "forward-token" "$([ "$CODE" = 200 ] || [ "$CODE" = 303 ] && echo 0 || echo 1)" "token HTTP $CODE"
fi
exit $FAIL