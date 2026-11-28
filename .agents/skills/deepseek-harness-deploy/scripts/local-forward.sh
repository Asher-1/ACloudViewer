#!/usr/bin/env bash
# local-forward.sh — idempotent local ssh -L to the remote dsh web loopback,
# then print the FULL token URL (token regenerates on every remote boot).
# Usage: local-forward.sh <ssh-alias> [--local-port 3180] [--remote-port 3080]
#        [--container dsh-remote | --native]
set -euo pipefail
ALIAS=${1:?usage: local-forward.sh <ssh-alias> [--local-port N] [--remote-port N] [--container NAME|--native]}
shift || true
LOCAL_PORT=3180 REMOTE_PORT=3080 CONTAINER=dsh-remote NATIVE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --local-port) LOCAL_PORT=$2; shift ;;
    --remote-port) REMOTE_PORT=$2; shift ;;
    --container) CONTAINER=$2; shift ;;
    --native) NATIVE=1 ;;
  esac
  shift
done

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$LOCAL_PORT\$"; then
  echo "[forward] local port $LOCAL_PORT already listening; reusing"
else
  ssh -fN -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$ALIAS"
  echo "[forward] ssh -L $LOCAL_PORT:127.0.0.1:$REMOTE_PORT $ALIAS established"
fi

if [ "$NATIVE" = 1 ]; then
  URL=$(ssh "$ALIAS" "grep -o 'http://127.0.0.1:$REMOTE_PORT/?token=[A-Za-z0-9_-]*' \$HOME/dsh-web.log | head -1")
else
  URL=$(ssh "$ALIAS" "docker exec $CONTAINER grep -o 'http://127.0.0.1:$REMOTE_PORT/?token=[A-Za-z0-9_-]*' /var/log/dsh-web.log | head -1")
fi
[ -n "$URL" ] || { echo "[forward] FAIL: no token URL in remote log" >&2; exit 1; }
echo "[forward] OPEN: ${URL/127.0.0.1:$REMOTE_PORT/127.0.0.1:$LOCAL_PORT}"