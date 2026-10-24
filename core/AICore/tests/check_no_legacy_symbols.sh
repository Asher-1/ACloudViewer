#!/usr/bin/env bash
# Fail if libAICore still exports pre-unification C API symbol names.
# Uses only nm/awk/grep — no ripgrep: a missing external tool used to make
# every check here silently pass (rg not found -> non-zero pipe -> if-branch
# skipped), which is worse than no gate at all.
set -euo pipefail
lib="${1:?usage: check_no_legacy_symbols.sh /path/to/libAICore.so [public-include-dir]}"
public_include="${2:-}"
if [[ ! -f "$lib" ]]; then
  echo "missing library: $lib" >&2
  exit 1
fi

# Capture once: piping nm straight into grep -q risks SIGPIPE-killing nm
# under pipefail, which would flip a detected failure into a false pass.
symbols="$(nm -D --defined-only "$lib" 2>/dev/null || true)"

if printf '%s\n' "$symbols" | grep -qE ' (da_capi_|fs_capi_)'; then
  echo "legacy symbols found in $lib:" >&2
  printf '%s\n' "$symbols" | grep -E ' (da_capi_|fs_capi_)' >&2 || true
  exit 1
fi

unexpected="$(printf '%s\n' "$symbols" \
  | awk '$2 ~ /^[TDBRWV]$/ { print $3 }' \
  | grep -vE '^(aicore_|_ZN6aicore5depth10ImageDepth)' || true)"
if [[ -n "$unexpected" ]]; then
  echo "unexpected public symbols found in $lib:" >&2
  echo "$unexpected" >&2
  exit 1
fi

if [[ -n "$public_include" ]] \
   && grep -rnE '(^|[/<"])(ggml|gguf)([-_.>"/]|$)' "$public_include"; then
  echo "ggml implementation detail leaked through public AICore headers" >&2
  exit 1
fi

echo "AICore exports only its public ABI and public headers do not expose ggml"
exit 0
