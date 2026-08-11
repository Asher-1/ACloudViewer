#!/usr/bin/env bash
# Fail if libAICore still exports pre-unification C API symbol names.
set -euo pipefail
lib="${1:?usage: check_no_legacy_symbols.sh /path/to/libAICore.so [public-include-dir]}"
public_include="${2:-}"
if [[ ! -f "$lib" ]]; then
  echo "missing library: $lib" >&2
  exit 1
fi
if nm -D --defined-only "$lib" 2>/dev/null | rg -q ' (da_capi_|fs_capi_)'; then
  echo "legacy symbols found in $lib:" >&2
  nm -D --defined-only "$lib" | rg ' (da_capi_|fs_capi_)' >&2 || true
  exit 1
fi

unexpected="$({ nm -D --defined-only "$lib" 2>/dev/null || true; } \
  | awk '$2 ~ /^[TDBRWV]$/ { print $3 }' \
  | rg -v '^(aicore_|_ZN6aicore5depth10ImageDepth)' || true)"
if [[ -n "$unexpected" ]]; then
  echo "unexpected public symbols found in $lib:" >&2
  echo "$unexpected" >&2
  exit 1
fi

if [[ -n "$public_include" ]] && rg -n '(^|[/<"])(ggml|gguf)([-_.>"/]|$)' "$public_include"; then
  echo "ggml implementation detail leaked through public AICore headers" >&2
  exit 1
fi

echo "AICore exports only its public ABI and public headers do not expose ggml"
exit 0
