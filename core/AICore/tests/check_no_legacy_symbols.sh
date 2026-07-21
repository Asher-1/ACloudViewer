#!/usr/bin/env bash
# Fail if libAICore still exports pre-unification C API symbol names.
set -euo pipefail
lib="${1:?usage: check_no_legacy_symbols.sh /path/to/libAICore.so}"
if [[ ! -f "$lib" ]]; then
  echo "missing library: $lib" >&2
  exit 1
fi
if nm -D --defined-only "$lib" 2>/dev/null | rg -q ' (da_capi_|fs_capi_)'; then
  echo "legacy symbols found in $lib:" >&2
  nm -D --defined-only "$lib" | rg ' (da_capi_|fs_capi_)' >&2 || true
  exit 1
fi
if nm -D --defined-only "$lib" 2>/dev/null | rg -q ' T aicore_(depth|gaussian)_'; then
  echo "canonical aicore C API symbols present"
fi
echo "no legacy exported symbols in $lib"
exit 0
