#!/usr/bin/env bash
# Path-hygiene gate for app/reconstruction.
#
# The COLMAP fork has migrated its core layer (Options members, file.h,
# mvs/controllers entry points) to std::filesystem::path. The reconstruction
# widgets were aligned to match: path-typed members/locals, path operator/
# for joining, and explicit .string()/toUtf8() bridges only at the Qt and
# cloudViewer-string boundaries.
#
# MSVC rejects the implicit path -> std::string conversion that POSIX
# compilers accept, so a regression here compiles green on Linux CI and only
# explodes on Windows (C2679/C2664, historically the OCIOYaml-class wheel
# breakers). This gate fails fast on the known regression shapes instead.
#
# Usage: check_fs_path_hygiene.sh [app/reconstruction dir]
# Exit code 0 iff no regression pattern is found.

set -euo pipefail

DIR="${1:-$(dirname "$0")}"
FAILURES=0

fail() {
    echo "FAIL: $1" >&2
    FAILURES=$((FAILURES + 1))
}

# 1. colmap::JoinPaths (boost append semantics, string-join) must not be
#    re-introduced in the UI layer. Path joining uses the operator/ idiom;
#    the helper itself remains available to the string-domain callers inside
#    libs/Reconstruction (exe tools, controllers, mvs internals).
if grep -rn "JoinPaths(" "$DIR" --include='*.cpp' --include='*.h' >/dev/null 2>&1; then
    grep -rn "JoinPaths(" "$DIR" --include='*.cpp' --include='*.h' | head -10 >&2
    fail "JoinPaths usage found in $DIR - use std::filesystem::path operator/ instead"
fi

# 2. Path-typed widget members must be std::filesystem::path, not
#    std::string (the mixed state is what produced the MSVC-only errors).
while IFS= read -r line; do
    fail "string-typed path member: $line"
done < <(grep -rnE 'std::string[[:space:]]+[a-z_]*(path|dir|folder)[a-z_]*[[:space:]]*;' \
    "$DIR" --include='*.h' 2>/dev/null || true)

if [ "$FAILURES" -ne 0 ]; then
    echo "check_fs_path_hygiene: $FAILURES failure(s)" >&2
    exit 1
fi
echo "check_fs_path_hygiene: OK ($DIR)"
