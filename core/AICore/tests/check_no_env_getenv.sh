#!/usr/bin/env bash
# Fail if AICore source reads or writes process environment variables outside
# the two sanctioned files:
#   src/common/data_root_util.cpp  - deployment data-root convention
#   src/common/ggml_env_bridge.cpp - the ONLY writer of ggml-side variables
#                                     (explicit options -> env, single point)
# All logic control must flow through explicit options/APIs (see the AICore
# unification plan): a getenv/setenv call in a task or common module is a
# regression.
set -euo pipefail
src_dir="${1:?usage: check_no_env_getenv.sh /path/to/AICore/src}"
if [[ ! -d "$src_dir" ]]; then
    echo "missing src dir: $src_dir" >&2
    exit 1
fi

# White-list as a relative path suffix so the check survives any prefix.
whitelist=(
    "common/data_root_util.cpp"
    "common/ggml_env_bridge.cpp"
)

# Match actual calls (not comments/doc mentions).
pattern='(^|[^[:alnum:]_])(getenv|secure_getenv|setenv|unsetenv|putenv|_putenv_s)[[:space:]]*\('

violations=0
while IFS= read -r file; do
    rel="${file#"$src_dir"/}"
    allowed=0
    for w in "${whitelist[@]}"; do
        if [[ "$rel" == "$w" ]]; then
            allowed=1
            break
        fi
    done
    [[ "$allowed" == 1 ]] && continue
    if grep -nE "$pattern" "$file"; then
        echo "  ^-- env access outside the whitelist ($rel)" >&2
        violations=1
    fi
done < <(find "$src_dir" -type f \( -name '*.cpp' -o -name '*.c' \
    -o -name '*.hpp' -o -name '*.h' \) | sort)

if [[ "$violations" != 0 ]]; then
    echo "FAIL: environment access is limited to $src_dir/common/{data_root_util.cpp, ggml_env_bridge.cpp}" >&2
    exit 1
fi

echo "AICore reads/writes environment only through the sanctioned bridge"
exit 0
