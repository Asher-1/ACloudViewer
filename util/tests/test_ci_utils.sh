#!/usr/bin/env bash
# Unit tests for the pure contract layer of util/ci_utils.sh — the bash
# counterpart of util/tests/test_ci_utils.ps1. Covered: strict-mode
# preservation when sourced by a non-interactive script (the contract whose
# violation once let a failed wheel build continue), the
# CLOUDVIEWER_CMAKE_BIN guard, the AICore USE_* option resolver family and
# the python_release_deps_required gate. Build/install/docker helpers are
# out of scope: they need toolchains, network or a build tree.
#
# Run locally or on CI (needs only bash):
#   bash util/tests/test_ci_utils.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$REPO_ROOT/util/ci_utils.sh"

if [[ ! -f "$LIB" ]]; then
    echo "FAIL: util/ci_utils.sh not found at $LIB" >&2
    exit 1
fi

PASS=0
FAIL=0

# Every case runs in a subshell: ci_utils.sh mutates the environment
# (CONDA_PREFIX, NPROC, AICore_*, PATH) as a sourcing side effect.
assert_ok() { # assert_ok <description> <command...>
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        PASS=$((PASS + 1)); echo "PASS: $desc"
    else
        FAIL=$((FAIL + 1)); echo "FAIL: $desc"
    fi
}

assert_ko() { # assert_ko <description> <command...>  (command must FAIL)
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        FAIL=$((FAIL + 1)); echo "FAIL: $desc"
    else
        PASS=$((PASS + 1)); echo "PASS: $desc"
    fi
}

# --- 1. Direct execution: strict-mode branch, no side effects, exit 0 ---
assert_ok "direct execution exits 0 (strict mode branch)" \
    bash "$LIB"

# --- 2. Sourcing must preserve a non-interactive caller's strict mode ---
# The incident contract (see ci_utils.sh header): the old unconditional
# `set +e +u +o pipefail` here silently disabled `set -euo pipefail`
# declared by docker/build_cloudviewer_whl.sh, letting a failed wheel build
# continue and ship broken wheels. `false` must therefore abort the caller.
assert_ko "sourcing preserves caller set -euo pipefail (false aborts)" \
    bash -c "set -euo pipefail; source '$LIB'; false; echo UNREACHABLE"
assert_ko "sourcing preserves caller set -u (unset var aborts)" \
    bash -c "set -euo pipefail; source '$LIB'; echo \"\$NO_SUCH_VAR\""

# --- 3. CLOUDVIEWER_CMAKE_BIN guard ---
assert_ko "CLOUDVIEWER_CMAKE_BIN pointing at a missing binary aborts direct execution" \
    env CLOUDVIEWER_CMAKE_BIN="$LIB/no-such-cmake" bash "$LIB"

_scratch="$(mktemp -d)"
touch "$_scratch/cmake"; chmod +x "$_scratch/cmake"
# CI sets the variable through the environment (GITHUB_ENV), so the case
# exports it instead of relying on assignment-prefix persistence for `source`.
assert_ok "CLOUDVIEWER_CMAKE_BIN valid binary prepends its dir to PATH" \
    bash -c "export CLOUDVIEWER_CMAKE_BIN='$_scratch/cmake'; source '$LIB' && [[ \":\$PATH:\" == *\":$_scratch:\"* ]]"
rm -rf "$_scratch"

# --- 4. _ci_env_on_off truthy table ---
assert_ok "_ci_env_on_off accepts ON/on/1/true/TRUE/yes/YES and rejects the rest" \
    bash -c "source '$LIB' &&
        [[ \"\$(_ci_env_on_off ON)\" == ON && \"\$(_ci_env_on_off on)\" == ON &&
            \"\$(_ci_env_on_off 1)\" == ON && \"\$(_ci_env_on_off true)\" == ON &&
            \"\$(_ci_env_on_off TRUE)\" == ON && \"\$(_ci_env_on_off yes)\" == ON &&
            \"\$(_ci_env_on_off YES)\" == ON &&
            \"\$(_ci_env_on_off OFF)\" == OFF && \"\$(_ci_env_on_off off)\" == OFF &&
            \"\$(_ci_env_on_off garbage)\" == OFF && \"\$(_ci_env_on_off '')\" == OFF ]]"

# --- 5. AICore USE_* option resolver family ---
# Resolution order everywhere: environment override > option token > default.
# Option tokens are pipe-separated by the callers (same convention as
# install_python_dependencies).

# Vulkan: without_vulkan -> OFF, with_vulkan -> ON, otherwise platform
# default (ON on Linux, OFF on macOS — asserted for Linux here).
assert_ok "_ci_resolve_aicore_use_vulkan: option tokens and Linux default" \
    bash -c "source '$LIB' &&
        [[ \"\$(_ci_resolve_aicore_use_vulkan 'with_conda|without_vulkan')\" == OFF &&
            \"\$(_ci_resolve_aicore_use_vulkan 'with_conda|with_vulkan')\" == ON &&
            \"\$(_ci_resolve_aicore_use_vulkan 'with_conda')\" == ON ]]"
assert_ok "_ci_resolve_aicore_use_vulkan: environment overrides the token" \
    bash -c "source '$LIB' &&
        AICore_USE_VULKAN=OFF &&
        [[ \"\$(_ci_resolve_aicore_use_vulkan 'with_conda|with_vulkan')\" == OFF ]]"

assert_ok "_ci_resolve_aicore_use_cuda: with_aicore_cuda token, default OFF" \
    bash -c "source '$LIB' &&
        [[ \"\$(_ci_resolve_aicore_use_cuda 'with_conda|with_aicore_cuda')\" == ON &&
            \"\$(_ci_resolve_aicore_use_cuda 'with_conda')\" == OFF ]]"
assert_ok "_ci_resolve_aicore_use_cuda: environment overrides the token" \
    bash -c "source '$LIB' &&
        AICore_USE_CUDA=ON &&
        [[ \"\$(_ci_resolve_aicore_use_cuda 'with_conda')\" == ON ]]"

assert_ok "_ci_resolve_aicore_bundle_cuda_runtime: token and default OFF" \
    bash -c "source '$LIB' &&
        [[ \"\$(_ci_resolve_aicore_bundle_cuda_runtime 'bundle_cuda_runtime')\" == ON &&
            \"\$(_ci_resolve_aicore_bundle_cuda_runtime 'with_conda')\" == OFF ]]"
assert_ok "_ci_resolve_aicore_bundle_cuda_runtime: environment overrides the token" \
    bash -c "source '$LIB' &&
        AICore_BUNDLE_CUDA_RUNTIME=OFF &&
        [[ \"\$(_ci_resolve_aicore_bundle_cuda_runtime 'bundle_cuda_runtime')\" == OFF ]]"

assert_ok "_ci_resolve_aicore_cpu_all_variants: no_cpu_all_variants token, default ON" \
    bash -c "source '$LIB' &&
        [[ \"\$(_ci_resolve_aicore_cpu_all_variants 'no_cpu_all_variants')\" == OFF &&
            \"\$(_ci_resolve_aicore_cpu_all_variants 'with_conda')\" == ON ]]"
assert_ok "_ci_resolve_aicore_cpu_all_variants: environment overrides the token" \
    bash -c "source '$LIB' &&
        AICore_CPU_ALL_VARIANTS=OFF &&
        [[ \"\$(_ci_resolve_aicore_cpu_all_variants 'with_conda')\" == OFF ]]"

# --- 6. python_release_deps_required gate ---
# Defaults: PLUGIN_PYTHON=ON, BUILD_PYTHON_MODULE=ON, COPY_ENV=OFF,
# COPY_MINIMAL_ENV=ON -> required.
assert_ok "python_release_deps_required: defaults require the release deps" \
    bash -c "source '$LIB' && unset PLUGIN_PYTHON BUILD_PYTHON_MODULE \
        PLUGIN_PYTHON_COPY_ENV PLUGIN_PYTHON_COPY_MINIMAL_ENV \
        && python_release_deps_required"
assert_ko "python_release_deps_required: PLUGIN_PYTHON=OFF skips" \
    bash -c "source '$LIB' && PLUGIN_PYTHON=OFF python_release_deps_required"
assert_ko "python_release_deps_required: BUILD_PYTHON_MODULE=OFF skips" \
    bash -c "source '$LIB' && BUILD_PYTHON_MODULE=OFF python_release_deps_required"
assert_ko "python_release_deps_required: copied env skips" \
    bash -c "source '$LIB' && PLUGIN_PYTHON_COPY_ENV=ON python_release_deps_required"
assert_ko "python_release_deps_required: non-minimal copied env skips" \
    bash -c "source '$LIB' && PLUGIN_PYTHON_COPY_MINIMAL_ENV=OFF python_release_deps_required"

# --- Summary ---
echo ""
echo "test_ci_utils.sh: $PASS passed, $FAIL failed"
if [[ "$FAIL" -ne 0 ]]; then
    exit 1
fi
exit 0
