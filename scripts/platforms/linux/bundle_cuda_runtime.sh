#!/usr/bin/env bash
# Bundle NVIDIA CUDA runtime libraries required by libggml-cuda for driver-only
# deployment. libcuda.so.* is never copied (provided by the NVIDIA driver).
#
# Usage: bundle_cuda_runtime.sh <libggml-cuda.so> <dest_dir> [extra_lib_dirs_colon_separated]

set -euo pipefail

GGML_CUDA_MODULE="${1:?libggml-cuda module path required}"
DEST_DIR="${2:?destination directory required}"
EXTRA_LIB_DIRS="${3:-}"

if [ ! -f "$GGML_CUDA_MODULE" ]; then
    echo "Error: ggml CUDA module not found: $GGML_CUDA_MODULE" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

declare -A COPIED=()

should_bundle_name() {
    local base="$1"
    case "$base" in
        libcudart.so*|libcublas.so*|libcublasLt.so*|libnvrtc.so*|\
        libnvJitLink.so*|libculibos.so*)
            return 0
            ;;
    esac
    return 1
}

is_driver_lib() {
    local base="$1"
    case "$base" in
        libcuda.so*) return 0 ;;
    esac
    return 1
}

find_library_in_paths() {
    local lib_name="$1"
    local search_paths="$2"

    IFS=':' read -ra PATHS <<< "$search_paths"
    for path in "${PATHS[@]}"; do
        [ -n "$path" ] || continue
        [ -d "$path" ] || continue
        if [ -f "$path/$lib_name" ]; then
            echo "$path/$lib_name"
            return 0
        fi
    done
    return 1
}

copy_real_file() {
    local src_path="$1"
    local dest_dir="$2"
    local dest_name
    dest_name="$(basename "$src_path")"

    if [ -L "$src_path" ]; then
        src_path="$(readlink -f "$src_path")"
    fi
    if [ ! -f "$src_path" ]; then
        echo "Warning: missing CUDA runtime file: $src_path" >&2
        return 1
    fi

    if [ "${COPIED[$dest_name]:-}" = "1" ]; then
        return 0
    fi

    echo "Bundling CUDA runtime: $src_path -> $dest_dir/$dest_name" >&2
    cp -f "$src_path" "$dest_dir/$dest_name"
    COPIED[$dest_name]=1
    return 0
}

resolve_and_copy() {
    local lib_ref="$1"
    local base_name
    base_name="$(basename "$lib_ref")"

    if is_driver_lib "$base_name"; then
        return 0
    fi
    if ! should_bundle_name "$base_name"; then
        return 0
    fi

    local resolved="$lib_ref"
    if [ ! -f "$resolved" ]; then
        local found
        found="$(find_library_in_paths "$base_name" "$EXTRA_LIB_DIRS")" || true
        if [ -n "$found" ]; then
            resolved="$found"
        else
            echo "Warning: could not locate $base_name for bundling" >&2
            return 1
        fi
    fi

    local copied_path="$DEST_DIR/$base_name"
    copy_real_file "$resolved" "$DEST_DIR" || return 1

    # Recurse into bundled runtime dependencies (still whitelisted).
    local dep dep_base
    while IFS= read -r dep; do
        dep_base="$(basename "$dep")"
        if should_bundle_name "$dep_base" && [ "${COPIED[$dep_base]:-}" != "1" ]; then
            resolve_and_copy "$dep"
        fi
    done < <(ldd "$copied_path" 2>/dev/null | awk '/=>/ {print $3}' | grep -E '^/')
}

echo "Scanning ggml CUDA module: $GGML_CUDA_MODULE"
while IFS= read -r lib_ref; do
    [ -n "$lib_ref" ] || continue
    resolve_and_copy "$lib_ref"
done < <(ldd "$GGML_CUDA_MODULE" 2>/dev/null | awk '/=>/ {print $3}' | grep -E '^/')

if [ "${#COPIED[@]}" -eq 0 ]; then
    echo "Error: no CUDA runtime libraries were bundled (check build CUDA toolkit paths)" >&2
    exit 1
fi

echo "Bundled ${#COPIED[@]} CUDA runtime libraries into $DEST_DIR"
