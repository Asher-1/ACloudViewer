#!/usr/bin/env bash
# Bundle NVIDIA CUDA runtime libraries required by a CUDA module for driver-only
# deployment. libcuda.so.* is never copied (provided by the NVIDIA driver).
#
# Usage: bundle_cuda_runtime.sh <module.so> <dest_dir> [extra_lib_dirs_colon_separated] [ggml|torch]

set -euo pipefail

GGML_CUDA_MODULE="${1:?libggml-cuda module path required}"
DEST_DIR="${2:?destination directory required}"
EXTRA_LIB_DIRS="${3:-}"
RUNTIME_PROFILE="${4:-ggml}"
case "$RUNTIME_PROFILE" in
    ggml|torch) ;;
    *) echo "Error: unknown CUDA runtime profile: $RUNTIME_PROFILE" >&2; exit 1 ;;
esac

if [ ! -f "$GGML_CUDA_MODULE" ]; then
    echo "Error: ggml CUDA module not found: $GGML_CUDA_MODULE" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

declare -A COPIED=()
BUNDLE_FAILED=0

should_bundle_name() {
    local base="$1"
    case "$base" in
        libcudart.so*|libcublas.so*|libcublasLt.so*|libnvrtc.so*|\
        libnvJitLink.so*|libculibos.so*)
            return 0
            ;;
    esac
    if [ "$RUNTIME_PROFILE" = torch ]; then
        case "$base" in
            libcusparse*.so*|libcusolver*.so*|libcufft.so*|libcurand.so*|\
            libcudnn*.so*|libnccl.so*|libcufile.so*|libcupti.so*|libnvToolsExt.so*)
                return 0
                ;;
        esac
    fi
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

    # Reinstalling into the same prefix can resolve a dependency from there.
    if [ "$src_path" -ef "$dest_dir/$dest_name" ]; then
        COPIED[$dest_name]=1
        return 0
    fi

    echo "Bundling CUDA runtime: $src_path -> $dest_dir/$dest_name" >&2
    cp -f "$src_path" "$dest_dir/$dest_name" || return 1
    COPIED[$dest_name]=1
    return 0
}

library_refs() {
    # ldd appends a load address to resolved paths. Strip only that suffix,
    # retaining spaces in paths and unresolved sonames for the search fallback.
    ldd "$1" 2>/dev/null | sed -n -E \
        -e 's/[[:space:]]+\(0x[[:xdigit:]]+\)[[:space:]]*$//' \
        -e 's/^[[:space:]]*([A-Za-z0-9._+-]+\.so[^ ]*)[[:space:]]*=>[[:space:]]*(\/.*)$/\2/p' \
        -e 's/^[[:space:]]*([A-Za-z0-9._+-]+\.so[^ ]*)[[:space:]]*=>[[:space:]]*not found[[:space:]]*$/\1/p'
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
    if [ "${COPIED[$base_name]:-}" = "1" ]; then
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
            resolve_and_copy "$dep" || return 1
        fi
    done < <(library_refs "$copied_path")
    return 0
}

echo "Scanning CUDA module ($RUNTIME_PROFILE): $GGML_CUDA_MODULE"
# Torch wheels resolve CUDA dependencies through package-relative RUNPATHs.
# Retain those resolved directories before flattening files into the bundle,
# so recursive checks can still locate their transitive dependencies.
mapfile -t MODULE_REFS < <(library_refs "$GGML_CUDA_MODULE")
for lib_ref in "${MODULE_REFS[@]}"; do
    if [ -f "$lib_ref" ]; then
        EXTRA_LIB_DIRS="${EXTRA_LIB_DIRS:+$EXTRA_LIB_DIRS:}$(dirname "$lib_ref")"
    fi
done
# Feed BOTH resolved paths and "not found" bare sonames into resolve_and_copy:
# when the CUDA toolkit lib dir is not registered in ldconfig, ldd reports
# "not found" for libcudart/libcublas even though EXTRA_LIB_DIRS can locate
# them. The old loop dropped those lines, so nothing was ever bundled.
while IFS= read -r lib_ref; do
    [ -n "$lib_ref" ] || continue
    resolve_and_copy "$lib_ref" || BUNDLE_FAILED=1
done < <(printf '%s\n' "${MODULE_REFS[@]}")

if [ "$BUNDLE_FAILED" -ne 0 ]; then
    echo "Error: CUDA runtime dependency closure is incomplete" >&2
    exit 1
fi
if [ "${#COPIED[@]}" -eq 0 ]; then
    echo "Error: no CUDA runtime libraries were bundled (check build CUDA toolkit paths)" >&2
    exit 1
fi

echo "Bundled ${#COPIED[@]} CUDA runtime libraries into $DEST_DIR"
