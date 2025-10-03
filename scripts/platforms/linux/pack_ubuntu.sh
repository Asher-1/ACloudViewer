#!/bin/bash

# options="$(echo "$@" | tr ' ' '|')"
# if [[ "skip_libs" =~ ^($options)$ ]]; then
#     exclude_libs="libc.so.* libdl.so.* libselinux.so.* libpthread.so.* librt.so.* ld-linux-x86-64.so.* libdrm.so.* libm.so.* libdrm_intel.so.* libstdc++.so.*"
# else
#     exclude_libs=""
# fi
exclude_libs="libc.so.* libdl.so.* libselinux.so.* libpthread.so.* librt.so.* ld-linux-x86-64.so.* libdrm.so.* libm.so.* libdrm_intel.so.* libstdc++.so.*"

# Support multiple extra library paths separated by colon
# Convert to absolute paths
if [ -n "$3" ]; then
    EXTRA_LIB_PATHS=""
    IFS=':' read -ra INPUT_PATHS <<< "$3"
    for path in "${INPUT_PATHS[@]}"; do
        if [ -d "$path" ]; then
            abs_path=$(cd "$path" && pwd)
            if [ -n "$EXTRA_LIB_PATHS" ]; then
                EXTRA_LIB_PATHS="$EXTRA_LIB_PATHS:$abs_path"
            else
                EXTRA_LIB_PATHS="$abs_path"
            fi
        else
            echo "Warning: Library path $path does not exist, skipping."
        fi
    done
else
    EXTRA_LIB_PATHS=""
fi

# Array to track processed libraries to avoid infinite recursion
declare -A PROCESSED_LIBS


find_library_in_paths() {
    local lib_name=$1
    local search_paths=$2
    
    # Try each search path
    IFS=':' read -ra PATHS <<< "$search_paths"
    for path in "${PATHS[@]}"; do
        if [ -f "$path/$lib_name" ]; then
            echo "$path/$lib_name"
            return 0
        fi
    done
    return 1
}

copy_real_file_with_symlink_name() {
    local symlink_path=$1
    local target_dir=$2
    local symlink_name=$(basename "$symlink_path")

    local resolved_target=$(readlink -f "$symlink_path")
    if [ -f "$resolved_target" ]; then
        echo "Copying REAL file: $resolved_target as $symlink_name to $target_dir"
        cp -f "$resolved_target" "$target_dir/$symlink_name"
        return 0
    else
        # Try to find in extra library paths
        local found_lib=$(find_library_in_paths "$symlink_name" "$EXTRA_LIB_PATHS")
        if [ -n "$found_lib" ]; then
            local alt_resolved=$(readlink -f "$found_lib")
            cp -f "$alt_resolved" "$target_dir/$symlink_name"
            echo "Copied from alt: $alt_resolved as $symlink_name"
            return 0
        else
            echo "Error: Target not found for symlink $symlink_path"
            return 1
        fi
    fi
}

process_target() {
    local Target=$1
    local LibDir=$2
    local Recursive=${3:-true}  # Default to recursive processing

    # Skip if target doesn't exist or is not executable/library
    if [ ! -f "$Target" ]; then
        echo "Warning: Target $Target does not exist, skipping."
        return
    fi

    # Get base name for tracking
    local target_basename=$(basename "$Target")
    
    # Skip if already processed (avoid infinite recursion)
    if [ "${PROCESSED_LIBS[$target_basename]}" = "1" ]; then
        return
    fi
    PROCESSED_LIBS[$target_basename]=1

    echo "Processing dependencies for: $Target"

    not_found_libs=($(ldd "$Target" 2>/dev/null | grep "not found" | awk '{print $1}'))
    found_libs=($(ldd "$Target" 2>/dev/null | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))
    lib_array=("${not_found_libs[@]}" "${found_libs[@]}")

    for Variable in ${lib_array[@]}
    do
        base_name=$(basename "$Variable")
        sub_name=${base_name:0:7}
        
        # Skip if already processed
        if [ "${PROCESSED_LIBS[$base_name]}" = "1" ]; then
            continue
        fi
        
        if [[ ! "${exclude_libs[@]}" =~ "$sub_name" ]]; then
            local copied_lib=""
            
            if [ -L "$Variable" ]; then
                echo "Add soft dependence: $Variable to $LibDir"
                if copy_real_file_with_symlink_name "$Variable" "$LibDir"; then
                    copied_lib="$LibDir/$base_name"
                fi
            elif [ -f "$Variable" ]; then
                echo "Add dependence: $Variable to $LibDir"
                cp "$Variable" "$LibDir"
                copied_lib="$LibDir/$base_name"
            else
                # Library not found, try to find it in extra paths
                local found_lib=$(find_library_in_paths "$base_name" "$EXTRA_LIB_PATHS")
                
                if [ -n "$found_lib" ]; then
                    echo "Library not found at $Variable. Found at $found_lib. Copying to $LibDir"
                    if [ -L "$found_lib" ]; then
                        copy_real_file_with_symlink_name "$found_lib" "$LibDir"
                    else
                        cp -f "$found_lib" "$LibDir"
                    fi
                    copied_lib="$LibDir/$base_name"
                else
                    echo "Warning: $Variable not found in any search paths."
                fi
            fi
            
            # Recursively process the copied library's dependencies
            if [ -n "$copied_lib" ] && [ -f "$copied_lib" ] && [ "$Recursive" = "true" ]; then
                process_target "$copied_lib" "$LibDir" true
            fi
        else
            echo "+++++++++++++Ignore library: $Variable"
        fi
    done
}


if [ -d $1 ]; then
    LibDir=$2
    # Add the target directory to extra library paths
    target_dir=$(readlink -f "$1")
    if [ -n "$EXTRA_LIB_PATHS" ]; then
        EXTRA_LIB_PATHS="$target_dir:$EXTRA_LIB_PATHS"
    else
        EXTRA_LIB_PATHS="$target_dir"
    fi
    
    echo "Using library search paths: $EXTRA_LIB_PATHS"
    
    for element in $(ls $1)
    do
        Target=$1"/"$element
        if [ -f $Target ]; then
            process_target $Target $LibDir
        fi
    done
else
    LibDir=$2
    Target=$1
    
    # Add the target's directory to extra library paths
    target_dir=$(dirname "$(readlink -f "$Target")")
    if [ -n "$EXTRA_LIB_PATHS" ]; then
        EXTRA_LIB_PATHS="$target_dir:$EXTRA_LIB_PATHS"
    else
        EXTRA_LIB_PATHS="$target_dir"
    fi
    
    echo "Using library search paths: $EXTRA_LIB_PATHS"
    
    process_target $Target $LibDir
fi
