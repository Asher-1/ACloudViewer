#!/bin/bash

# options="$(echo "$@" | tr ' ' '|')"
# if [[ "skip_libs" =~ ^($options)$ ]]; then
#     exclude_libs="libc.so.* libdl.so.* libselinux.so.* libpthread.so.* librt.so.* ld-linux-x86-64.so.* libdrm.so.* libm.so.* libdrm_intel.so.* libstdc++.so.*"
# else
#     exclude_libs=""
# fi
exclude_libs="libc.so.* libdl.so.* libselinux.so.* libpthread.so.* librt.so.* ld-linux-x86-64.so.* libdrm.so.* libm.so.* libdrm_intel.so.* libstdc++.so.*"

if [ -n "$3" ]; then
    EXTRA_LIB_PATH=$3
else
    EXTRA_LIB_PATH=""
fi


copy_real_file_with_symlink_name() {
    local symlink_path=$1
    local target_dir=$2
    local symlink_name=$(basename "$symlink_path")

    local resolved_target=$(readlink -f "$symlink_path")
    if [ -f "$resolved_target" ]; then
        echo "Copying REAL file: $resolved_target as $symlink_name to $target_dir"
        cp -f "$resolved_target" "$target_dir/$symlink_name"
    else
        local alt_path="$EXTRA_LIB_PATH/$(basename "$symlink_path")"
        if [ -f "$alt_path" ]; then
            local alt_resolved=$(readlink -f "$alt_path")
            cp -f "$alt_resolved" "$target_dir/$symlink_name"
            echo "Copied from alt: $alt_resolved as $symlink_name"
        else
            echo "Error: Target not found for symlink $symlink_path"
        fi
    fi
}

process_target() {
    local Target=$1
    local LibDir=$2

    not_found_libs=($(ldd "$Target" | grep "not found" | awk '{print $1}'))
    found_libs=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))
    lib_array=("${not_found_libs[@]}" "${found_libs[@]}")

    for Variable in ${lib_array[@]}
    do
        base_name=$(basename $Variable)
        sub_name=${base_name:0:7}
        if [[ ! "${exclude_libs[@]}" =~ "$sub_name" ]]; then
            if [ -L "$Variable" ]; then
                echo "Add soft dependence: $Variable to $LibDir"
                copy_real_file_with_symlink_name "$Variable" "$LibDir"
            elif [ -f "$Variable" ]; then
                echo "Add dependence: $Variable to $LibDir"
                cp "$Variable" "$LibDir"
            else
                alternative_path="$EXTRA_LIB_PATH/$Variable"

                if [ -f "$alternative_path" ]; then
                    echo "File not found at $Variable. Found at $alternative_path. Copying to $LibDir"
                    [ -L "$alternative_path" ] && copy_real_file_with_symlink_name "$alternative_path" "$LibDir" || cp -f "$alternative_path" "$LibDir"
                else
                    echo "Warning: $Variable (and alternative $alternative_path) are not regular files or do not exist."
                fi
            fi
        else
            echo "+++++++++++++Ignore library: $Variable"
        fi
    done
}


if [ -d $1 ]; then
    LibDir=$2
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
    process_target $Target $LibDir
fi
