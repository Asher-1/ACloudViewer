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
            if [ -f "$Variable" ]; then
                echo "Add dependence: $Variable to $LibDir"
                cp "$Variable" "$LibDir"
            else
                alternative_path="$EXTRA_LIB_PATH/$Variable"
                if [ -f "$alternative_path" ]; then
                    echo "File not found at $Variable. Found at $alternative_path. Copying to $LibDir"
                    cp "$alternative_path" "$LibDir"
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

# if [ -d $1 ]; then
#     LibDir=$2
#     for element in `ls $1`
#     do
#         Target=$1"/"$element
#         if [ -f $Target ]; then
#             # lib_array=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))
#             lib_array=($(ldd $Target | grep -E "/.*|not found" | grep -o "/.*/[^[:space:]]*"))

#             for Variable in ${lib_array[@]}
#             do
#                 base_name=$(basename $Variable)
#                 sub_name=${base_name:0:7}
#                 if [[ ! "${exclude_libs[@]}" =~ "$sub_name" ]]; then
#                     if [ -f "$Variable" ]; then
#                         echo "Add dependence: $Variable to $LibDir"
#                         cp "$Variable" "$LibDir"
#                     else
#                         alternative_path="$EXTRA_LIB_PATH/$base_name"
#                         if [ -f "$alternative_path" ]; then
#                             echo "File not found at $Variable. Found at $alternative_path. Copying to $LibDir"
#                             cp "$alternative_path" "$LibDir"
#                         else
#                             echo "Warning: $Variable and $alternative_path are not regular files or do not exist."
#                         fi
#                     fi
#                 else
#                     echo "+++++++++++++Ignore library: $Variable"
#                 fi
#             done
#         fi
#     done
# else
#     LibDir=$2
#     Target=$1
#     lib_array=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))

#     for Variable in ${lib_array[@]}
#     do
#         base_name=$(basename $Variable)
#         sub_name=${base_name:0:7}
#         if [[ ! "${exclude_libs[@]}" =~ "$sub_name" ]]; then
#             if [ -f "$Variable" ]; then
#                 echo "Add dependence: $Variable to $LibDir"
#                 cp "$Variable" "$LibDir"
#             else
#                 alternative_path="$EXTRA_LIB_PATH/$base_name"
#                 if [ -f "$alternative_path" ]; then
#                     echo "File not found at $Variable. Found at $alternative_path. Copying to $LibDir"
#                     cp "$alternative_path" "$LibDir"
#                 else
#                     echo "Warning: $Variable and $alternative_path are not regular files or do not exist."
#                 fi
#             fi
#         else
#             echo "+++++++++++++Ignore library: $Variable"
#         fi
#     done
# fi