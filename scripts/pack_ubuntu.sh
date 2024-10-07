#!/bin/bash

# options="$(echo "$@" | tr ' ' '|')"
# if [[ "skip_libs" =~ ^($options)$ ]]; then
#     exclude_libs="libc.so.* libdl.so.* libselinux.so.* libpthread.so.* librt.so.* ld-linux-x86-64.so.* libdrm.so.* libm.so.* libdrm_intel.so.* libstdc++.so.*"
# else
#     exclude_libs=""
# fi
exclude_libs="libc.so.* libdl.so.* libselinux.so.* libpthread.so.* librt.so.* ld-linux-x86-64.so.* libdrm.so.* libm.so.* libdrm_intel.so.* libstdc++.so.*"

if [ -d $1 ]; then
    LibDir=$2
    for element in `ls $1`
    do
        Target=$1"/"$element
        if [ -f $Target ]; then
            lib_array=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))

            for Variable in ${lib_array[@]}
            do
                base_name=$(basename $Variable)
                sub_name=${base_name:0:7}
                if [[ ! "${exclude_libs[@]}" =~ "$sub_name" ]]; then
                    echo "Add dependence: $Variable to $LibDir"
                    cp "$Variable" $LibDir
                else
                    echo "+++++++++++++Ignore library: $Variable"
                fi
            done
        fi
    done
else
    LibDir=$2
    Target=$1
    lib_array=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))

    for Variable in ${lib_array[@]}
    do
        base_name=$(basename $Variable)
        sub_name=${base_name:0:7}
        if [[ ! "${exclude_libs[@]}" =~ "$sub_name" ]]; then
            echo "Add dependence: $Variable to $LibDir"
            cp "$Variable" $LibDir
        else
            echo "+++++++++++++Ignore library: $Variable"
        fi
    done
fi