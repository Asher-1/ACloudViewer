#!/bin/bash

if [ -d $1 ]; then
    LibDir=$2
    for element in `ls $1`
    do
        Target=$1"/"$element
        lib_array=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))

        for Variable in ${lib_array[@]}
        do
            echo "Add dependence: $Variable to $LibDir"
            cp "$Variable" $LibDir
        done
    done
else
    LibDir=$2
    Target=$1
    lib_array=($(ldd $Target | grep -o "/.*" | grep -o "/.*/[^[:space:]]*"))

    for Variable in ${lib_array[@]}
    do
        echo "Add dependence: $Variable to $LibDir"
        cp "$Variable" $LibDir
    done
fi