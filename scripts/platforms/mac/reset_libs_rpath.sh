#!/bin/bash

if [ -d $1 ]; then
    for element in `ls $1`
    do
        exe=$1"/"$element
        # Copy any external libraries and change the library paths to @executable_path
        lib_array=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`

        for lib in ${lib_array}
        do
            if [[ ${lib:0:1} == "@" ]]; then    # external library with a regular path
                # copy the external library
                libname=`basename $lib`
                # change its path in the executable
                newpath="@rpath/$libname"
                echo "$lib -> $newpath"
                install_name_tool -change "$lib" "$newpath" "$exe"
            fi
        done
    done
else
    exe=$1
    lib_array=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`

    for lib in ${lib_array}
    do
        if [[ ${lib:0:1} == "@" ]]; then    # external library with a regular path
            # copy the external library
            libname=`basename $lib`
            # change its path in the executable
            newpath="@rpath/$libname"
            echo "$lib -> $newpath"
            install_name_tool -change "$lib" "$newpath" "$exe"
        fi
    done
fi