#!/bin/bash

if [ -d $1 ]; then
    FrameworksDir=$2
    for element in `ls $1`
    do
        exe=$1"/"$element
        # Copy any external libraries and change the library paths to @executable_path
        lib_array=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`

        for lib in ${lib_array}
        do
            if [[ ${lib:0:1} != "@" ]]; then    # external library with a regular path
                # copy the external library
                libname=`basename $lib`
                # copy the external library
                # Note: must judge if exists to some issues with dependencies
                if [[ ! -f "$FrameworksDir/$libname" ]]; then
                    cp "$lib" "$FrameworksDir"
                    chmod +w "$FrameworksDir/$libname"
                fi

                # change its path in the executable
                newpath="@executable_path/../$FrameworksDir/$libname"
                echo "$lib -> $newpath"
                install_name_tool -change "$lib" "$newpath" "$exe"

            elif [[ $lib == @rpath/* ]]; then   # external library with @rpath
                libname=${lib:7}
                # copy the external library. Since it uses an rpath, we need to
                # prepend each rpath to see which one gives a valid path
                for rp in $rpaths; do
                    if [[ -f "$rp/$libname" ]]; then
                        cp "$rp/$libname" "$FrameworksDir"
                        chmod +w "$FrameworksDir/$libname"
                        break
                    fi
                done
                # change its path in the executable
                newpath="@executable_path/../$FrameworksDir/$libname"
                echo "$lib -> $newpath"
                install_name_tool -change "$lib" "$newpath" $exe
            fi
        done
    done
else
    FrameworksDir=$2
    exe=$1
    lib_array=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`

    for lib in ${lib_array}
    do
        if [[ ${lib:0:1} != "@" ]]; then    # external library with a regular path
            # copy the external library
            libname=`basename $lib`
            # copy the external library
            # Note: must judge if exists to some issues with dependencies
            if [[ ! -f "$FrameworksDir/$libname" ]]; then
                cp "$lib" "$FrameworksDir"
                chmod +w "$FrameworksDir/$libname"
            fi

            # change its path in the executable
            newpath="@executable_path/../$FrameworksDir/$libname"
            echo "$lib -> $newpath"
            install_name_tool -change "$lib" "$newpath" "$exe"

        elif [[ $lib == @rpath/* ]]; then   # external library with @rpath
            libname=${lib:7}
            # copy the external library. Since it uses an rpath, we need to
            # prepend each rpath to see which one gives a valid path
            for rp in $rpaths; do
                if [[ -f "$rp/$libname" ]]; then
                    cp "$rp/$libname" "$FrameworksDir"
                    chmod +w "$FrameworksDir/$libname"
                    break
                fi
            done
            # change its path in the executable
            newpath="@executable_path/../$FrameworksDir/$libname"
            echo "$lib -> $newpath"
            install_name_tool -change "$lib" "$newpath" $exe
        fi
    done
fi