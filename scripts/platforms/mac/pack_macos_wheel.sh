#!/bin/bash

if [ -d $1 ]; then
    FrameworksDir=$2
    LIBS_NAME=`basename $FrameworksDir`
    for element in `ls $1`
    do
        exe=$1"/"$element

        # Find the rpath paths
        rpaths=$(otool -l "$exe" | grep "path " | awk '{print $2}')
        if [[ $rpaths != "" ]]; then
            echo "@rpath:"
            for rp in $rpaths; do
                echo "    $rp"
            done
        fi

        # Set IFS so that newlines don't become spaces; helps parsing the otool -L output
        IFS='
        '

        # Copy any external libraries and change the library paths to @loader_path
        lib_array=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`

        for lib in ${lib_array}
        do
            if [[ "$lib" == *"libtorch"*.dylib || "$lib" == *"libcudnn"*.dylib ]]; then
                echo "skip: $lib"
                continue
            fi

            if [[ ${lib:0:1} != "@" ]]; then    # external library with a regular path
                # copy the external library
                libname=`basename $lib`
                # copy the external library
                # Note: must judge if exists to some issues with dependencies
                if [[ ! -f "$FrameworksDir/$libname" ]]; then
                    cp "$lib" "$FrameworksDir"
                    chmod +w "$FrameworksDir/$libname"
                fi

                if [[ "$libname" != "$element" ]]; then
                    # change its path in the executable
                    newpath="@loader_path/../$LIBS_NAME/$libname"
                    echo "$lib -> $newpath"
                    install_name_tool -change "$lib" "$newpath" "$exe"
                fi

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

                if [[ "$libname" != "$element" ]]; then
                    # change its path in the executable
                    newpath="@loader_path/../$LIBS_NAME/$libname"
                    echo "$lib -> $newpath"
                    install_name_tool -change "$lib" "$newpath" $exe
                fi
            fi
        done

        # Remove rpaths
        for rp in $rpaths; do
            install_name_tool -delete_rpath "$rp" "$exe"
        done
    done
else
    FrameworksDir=$2
    LIBS_NAME=`basename $FrameworksDir`
    exe=$1

    # Find the rpath paths
    rpaths=$(otool -l "$exe" | grep "path " | awk '{print $2}')
    if [[ $rpaths != "" ]]; then
        echo "@rpath:"
        for rp in $rpaths; do
            echo "    $rp"
        done
    fi

    # Set IFS so that newlines don't become spaces; helps parsing the otool -L output
    IFS='
    '

    lib_array=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`

    for lib in ${lib_array}
    do
        if [[ "$lib" == *"libtorch"*.dylib || "$lib" == *"libcudnn"*.dylib ]]; then
                echo "skip: $lib"
                continue
        fi

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
            newpath="@loader_path/../$LIBS_NAME/$libname"
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
            newpath="@loader_path/../$LIBS_NAME/$libname"
            echo "$lib -> $newpath"
            install_name_tool -change "$lib" "$newpath" $exe
        fi
    done

    # Remove rpaths
    for rp in $rpaths; do
        install_name_tool -delete_rpath "$rp" "$exe"
    done
fi