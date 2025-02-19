#!/bin/bash

# A macOS executable linked to dynamic libraries will either link to a full path
# or an rpath (a series of which will be specified in the binary). In order to
# have a self-contained app bundle, we need to copy the external libraries into
# the bundle, and then update the executable to use @executable_path to point
# within the bundle.

if [[ `uname` != "Darwin" ]]; then
    echo "This script is only useful for macOS"
    exit 1
fi

if [[ $# != 1 ]]; then
    echo "Usage: $0 path/libname"
    exit 1
fi

# Find the path to the actual executable in the app bundle
exe=$1
# shellcheck disable=SC1068
SUB_DIR=`dirname $exe`
ContentsDir=`dirname $SUB_DIR`
FrameworksDir="$ContentsDir/Frameworks"
echo "FrameworksDir: $FrameworksDir"

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

# Copy any external libraries and change the library paths to @executable_path
libs=`otool -L "$exe" | grep -v "$exe" | grep -v /usr | grep -v /System | awk '{ print $1; }'`
for lib in $libs; do
    if [[ ${lib:0:1} != "@" ]]; then    # external library with a regular path
        # copy the external library
        libname=`basename $lib`
        # copy the external library
        # Note: must judge if exists to some issues with dependencies
        if [[ ! -f "$FrameworksDir/$libname" ]]; then
          echo "cp $lib $FrameworksDir"
          cp "$lib" "$FrameworksDir"
          chmod +w "$FrameworksDir/$libname"
        fi

        # change its path in the executable
        newpath="@executable_path/../Frameworks/$libname"
        echo "$lib -> $newpath"
        install_name_tool -change "$lib" "$newpath" "$exe"

    elif [[ $lib == @rpath/* ]]; then   # external library with @rpath
        libname=${lib:7}
        # copy the external library. Since it uses an rpath, we need to
        # prepend each rpath to see which one gives a valid path
        for rp in $rpaths; do
            if [[ -f "$rp/$libname" ]]; then
                echo "cp $rp/$libname $FrameworksDir"
                cp "$rp/$libname" "$FrameworksDir"
                chmod +w "$FrameworksDir/$libname"
                break
            fi
        done
        # change its path in the executable
        newpath="@executable_path/../Frameworks/$libname"
        echo "$lib -> $newpath"
        install_name_tool -change "$lib" "$newpath" $exe
    fi
done

# Remove rpaths
for rp in $rpaths; do
    echo "delete rpath $rp for $exe"
    install_name_tool -delete_rpath "$rp" "$exe"
done