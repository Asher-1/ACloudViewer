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
    echo "Usage: $0 path/to/lib_dir"
    exit 1
fi

shell_dir=`dirname $0`

# Find the path to the actual executable in the app bundle
LIB_DIR=$1
if [[ ! -d $LIB_DIR ]]; then
    echo "Expected path given and Usage: $0 path/to/lib_dir"
    exit 1
fi

BASE_NAME=$(basename $LIB_DIR)
if [ "$BASE_NAME" == "MacOS" ]; then
    exe=`find "$LIB_DIR" -type f -perm +111 | grep -v dylib`
    if [[ ! -f $exe ]]; then
        echo "No executable file in app bundle ($LIB_DIR)"
        exit 1
    fi

    echo "Found executable: $exe"
    echo "execute cmd: $shell_dir/fixup_macosx_libs.sh $exe"
    $shell_dir/fixup_macosx_libs.sh $exe
else
    libraries=`ls "$LIB_DIR" | grep dylib | awk '{ print $1; }'`

    # shellcheck disable=SC1068
    for lib in $libraries; do
        lib="$LIB_DIR/$lib"
        echo "libraries: $lib"
        $shell_dir/fixup_macosx_libs.sh $lib
    done
fi