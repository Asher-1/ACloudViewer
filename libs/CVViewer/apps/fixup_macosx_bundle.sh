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
    echo "Usage: $0 path/to/name.app"
    exit 1
fi

# Find the path to the actual executable in the app bundle
appBundle=$1
exeDir="$appBundle/Contents/MacOS"
shell_dir=`dirname $0`

exe=`find "$exeDir" -type f -perm +111 | grep -v dylib`
if [[ ! -f $exe ]]; then
    echo "No executable file in app bundle ($appBundle/Contents/MacOS)"
    exit 1
fi

echo "Found executable: $exe"

$shell_dir/fixup_macosx_libs.sh $exe