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
exeDir="$appBundle/Contents/Frameworks"
shell_dir=`dirname $0`

libraries=`ls "$exeDir" | grep dylib | awk '{ print $1; }'`

# shellcheck disable=SC1068
for lib in $libraries; do
  lib="$exeDir/$lib"
  echo "libraries: $lib"
  $shell_dir/fixup_macosx_libs.sh $lib
done