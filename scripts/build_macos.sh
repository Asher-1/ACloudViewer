#!/usr/bin/env bash
set -euo pipefail

# From build/bin directory:
# ./scripts/platforms/mac/sign_macos_app.sh ~/cloudViewer_install/ACloudViewer/ACloudViewer.app ./app/Mac/ACloudViewer.entitlements <apple-id> <cert-name> <team-id> <app-password>

# bundle app libs
# python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/lib_bundle_app.py ACloudViewer ~/cloudViewer_install/ACloudViewer

# sign apps mannually
# python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/signature_app.py ACloudViewer /Users/asher/cloudViewer_install/ACloudViewer

# test wheel
# python -W default -c "import cloudViewer; print('Installed:', cloudViewer); print('BUILD_CUDA_MODULE: ', cloudViewer._build_config['BUILD_CUDA_MODULE'])"
# python -W default -c "import cloudViewer; print('CUDA available: ', cloudViewer.core.cuda.is_available())"
# python -W default -c "import cloudViewer.ml.torch; print('PyTorch Ops library loaded:', cloudViewer.ml.torch._loaded)"

# test reconstruction
# python -W default -c "import cloudViewer; print('Installed:', cloudViewer); print('BUILD_RECONSTRUCTION: ', cloudViewer._build_config['BUILD_RECONSTRUCTION'])"
# python -W default -c "import cloudViewer as cv3d; cv3d.reconstruction.gui.run_graphical_gui()"

ACloudViewer_INSTALL=~/cloudViewer_install
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
CLOUDVIEWER_BUILD_DIR=${CLOUDVIEWER_SOURCE_ROOT}/build
export CLOUDVIEWER_ML_ROOT=/Users/asher/develop/code/github/CloudViewer-ML

MACOS_APP_BUILD_SHELL=${CLOUDVIEWER_SOURCE_ROOT}/scripts/build_macos_app.sh
if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "ACloudViewer*.dmg" | grep -q .; then
    echo "Start building ACloudViewer app with python3.12"
    rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_APP_BUILD_SHELL} 3.12
else
    echo "Ignore ACloudViewer GUI app building due to have builded before..."
fi
echo

echo "Start to build wheel with python3.10-3.12 On MacOS..."
echo

PYTHON_VERSIONS=("3.10" "3.11" "3.12")
MACOS_WHL_BUILD_SHELL=${CLOUDVIEWER_SOURCE_ROOT}/scripts/build_macos_whl.sh

for version in "${PYTHON_VERSIONS[@]}"; do
    if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "cloudviewer*cp${version//./}*.whl" | grep -q .; then
        echo "Start building cloudviewer wheel with python${version}..."
        rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} ${version}
    else
        echo "Ignore cloudviewer wheel with python${version}..."
    fi
done

echo "All install to ${ACloudViewer_INSTALL}"
