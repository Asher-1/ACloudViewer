#!/usr/bin/env bash
set -euo pipefail

# From build/bin directory:
# ./scripts/platforms/mac/sign_macos_app.sh ~/cloudViewer_install/ACloudViewer/ACloudViewer.app ./eCV/Mac/ACloudViewer.entitlements <apple-id> <cert-name> <team-id> <app-password>

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

MACOS_APP_BUILD_SHELL=${CLOUDVIEWER_SOURCE_ROOT}/scripts/build_macos_app.sh
if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "ACloudViewer*.dmg" | grep -q .; then
    echo "Start building ACloudViewer app..."
    rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_APP_BUILD_SHELL}
else
    echo "Ignore ACloudViewer GUI app building due to have builded before..."
fi
echo

echo "Start to build wheel for python3.8-3.11 On MacOS..."
echo
MACOS_WHL_BUILD_SHELL=${CLOUDVIEWER_SOURCE_ROOT}/scripts/build_macos_whl.sh
if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "cloudViewer*cp38*.whl" | grep -q .; then
    echo "Start building cloudViewer wheel for python3.8..."
    rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.8
else
    echo "Ignore cloudViewer wheel for python3.8..."
fi

if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "cloudViewer*cp39*.whl" | grep -q .; then
    echo "Start building cloudViewer wheel for python3.9..."
    rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.9
else
    echo "Ignore cloudViewer wheel for python3.9..."
fi

if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "cloudViewer*cp310*.whl" | grep -q .; then
    echo "Start building cloudViewer wheel for python3.10..."
    rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.10
else
    echo "Ignore cloudViewer wheel for python3.10..."
fi

if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "cloudViewer*cp311*.whl" | grep -q .; then
    echo "Start building cloudViewer wheel for python3.11..."
    rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.11
else
    echo "Ignore cloudViewer wheel for python3.11..."
fi

echo "All install to ${ACloudViewer_INSTALL}"
echo