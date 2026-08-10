#!/usr/bin/env bash
#
# build_vtk_pcl_deps.sh — build or consume the prebuilt VTK+PCL dependency
# layer for Ubuntu CI.
#
# These six dependencies (eigen, metslib, laszip, xerces-c, VTK, PCL) are the
# heaviest part of the Docker CI build (source-compiling them with `make -j2`
# is slow, RAM/disk hungry and is the main driver of the runner OOM /
# "hosted runner lost communication with the server" failures). To fix that we
# prebuild them once per Ubuntu release on a GitHub runner and publish the
# installed tree as a versioned tarball; every normal CI build then just
# downloads + extracts it instead of recompiling.
#
# Two modes (identical dependency build logic, so there is zero drift between
# the generated artifact and what a fallback source build would produce):
#
#   prebuild <BUILD_JOBS> <QT_DIR> <VTK_VERSION> <PCL_VERSION>
#     Build the six deps from source and package /usr/local (+ xerces bits that
#     land in /usr) into /out/vtk-pcl-deps-<codename>-vtk<VTK>-pcl<PCL>.tar.gz.
#     Used by the prebuild-deps workflow to generate the artifact.
#
#   consume  <BUILD_JOBS> <QT_DIR> <VTK_VERSION> <PCL_VERSION>
#     Download the prebuilt tarball (published on the ACloudViewer
#     `ubuntu-prebuild-deps` release) and extract it into /. If the tarball is
#     not published yet, falls back to building from source (current behaviour)
#     so the transition is non-breaking.
#
# ABI note: VTK/PCL are pure-CPU libraries; their ABI depends only on the
# Ubuntu release (glibc / libstdc++ / gcc from the same release), not on CUDA.
# The prebuilt tarball is therefore produced once per Ubuntu version on the
# plain `ubuntu:<ver>` image and is consumed by BOTH the CPU and the CUDA
# Docker builds of that Ubuntu version (they install the same apt dependency
# set in Dockerfile.ci).
set -euo pipefail

MODE="${1:?usage: build_vtk_pcl_deps.sh <prebuild|consume> <BUILD_JOBS> <QT_DIR> <VTK_VERSION> <PCL_VERSION>}"
BUILD_JOBS="${2:?BUILD_JOBS required}"
QT_DIR="${3:?QT_DIR required}"
VTK_VERSION="${4:?VTK_VERSION required}"
PCL_VERSION="${5:?PCL_VERSION required}"

CODENAME="$(lsb_release -c --short)"
TARBALL="vtk-pcl-deps-${CODENAME}-vtk${VTK_VERSION}-pcl${PCL_VERSION}.tar.gz"
# Published on ACloudViewer (works with the CI GITHUB_TOKEN). Mirror to
# Asher-1/cloudViewer_downloads manually if you want it hosted there.
URL="https://github.com/Asher-1/ACloudViewer/releases/download/ubuntu-prebuild-deps/${TARBALL}"

# ---------------------------------------------------------------------------
# The six dependency builds, byte-for-byte the steps that used to live inline
# in Dockerfile.ci (kept identical so the prebuilt artifact == the source
# build, guaranteeing ABI/environment consistency).
# ---------------------------------------------------------------------------
build_all_deps() {
    echo "[vtk-pcl-deps] building all six deps from source (BUILD_JOBS=${BUILD_JOBS})"

    # eigen 3.4+
    cd /opt
    mv -f /usr/include/eigen3 /usr/include/eigen337.bak
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -O "/opt/eigen-3.4.0.tar.gz"
    tar -xvf eigen-3.4.0.tar.gz
    cd eigen-3.4.0
    mkdir -p build && cd build
    cmake ..
    make install "-j${BUILD_JOBS}"
    ldconfig
    rm -rf /opt/eigen-3.4.0 /opt/eigen-3.4.0.tar.gz

    # metslib (PCL dependency)
    cd /opt
    wget -q https://github.com/Asher-1/cloudViewer_downloads/releases/download/docker_files/metslib-0.5.3.tgz -O "/opt/metslib-0.5.3.tgz"
    tar -xvf metslib-0.5.3.tgz
    cd metslib-0.5.3
    sh ./configure
    make
    make install
    ldconfig
    rm -rf /opt/metslib-0.5.3 /opt/metslib-0.5.3.tgz

    # laszip
    cd /opt
    wget -q https://github.com/Asher-1/cloudViewer_downloads/releases/download/docker_files/laszip-src-3.4.3.tar.gz -O "/opt/laszip-src-3.4.3.tar.gz"
    tar -xvf laszip-src-3.4.3.tar.gz
    cd laszip-src-3.4.3
    mkdir -p build && cd build
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    make "-j${BUILD_JOBS}"
    make install "-j${BUILD_JOBS}"
    ldconfig
    rm -rf /opt/laszip-src-3.4.3 /opt/laszip-src-3.4.3.tar.gz

    # xerces-c (installs to /usr via --prefix=/usr)
    cd /opt
    wget -q https://github.com/Asher-1/cloudViewer_downloads/releases/download/docker_files/xerces-c-3.2.3.zip -O "/opt/xerces-c-3.2.3.zip"
    unzip xerces-c-3.2.3.zip
    cd ./xerces-c-3.2.3
    chmod +x configure
    ./configure --prefix=/usr
    make
    make install
    make clean
    rm -rf /opt/xerces-c-3.2.3 /opt/xerces-c-3.2.3.zip

    # VTK (Qt5, shared)
    cd /opt
    wget -q https://vtk.org/files/release/9.3/VTK-${VTK_VERSION}.tar.gz -O "/opt/VTK-${VTK_VERSION}.tar.gz"
    tar -zxvf VTK-${VTK_VERSION}.tar.gz
    cd VTK-${VTK_VERSION}
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
        -DVTK_GROUP_ENABLE_Qt=YES \
        -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
        -DVTK_MODULE_ENABLE_VTK_GUISupportQtQuick=YES \
        -DVTK_MODULE_ENABLE_VTK_GUISupportQtSQL=YES \
        -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES \
        -DVTK_MODULE_ENABLE_VTK_ViewsQt=YES \
        -DVTK_QT_VERSION:STRING=5 \
        -DBUILD_SHARED_LIBS:BOOL=ON \
        -DQT_QMAKE_EXECUTABLE:PATH=${QT_DIR}/bin/qmake \
        -DCMAKE_PREFIX_PATH:PATH=${QT_DIR}/lib/cmake ..
    make "-j${BUILD_JOBS}"
    make install "-j${BUILD_JOBS}"
    ldconfig
    rm -rf /opt/VTK-${VTK_VERSION} /opt/VTK-${VTK_VERSION}.tar.gz

    # PCL
    cd /opt
    wget -q https://github.com/PointCloudLibrary/pcl/releases/download/pcl-${PCL_VERSION}/source.zip -O "/opt/pcl-${PCL_VERSION}.zip"
    unzip pcl-${PCL_VERSION}.zip
    cd pcl
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
        -DBUILD_GPU=OFF \
        -DBUILD_apps=OFF \
        -DBUILD_examples=OFF \
        -DBUILD_surface_on_nurbs=ON \
        -DPCL_ENABLE_MARCHNATIVE=OFF \
        -DQT_QMAKE_EXECUTABLE:PATH=${QT_DIR}/bin/qmake \
        -DCMAKE_PREFIX_PATH:PATH=${QT_DIR}/lib/cmake ..
    make "-j${BUILD_JOBS}"
    make install "-j${BUILD_JOBS}"
    ldconfig
    rm -rf /opt/pcl /opt/pcl-${PCL_VERSION}.zip

    echo "[vtk-pcl-deps] all six deps built from source"
}

case "${MODE}" in
    prebuild)
        build_all_deps
        # Package /usr/local (eigen, metslib, laszip, VTK, PCL) plus every
        # xerces-c artifact (installed with --prefix=/usr): headers, libs and
        # its CMake package config (needed for find_package(XercesC)).
        # `cd /` stores root-relative paths so `tar -xzf ... -C /` is a drop-in install.
        mkdir -p /out
        ( cd / && tar -czf /out/"${TARBALL}" \
            usr/local \
            usr/include/xercesc \
            $(ls -d usr/lib/libxerces* usr/lib/x86_64-linux-gnu/libxerces* 2>/dev/null || true) \
            $(ls -d usr/lib/cmake/xerces* usr/lib/x86_64-linux-gnu/cmake/xerces* 2>/dev/null || true) \
            $(ls -d usr/lib/pkgconfig/xerces* usr/lib/x86_64-linux-gnu/pkgconfig/xerces* 2>/dev/null || true) )
        echo "[vtk-pcl-deps] packaged /out/${TARBALL}"
        ;;
    consume)
        if wget -q --tries=3 "${URL}" -O /tmp/vtk-pcl-deps.tar.gz; then
            tar -xzf /tmp/vtk-pcl-deps.tar.gz -C /
            ldconfig || true
            rm -f /tmp/vtk-pcl-deps.tar.gz
            echo "[vtk-pcl-deps] used prebuilt deps: ${URL}"
        else
            echo "[vtk-pcl-deps] prebuilt deps not published yet (${URL}); building from source"
            build_all_deps
        fi
        ;;
    *)
        echo "unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
