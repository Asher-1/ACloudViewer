#!/usr/bin/env bash
# Install CloudViewer build dependencies from Ubuntu repositories
# CUDA (v10.1) and CUDNN (v7.6.5) are optional dependencies and are not
# installed here
# Use: install_deps_ubuntu.sh [ assume-yes ]

set -ev

SUDO=${SUDO:=sudo}      # SUDO=command in docker (running as root, sudo not available)
if [ "$1" == "assume-yes" ] ; then
    APT_CONFIRM="--assume-yes"
else
    APT_CONFIRM=""
fi

dependencies=(
	# CloudViewer deps
	xorg-dev
	libglu1-mesa-dev
	python3-dev
	# Filament build-from-source deps
	libsdl2-dev
	libc++-7-dev
	libc++abi-7-dev
	ninja-build
	libxi-dev
	# ML deps
	libtbb-dev
	# Headless rendering deps
	libosmesa6-dev
	# other deps
	build-essential 
	libxmu-dev 
	libxi-dev    
	g++
	libflann1.9 
	libflann-dev
	doxygen
	mpi-default-dev 
	openmpi-bin 
	openmpi-common
	libeigen3-dev
	libboost-all-dev
	libglew-dev
	'libqhull*'
	libgtest-dev
	libusb-1.0-0-dev 
	libusb-dev 
	libudev-dev
	git-core 
	freeglut3-dev 
	pkg-config
	libpcap-dev
	clang-format 
	libqhull-dev
	graphviz
	mono-complete
	phonon-backend-gstreamer
	phonon-backend-vlc
	libopenni-dev 
	libopenni2-dev
	libgl1-mesa-dev 
	libglu1-mesa-dev
	libx11-dev 
	libxext-dev 
	libxtst-dev 
	libxrender-dev 
	libxmu-dev 
	libxmuu-dev
)

$SUDO apt-get update
for package in "${dependencies[@]}" ; do
    $SUDO apt-get install "$APT_CONFIRM" "$package"
done
