# Build system for header-only boost libraries.
#
# In general, we prefer avoiding boost or use header-only boost libraries.
# Compiling boost libraries can addup to the build time.
#
# Current boost libraries:
# - predef (header-only)

include(ExternalProject)

ExternalProject_Add(
        ext_boost
        PREFIX boost
        URL https://github.com/alicevision/AliceVisionDependencies/releases/download/boost-src-1.73.0/boost_1_73_0.tar.bz2
        URL_HASH SHA256=4eb3b8d442b426dc35346235c8733b5ae35ba431690e38c6a8263dce9fcbb402
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/boost"
        BUILD_IN_SOURCE ON
        CONFIGURE_COMMAND ""
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        BUILD_COMMAND echo "Running Boost build..."
#        COMMAND ${Python3_EXECUTABLE} tools/boostdep/depinst/depinst.py predef
        COMMAND $<IF:$<PLATFORM_ID:Windows>,bootstrap.bat,./bootstrap.sh>
#        COMMAND $<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> headers
        COMMAND $<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> -j6 -q -d+2 cxxflags=-fPIC cflags=-fPIC variant=release
        UPDATE_COMMAND ""
        INSTALL_COMMAND "$<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> install --prefix=${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}"
)

ExternalProject_Get_Property(ext_boost SOURCE_DIR)
ExternalProject_Get_Property(ext_boost INSTALL_DIR)
message(STATUS "Boost source dir: ${SOURCE_DIR}")
message(STATUS "Boost install dir: ${INSTALL_DIR}")

# By default, BOOST_INCLUDE_DIRS should not have trailing "/".
# The actual headers files are located in `${SOURCE_DIR}/boost`.
set(BOOST_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(BOOST_ROOT ${INSTALL_DIR}/lib/cmake)