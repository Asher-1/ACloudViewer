# Build system for header-only boost libraries.

include(ExternalProject)

if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for boost")
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for boost")
endif ()

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
        COMMAND $<IF:$<PLATFORM_ID:Windows>,bootstrap.bat,./bootstrap.sh> --without-libraries=python
        # recompiling with -fPIC for fixing linking error for "-fPIC" with add_executable()
        COMMAND $<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> -j6 -q -d+2 cxxflags=-fPIC cflags=-fPIC variant=release define=_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}
        # COMMAND $<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> -j6 -q -d+2 cxxflags=-fPIC cflags=-fPIC variant=release
        UPDATE_COMMAND ""
        INSTALL_COMMAND $<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> install --prefix=${CLOUDVIEWER_EXTERNAL_INSTALL_DIR} --without-python
)

ExternalProject_Get_Property(ext_boost INSTALL_DIR)

# By default, BOOST_INCLUDE_DIRS should not have trailing "/".
# The actual headers files are located in `${SOURCE_DIR}/boost`.
set(BOOST_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(INTERNAL_BOOST_ROOT ${INSTALL_DIR})