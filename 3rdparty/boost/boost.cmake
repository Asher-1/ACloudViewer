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
        URL https://github.com/boostorg/boost/archive/refs/tags/boost-1.73.0.tar.gz
        URL_HASH SHA256=2f5f2b789edb00260aa71f03189da5f21cf4b5617c4fbba709e9fbcfc76a2f1e
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/boost"
        BUILD_IN_SOURCE ON
        CONFIGURE_COMMAND ""
        BUILD_COMMAND echo "Running Boost build..."
        COMMAND python tools/boostdep/depinst/depinst.py predef
        COMMAND $<IF:$<PLATFORM_ID:Windows>,bootstrap.bat,./bootstrap.sh>
        COMMAND $<IF:$<PLATFORM_ID:Windows>,b2.exe,./b2> headers
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_boost SOURCE_DIR)
message(STATUS "Boost source dir: ${SOURCE_DIR}")

# By default, BOOST_INCLUDE_DIRS should not have trailing "/".
# The actual headers files are located in `${SOURCE_DIR}/boost`.
set(BOOST_INCLUDE_DIRS ${SOURCE_DIR}/ext_boost)