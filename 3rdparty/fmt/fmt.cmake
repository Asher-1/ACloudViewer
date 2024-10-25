include(ExternalProject)

set(FMT_VER "6.0.0")
set(FMT_SHA256
    "f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78")

# set(FMT_VER "10.2.1")
# set(FMT_SHA256
#     "1250e4cc58bf06ee631567523f48848dc4596133e163f02615c97f78bab6c811")

ExternalProject_Add(
    ext_fmt
    PREFIX fmt
    URL https://github.com/fmtlib/fmt/archive/refs/tags/${FMT_VER}.tar.gz
    URL_HASH SHA256=${FMT_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/fmt"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_fmt SOURCE_DIR)
set(FMT_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
