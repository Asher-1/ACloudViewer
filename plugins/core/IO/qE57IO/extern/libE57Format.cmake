include(ExternalProject)

ExternalProject_Add(
    ext_libE57Format
    PREFIX libE57Format
    URL https://github.com/asmaloney/libE57Format/archive/refs/tags/v2.2.0.tar.gz
    URL_HASH SHA256=2f5f2b789edb00260aa71f03189da5f21cf4b5617c4fbba709e9fbcfc76a2f1e
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/libE57Format"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_libE57Format SOURCE_DIR)
set(LIBE57FORMAT_SOURCE_DIR ${SOURCE_DIR})
