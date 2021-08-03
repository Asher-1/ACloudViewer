include(ExternalProject)

ExternalProject_Add(
    ext_lasLibzip
    PREFIX lasLibzip
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.8.0/LasZipLib.7z
    URL_HASH SHA256=361d69857201dc4b7bf237b95dc820a3e9814d11078875705c9fc31578cc3ecd
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/lasLibzip"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_lasLibzip SOURCE_DIR)
set(LASLIBZIP_DIR ${SOURCE_DIR}) # Not using "/" is critical.