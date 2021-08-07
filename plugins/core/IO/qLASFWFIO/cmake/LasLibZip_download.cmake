include(ExternalProject)

ExternalProject_Add(
    ext_lasLibzip
    PREFIX lasLibzip
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.8.0/LasZipLib.7z
    URL_HASH MD5=38B48A3C413BBF34DF6F3BFF1E869CCB
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/lasLibzip"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_lasLibzip SOURCE_DIR)
set(LASLIBZIP_DIR ${SOURCE_DIR}) # Not using "/" is critical.