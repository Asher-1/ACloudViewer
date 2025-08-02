include(ExternalProject)

ExternalProject_Add(ext_cork
    PREFIX cork
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.9.0/cork.7z
    URL_HASH MD5=8014B0317BE35DE273A78780A688F49A
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/cork"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cork SOURCE_DIR)
set(CORK_DIR ${SOURCE_DIR}) # Not using "/" is critical.