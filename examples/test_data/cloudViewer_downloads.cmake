include(ExternalProject)

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/cloudViewer_downloads")

ExternalProject_Add(
    ext_cloudViewer_downloads
    PREFIX cloudViewer_downloads
    URL https://github.com/Asher-1/cloudViewer_downloads/archive/refs/tags/1.0.0.tar.gz
    URL_HASH SHA256=de306a156dbb1ba33493c5f0394fec088d668934c6815a317d1ddfe4f93dbf89
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/cloudViewer_downloads"
    SOURCE_DIR "${SOURCE_DIR}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cloudViewer_downloads SOURCE_DIR)
set(TEST_DATA_DIR ${SOURCE_DIR})
