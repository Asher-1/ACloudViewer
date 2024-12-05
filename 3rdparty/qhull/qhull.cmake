# include(ExternalProject)

# ExternalProject_Add(
#     ext_qhull
#     PREFIX qhull
#     URL https://github.com/qhull/qhull/archive/refs/tags/v7.3.0.tar.gz
#     URL_HASH SHA256=05a2311d8e6397c96802ee5a9d8db32b83dac7e42e2eb2cd81c5547c18e87de6
#     DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/qhull"
#     UPDATE_COMMAND ""
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND ""
#     INSTALL_COMMAND ""
# )

# ExternalProject_Get_Property(ext_qhull SOURCE_DIR)
# set(QHULL_SOURCE_DIR ${SOURCE_DIR})

include(FetchContent)

FetchContent_Declare(
    ext_qhull
    PREFIX qhull
    # v8.0.0+ causes seg fault
    URL https://github.com/qhull/qhull/archive/refs/tags/v8.0.2.tar.gz
    URL_HASH
    SHA256=8774e9a12c70b0180b95d6b0b563c5aa4bea8d5960c15e18ae3b6d2521d64f8b
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/qhull"
)

FetchContent_Populate(ext_qhull)
FetchContent_GetProperties(ext_qhull SOURCE_DIR QHULL_SOURCE_DIR)