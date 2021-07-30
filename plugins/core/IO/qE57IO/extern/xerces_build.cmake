include(ExternalProject)

set(lib_name xerces)

ExternalProject_Add(
    ext_xerces
    PREFIX xerces
    URL https://ftp.tsukuba.wide.ad.jp/software/apache//xerces/c/3/sources/xerces-c-3.2.3.tar.gz
    URL_HASH SHA256=fb96fc49b1fb892d1e64e53a6ada8accf6f0e6d30ce0937956ec68d39bd72c7e
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/xerces"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        ${ExternalProject_CMAKE_ARGS}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_xerces INSTALL_DIR)
set(XERCES_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(XERCES_LIB_DIR ${INSTALL_DIR}/lib)
set(XERCES_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:d>)