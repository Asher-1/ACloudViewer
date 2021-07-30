include(ExternalProject)

set(lib_name dlib)

ExternalProject_Add(
    ext_dlib
    PREFIX dlib
    URL https://github.com/davisking/dlib/archive/refs/tags/v19.22.tar.gz
    URL_HASH SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/dlib"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        ${ExternalProject_CMAKE_ARGS}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_dlib SOURCE_DIR)
ExternalProject_Get_Property(ext_dlib INSTALL_DIR)
set(DLIB_ROOT_DIR ${SOURCE_DIR})
set(DLIB_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(DLIB_LIB_DIR ${INSTALL_DIR}/lib)
set(DLIB_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:d>)