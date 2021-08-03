include(ExternalProject)

set(lib_name E57Format)

ExternalProject_Add(
    ext_libE57Format
    PREFIX libE57Format
    URL https://github.com/asmaloney/libE57Format/archive/refs/tags/v2.2.0.tar.gz
    URL_HASH SHA256=19df04af07925bf43e1793534b0c77cb1346a2bee7746859d2fe1714a24f1c7d
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/libE57Format"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_libE57Format INSTALL_DIR)
set(E57Format_INCLUDE_DIRS ${INSTALL_DIR}/include/E57Format/) # "/" is critical.
set(E57Format_LIB_DIR ${INSTALL_DIR}/lib)
set(E57Format_LIBRARIES ${lib_name}$<$<CONFIG:Debug>:d>)