include(ExternalProject)

set(lib_name QRANSAC_SD_PRIM_SHAPES_LIB)
ExternalProject_Add(
    ext_ransacSD
    PREFIX ransacSD
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.6.0/RANSAC_SD_orig_fixed.zip
    URL_HASH MD5=673B9CAFC0404684BFF1A0CE060FAD1D
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ransacSD"
    UPDATE_COMMAND ""
	CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
		${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_ransacSD SOURCE_DIR)
ExternalProject_Get_Property(ext_ransacSD INSTALL_DIR)
set(RANSACSD_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
set(RANSACSD_LIB_DIR ${INSTALL_DIR}/lib)
set(RANSACSD_LIBRARIES ${lib_name})
