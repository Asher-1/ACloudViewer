include(ExternalProject)

set(lib_name QRANSAC_SD_PRIM_SHAPES_LIB)
ExternalProject_Add(
    ext_ransacSD
    PREFIX ransacSD
    URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.6.0/RANSAC_SD_orig_fixed.zip
    URL_HASH MD5=673B9CAFC0404684BFF1A0CE060FAD1D
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ransacSD"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E echo "cmake_minimum_required(VERSION 3.10)" > <SOURCE_DIR>/temp_header.txt
    COMMAND ${CMAKE_COMMAND} -E cat <SOURCE_DIR>/CMakeLists.txt >> <SOURCE_DIR>/temp_header.txt
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/temp_header.txt <SOURCE_DIR>/CMakeLists.txt
    COMMAND ${CMAKE_COMMAND} -E remove <SOURCE_DIR>/temp_header.txt
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POLICY_CMP0000=NEW
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
