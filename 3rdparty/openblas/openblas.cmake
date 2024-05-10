include(ExternalProject)

if(LINUX_AARCH64 OR APPLE_AARCH64)
    set(OPENBLAS_TARGET "ARMV8")
else()
    set(OPENBLAS_TARGET "NEHALEM")
endif()

ExternalProject_Add(
        ext_openblas
        PREFIX openblas
        URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20.tar.gz
        URL_HASH SHA256=8495c9affc536253648e942908e88e097f2ec7753ede55aca52e5dead3029e3c
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/openblas"
        CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS}
        -DTARGET=${OPENBLAS_TARGET}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${lib_suffix}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_openblas INSTALL_DIR)
set(OPENBLAS_INCLUDE_DIR ${INSTALL_DIR}/include/openblas/) # "/" is critical.
set(OPENBLAS_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(OPENBLAS_LIBRARIES openblas)

message(STATUS "OPENBLAS_INCLUDE_DIR: ${OPENBLAS_INCLUDE_DIR}")
message(STATUS "OPENBLAS_LIB_DIR ${OPENBLAS_LIB_DIR}")
message(STATUS "OPENBLAS_LIBRARIES: ${OPENBLAS_LIBRARIES}")