include(ExternalProject)

ExternalProject_Add(ext_draco
        PREFIX draco
        URL https://github.com/google/draco/archive/refs/tags/1.5.7.tar.gz
        URL_HASH SHA256=bf6b105b79223eab2b86795363dfe5e5356050006a96521477973aba8f036fe1
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/draco"
        UPDATE_COMMAND ""
        CMAKE_ARGS
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5
                ${ExternalProject_CMAKE_ARGS_hidden}
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DBUILD_SHARED_LIBS=OFF
                -DCMAKE_BUILD_TYPE=$<$<PLATFORM_ID:Windows,Darwin>:${CMAKE_BUILD_TYPE}:Release>
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}draco${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Get_Property(ext_draco INSTALL_DIR)
set(DRACO_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(DRACO_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(DRACO_LIBRARIES draco)