include(ExternalProject)

if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for pcl")
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for pcl")
endif ()

if(WIN32)
    set(PCL_LIB_SUFFIX $<$<CONFIG:Debug>:d>)
else()
    set(PCL_LIB_SUFFIX "")
endif()

set(PCL_VERSION 1.14)

set(PCL_LIBRARIES   pcl_io${PCL_LIB_SUFFIX}
                    pcl_ml${PCL_LIB_SUFFIX}
                    pcl_common${PCL_LIB_SUFFIX}
                    pcl_io_ply${PCL_LIB_SUFFIX}
                    pcl_kdtree${PCL_LIB_SUFFIX}
                    pcl_octree${PCL_LIB_SUFFIX}
                    pcl_search${PCL_LIB_SUFFIX}
                    pcl_filters${PCL_LIB_SUFFIX}
                    pcl_surface${PCL_LIB_SUFFIX}
                    pcl_features${PCL_LIB_SUFFIX}
                    pcl_recognition${PCL_LIB_SUFFIX}
                    pcl_registration${PCL_LIB_SUFFIX}
                    pcl_segmentation${PCL_LIB_SUFFIX}
                    pcl_visualization${PCL_LIB_SUFFIX}
                    pcl_sample_consensus${PCL_LIB_SUFFIX})

foreach(item IN LISTS PCL_LIBRARIES)
    list(APPEND PCL_BUILD_BYPRODUCTS <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${item}${CMAKE_STATIC_LIBRARY_SUFFIX}.${PCL_VERSION})
endforeach()

ExternalProject_Add(
        ext_pcl
        PREFIX pcl
        URL https://github.com/PointCloudLibrary/pcl/releases/download/pcl-${PCL_VERSION}.1/source.zip
        URL_HASH MD5=91cb583c1d5ecf221ee16a736c0ae39d
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/pcl"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        CMAKE_ARGS
        ${VTK_CMAKE_FLAGS}
        ${EIGEN_CMAKE_FLAGS}
        ${CERES_CMAKE_FLAGS}
        ${SUITESPARSE_CMAKE_FLAGS}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
        # checking CXX_COMPILER_ID is not supported.
        $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}>
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_GPU=OFF
        -DBUILD_apps=OFF
        -DBUILD_examples=OFF
        -DBUILD_surface_on_nurbs=ON
        -DQT_QMAKE_EXECUTABLE:PATH=${QT5_ROOT_PATH}/bin/qmake
        -DCMAKE_PREFIX_PATH:PATH=${QT5_ROOT_PATH}/lib/cmake
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS ${VTK_TARGET} ext_vtk
        BUILD_BYPRODUCTS
            ${PCL_BUILD_BYPRODUCTS}
)

ExternalProject_Get_Property(ext_pcl INSTALL_DIR)
set(PCL_INCLUDE_DIRS "${INSTALL_DIR}/include/pcl-${PCL_VERSION}/")
set(PCL_LIB_DIR ${INSTALL_DIR}/lib)

set(PCL_CMAKE_FLAGS  ${EIGEN_CMAKE_FLAGS} ${VTK_CMAKE_FLAGS} ${SUITESPARSE_CMAKE_FLAGS} ${CERES_CMAKE_FLAGS} -DPcl_DIR=${INSTALL_DIR}/share/pcl-${PCL_VERSION})
