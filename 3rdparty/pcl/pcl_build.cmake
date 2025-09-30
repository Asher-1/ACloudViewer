include(ExternalProject)

if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for pcl")
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for pcl")
endif ()

set(PCL_RELEASE_SUFFIX "")
set(PCL_DEBUG_SUFFIX "d")
if(WIN32)
    set(PCL_LIB_SUFFIX $<$<CONFIG:Debug>:d>)
else()
    set(PCL_LIB_SUFFIX "")
endif()

set(PCL_VERSION 1.14.1)
set(PCL_MAJOR_VERSION 1.14)

set(PCL_LIBRARIES   pcl_io${PCL_LIB_SUFFIX}
                    pcl_ml${PCL_LIB_SUFFIX}
                    pcl_common${PCL_LIB_SUFFIX}
                    pcl_io_ply${PCL_LIB_SUFFIX}
                    pcl_keypoints${PCL_LIB_SUFFIX}
                    pcl_tracking${PCL_LIB_SUFFIX}
                    # pcl_outofcore${PCL_LIB_SUFFIX}
                    pcl_octree${PCL_LIB_SUFFIX}
                    pcl_kdtree${PCL_LIB_SUFFIX}
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
    list(APPEND PCL_BUILD_BYPRODUCTS <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${item}${CMAKE_STATIC_LIBRARY_SUFFIX}.${PCL_MAJOR_VERSION})
endforeach()

if (BUILD_WITH_CONDA)
    if (WIN32)
        SET(CONDA_LIB_DIR ${CONDA_PREFIX}/Library)
    else ()
        SET(CONDA_LIB_DIR ${CONDA_PREFIX}/lib ${CONDA_PREFIX}/lib/cmake)
    endif()
else()
    set(CONDA_LIB_DIR "")
endif()

ExternalProject_Add(ext_pcl
        PREFIX pcl
        URL https://github.com/PointCloudLibrary/pcl/releases/download/pcl-${PCL_VERSION}/source.zip
        URL_HASH MD5=91cb583c1d5ecf221ee16a736c0ae39d
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/pcl"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            ${ExternalProject_CMAKE_ARGS_hidden}
            ${VTK_CMAKE_FLAGS}
            ${EIGEN_CMAKE_FLAGS}
            -DBUILD_SHARED_LIBS=ON
            -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
            # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
            # checking CXX_COMPILER_ID is not supported.
            $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}>
            # Fix Failed to find shared boost libs installed with conda issues.
            -DPCL_ALLOW_BOTH_SHARED_AND_STATIC_DEPENDENCIES=ON
            -DPCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32=ON
            -DPCL_BUILD_WITH_FLANN_DYNAMIC_LINKING_WIN32=ON
            -DPCL_BUILD_WITH_QHULL_DYNAMIC_LINKING_WIN32=ON
            # -DPCL_SHARED_LIBS=OFF
            # -DPCL_ENABLE_MARCHNATIVE=OFF
            # WITH
            -DWITH_DAVIDSDK=OFF
            -DWITH_DOCS=OFF
            -DWITH_DSSDK=OFF
            -DWITH_ENSENSO=OFF
            -DWITH_OPENNI=OFF
            -DBUILD_GPU=OFF
            -DBUILD_apps=OFF
            -DBUILD_tools=OFF
            -DBUILD_examples=OFF
            -DWITH_VTK=ON
            -DWITH_PNG=ON
            -DWITH_QHULL=ON
            -DBUILD_surface_on_nurbs=ON
            -DCMAKE_PREFIX_PATH=${CONDA_LIB_DIR}
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS 3rdparty_eigen3 3rdparty_vtk
        BUILD_BYPRODUCTS
            ${PCL_BUILD_BYPRODUCTS})

ExternalProject_Get_Property(ext_pcl INSTALL_DIR)
set(PCL_INCLUDE_DIRS "${INSTALL_DIR}/include/pcl-${PCL_MAJOR_VERSION}/")
set(PCL_LIBRARY_DIRS ${INSTALL_DIR}/lib)
set(PCL_CMAKE_FLAGS  ${EIGEN_CMAKE_FLAGS} ${VTK_CMAKE_FLAGS} -DPCL_DIR=${INSTALL_DIR}/CMake)
set(PCL_DIR ${INSTALL_DIR}/CMake)
set(PCL_DEFINITIONS "-D__SSE4_2__;-D__SSE4_1__;-D__SSSE3__;-D__SSE3__;-D__SSE2__;-D__SSE__;-DEIGEN_HAS_CXX17_OVERALIGN=0;-DBOOST_ALL_NO_LIB")

if (WIN32)
    copy_shared_library(ext_pcl
                        LIB_DIR   ${INSTALL_DIR}/bin
                        LIBRARIES ${PCL_LIBRARIES})
endif()