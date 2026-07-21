cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

add_library(MCALIB_BEV_STITCH STATIC)

set(MCALIB_BEV_STITCH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BirdEyeView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BevAlphaFusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapOcl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LidarProjBackend.cpp
)

target_sources(MCALIB_BEV_STITCH PRIVATE ${MCALIB_BEV_STITCH_SRCS})

target_compile_features(MCALIB_BEV_STITCH PUBLIC cxx_std_17)

target_include_directories(MCALIB_BEV_STITCH
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_definitions(MCALIB_BEV_STITCH PRIVATE _USE_MATH_DEFINES NOMINMAX)
elseif(APPLE)
    target_compile_options(MCALIB_BEV_STITCH PRIVATE -Wall -Wextra)
else()
    target_compile_options(MCALIB_BEV_STITCH PRIVATE -Wall -Wextra)
endif()

target_link_libraries(MCALIB_BEV_STITCH
    PUBLIC
        3rdparty_opencv
        3rdparty_eigen3
        MCALIB_CALIB_IO
    PRIVATE
        CVCoreLib
)

# GPU backend policy (all platforms):
# - CUDA build (BUILD_CUDA_MODULE=ON, non-macOS): prefer CUDA, fallback OpenCL -> CPU
# - CPU build (BUILD_CUDA_MODULE=OFF) or macOS: no CUDA, prefer OpenCL -> CPU
if(APPLE OR NOT BUILD_CUDA_MODULE)
    option(MCALIB_BEV_CUDA "Enable CUDA BEV remap/fusion in qManualCalib bev_stitch" OFF)
else()
    option(MCALIB_BEV_CUDA "Enable CUDA BEV remap/fusion in qManualCalib bev_stitch" ON)
endif()
option(MCALIB_BEV_OPENCL "Enable OpenCL BEV remap/fusion in qManualCalib bev_stitch" ON)

if(MCALIB_BEV_CUDA AND BUILD_CUDA_MODULE)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_sources(MCALIB_BEV_STITCH PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapCuda.cu)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(MCALIB_BEV_STITCH PRIVATE CUDA::cudart)
        target_compile_definitions(MCALIB_BEV_STITCH PRIVATE MCALIB_BEV_CUDA_ENABLED)
        set_target_properties(MCALIB_BEV_STITCH PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        message(STATUS "qManualCalib bev_stitch: CUDA enabled")
    else()
        message(STATUS "qManualCalib bev_stitch: CUDA compiler not found")
    endif()
elseif(MCALIB_BEV_CUDA AND NOT BUILD_CUDA_MODULE)
    message(STATUS "qManualCalib bev_stitch: CUDA skipped (BUILD_CUDA_MODULE=OFF, OpenCL preferred)")
endif()

if(MCALIB_BEV_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        target_link_libraries(MCALIB_BEV_STITCH PRIVATE OpenCL::OpenCL)
        target_compile_definitions(MCALIB_BEV_STITCH PRIVATE MCALIB_BEV_OPENCL_ENABLED)
        message(STATUS "qManualCalib bev_stitch: OpenCL enabled (OpenCL::OpenCL)")
    else()
        if(APPLE)
            find_library(MCALIB_OPENCL_LIB OpenCL)
        else()
            find_library(MCALIB_OPENCL_LIB NAMES OpenCL)
        endif()
        if(MCALIB_OPENCL_LIB)
            target_link_libraries(MCALIB_BEV_STITCH PRIVATE ${MCALIB_OPENCL_LIB})
            target_compile_definitions(MCALIB_BEV_STITCH PRIVATE MCALIB_BEV_OPENCL_ENABLED)
            message(STATUS "qManualCalib bev_stitch: OpenCL enabled via ${MCALIB_OPENCL_LIB}")
        else()
            message(STATUS "qManualCalib bev_stitch: OpenCL not found")
        endif()
    endif()
endif()
