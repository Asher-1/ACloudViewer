cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

# CMP0079 (CMake 3.13): allow target_link_libraries() from this subdirectory on
# QMANUAL_CALIB_PLUGIN, which is created in the PARENT directory by AddPlugin().
# The subproject minimum version (3.10) would otherwise reset this policy to
# OLD and CMake errors out with "target not built in this directory" (Windows
# CI failure). target_link_options() is not affected by CMP0079.
if(POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()

add_library(MCALIB_BEV_STITCH STATIC)

set(MCALIB_BEV_STITCH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BirdEyeView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BevAlphaFusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LidarProjBackend.cpp
)
# BevRemapOcl.cpp is added conditionally below, only when a usable OpenCL
# development package (library + CL/cl.h header) is detected.

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
            ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapCuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapCudaProbe.cpp)
        find_package(CUDAToolkit REQUIRED)
        # Same CUDA runtime policy as qSIBR / AICore:
        # - UNIX: link cudart STATICALLY so libQMANUAL_CALIB_PLUGIN.so has no
        #   DT_NEEDED on libcudart.so.X and loads on machines without a CUDA
        #   toolkit; the NVIDIA driver (libcuda.so.1) is dlopen'd by cudart
        #   itself only when CUDA APIs are actually called.
        # - Windows: NVIDIA ships no static cudart, keep the shared runtime but
        #   delay-load it (/DELAYLOAD). NOTE: this only protects the runtime
        #   API calls (probed via SEH in BevRemapCudaProbe.cpp); nvcc's
        #   __sti____cudaRegisterAll registers the fat binary during static
        #   initialization, before any probe, so a CUDA-enabled plugin still
        #   fails to load when cudart64_*.dll is absent. This matches qSIBR's
        #   current behavior: CUDA builds require the cudart DLL at load time
        #   on Windows; machines without CUDA should use a CPU build
        #   (BUILD_CUDA_MODULE=OFF, OpenCL/CPU backends only).
        # NO separable compilation: BevRemapCuda.cu is a single translation unit
        # with no cross-file device symbols, so -rdc=true adds nothing here, and
        # on Windows the dynamic cudart.dll does not export the device-link stub
        # __cudaRegisterLinkedBinary (only cudart_static has it), which breaks
        # linking the Qt plugin (pure C++ target, no CUDA device-link step).
        if(WIN32)
            target_link_libraries(MCALIB_BEV_STITCH PRIVATE CUDA::cudart)
            # CUDA::cudart's IMPORTED_LOCATION is the IMPORT LIB (cudart.lib)
            # on Windows - find_library only matches .lib there. /DELAYLOAD
            # however needs the actual RUNTIME DLL name (cudart64_12.dll), so
            # resolve it from the toolkit bin/ directory next to the import
            # lib. Passing "cudart.lib" would be ignored by the linker
            # (LNK4199) and the delay-load would silently not apply.
            set(_mcalib_cudart_dll "")
            get_target_property(_mcalib_cudart_implib CUDA::cudart IMPORTED_LOCATION)
            if(NOT _mcalib_cudart_implib)
                get_target_property(_mcalib_cudart_implib CUDA::cudart IMPORTED_IMPLIB)
            endif()
            if(_mcalib_cudart_implib)
                # .../lib/x64/cudart.lib -> toolkit root .../CUDA/vX.Y
                get_filename_component(_mcalib_cudart_libdir "${_mcalib_cudart_implib}" DIRECTORY)
                get_filename_component(_mcalib_cudart_libparent "${_mcalib_cudart_libdir}" DIRECTORY)
                get_filename_component(_mcalib_cudart_toolkit "${_mcalib_cudart_libparent}" DIRECTORY)
                file(GLOB _mcalib_cudart_dlls
                    "${_mcalib_cudart_toolkit}/bin/cudart64_*.dll"
                    "${_mcalib_cudart_libdir}/cudart64_*.dll"
                    "${_mcalib_cudart_libparent}/cudart64_*.dll")
            endif()
            if(NOT _mcalib_cudart_dlls AND DEFINED ENV{CUDA_PATH})
                file(GLOB _mcalib_cudart_dlls "$ENV{CUDA_PATH}/bin/cudart64_*.dll")
            endif()
            if(_mcalib_cudart_dlls)
                list(GET _mcalib_cudart_dlls 0 _mcalib_cudart_dll_path)
                # Normalize Windows backslashes to forward slashes so CMake
                # does not misinterpret \P, \S, etc. as escape sequences in
                # the generated cmake_install.cmake (e.g. "\Program").
                file(TO_CMAKE_PATH "${_mcalib_cudart_dll_path}" _mcalib_cudart_dll_path)
                get_filename_component(_mcalib_cudart_dll "${_mcalib_cudart_dll_path}" NAME)
                # Install cudart next to ACloudViewer.exe so the packaged app
                # finds it at load time (the DLL is not in the import table
                # thanks to /DELAYLOAD, but the nvcc fatbin register runs
                # during static initialization and still needs it). This is
                # redundant with the global PostInstall.cmake bundle into
                # lib/cuda-runtime/ (PATH-injected by ACloudViewer.bat); both
                # locations are harmless. Note: no dev-run copy here - the
                # CMake configure-time copy landed in the wrong directory on
                # MSVC (bin/ instead of bin/<Config>/) and the plugin target's
                # output dir is not on the DLL search path anyway; dev machines
                # have CUDA_PATH\\bin on PATH.
                install(FILES "${_mcalib_cudart_dll_path}"
                    DESTINATION ${ACloudViewer_DEST_FOLDER})
                target_link_options(QMANUAL_CALIB_PLUGIN PRIVATE "/DELAYLOAD:${_mcalib_cudart_dll}")
                # Plain signature on purpose: AddPlugin() (plugins/cmake/Plugins.cmake)
                # and qManualCalib/CMakeLists.txt already link QMANUAL_CALIB_PLUGIN
                # with the plain signature; mixing keyword/plain signatures on the
                # same target is a hard CMake error (Windows CI failure).
                target_link_libraries(QMANUAL_CALIB_PLUGIN delayimp)
                message(STATUS "qManualCalib bev_stitch: Windows - delay-loading ${_mcalib_cudart_dll}")
            else()
                message(WARNING "qManualCalib bev_stitch: cudart64_*.dll not found next to CUDA::cudart; /DELAYLOAD skipped, the plugin will hard-depend on the cudart DLL at load time")
            endif()
        else()
            target_link_libraries(MCALIB_BEV_STITCH PRIVATE CUDA::cudart_static)
        endif()
        target_compile_definitions(MCALIB_BEV_STITCH PRIVATE MCALIB_BEV_CUDA_ENABLED)
        message(STATUS "qManualCalib bev_stitch: CUDA enabled")
    else()
        message(STATUS "qManualCalib bev_stitch: CUDA compiler not found")
    endif()
elseif(MCALIB_BEV_CUDA AND NOT BUILD_CUDA_MODULE)
    message(STATUS "qManualCalib bev_stitch: CUDA skipped (BUILD_CUDA_MODULE=OFF, OpenCL preferred)")
endif()

set(MCALIB_BEV_OPENCL_FOUND FALSE)
if(MCALIB_BEV_OPENCL)
    if(TARGET CUDA::OpenCL)
        # CUDA build: link CUDA's OpenCL (same dir as cudart). This keeps the
        # runtime search path unambiguous and avoids the CMake warning about
        # libOpenCL.so.1 in /usr/lib/x86_64-linux-gnu being hidden by /usr/local/cuda/lib64.
        # The CUDA toolkit ships the CL/cl.h header, so no extra header check is needed.
        target_link_libraries(MCALIB_BEV_STITCH PRIVATE CUDA::OpenCL)
        set(MCALIB_BEV_OPENCL_FOUND TRUE)
        message(STATUS "qManualCalib bev_stitch: OpenCL enabled (CUDA::OpenCL)")
    else()
        find_package(OpenCL)
        # FindOpenCL matches either CL/cl.h or OpenCL/cl.h; bev_ocl includes
        # <CL/cl.h> on Linux/Windows but <OpenCL/cl.h> on macOS (the Xcode SDK
        # only ships OpenCL/cl.h under usr/include plus the OpenCL.framework
        # Headers), so verify the exact header path for this platform before
        # enabling.
        if(OpenCL_FOUND)
            if(APPLE)
                find_path(MCALIB_OPENCL_CL_H OpenCL/cl.h
                    HINTS ${OpenCL_INCLUDE_DIR}
                    NO_DEFAULT_PATH)
            else()
                find_path(MCALIB_OPENCL_CL_H CL/cl.h
                    HINTS ${OpenCL_INCLUDE_DIR}
                    NO_DEFAULT_PATH)
            endif()
        endif()
        if(OpenCL_FOUND AND MCALIB_OPENCL_CL_H)
            target_link_libraries(MCALIB_BEV_STITCH PRIVATE OpenCL::OpenCL)
            set(MCALIB_BEV_OPENCL_FOUND TRUE)
            message(STATUS "qManualCalib bev_stitch: OpenCL enabled (OpenCL::OpenCL)")
        elseif(OpenCL_FOUND)
            message(STATUS "qManualCalib bev_stitch: OpenCL package found but OpenCL header missing, OpenCL backend disabled")
        else()
            if(APPLE)
                find_library(MCALIB_OPENCL_LIB OpenCL)
            else()
                find_library(MCALIB_OPENCL_LIB NAMES OpenCL)
            endif()
            # The runtime library (e.g. libOpenCL.so.1 shipped by GPU drivers, or the
            # macOS system OpenCL.framework) may exist without the development headers.
            # Only enable the OpenCL backend when BOTH the library and the platform
            # OpenCL header are present; otherwise BevRemapOcl.cpp cannot be compiled.
            # This is what broke the GitHub CI builds on macOS/Linux/Windows.
            if(APPLE)
                find_path(MCALIB_OPENCL_INCLUDE_DIR OpenCL/cl.h)
            else()
                find_path(MCALIB_OPENCL_INCLUDE_DIR CL/cl.h)
            endif()
            if(MCALIB_OPENCL_LIB AND MCALIB_OPENCL_INCLUDE_DIR)
                target_link_libraries(MCALIB_BEV_STITCH PRIVATE ${MCALIB_OPENCL_LIB})
                target_include_directories(MCALIB_BEV_STITCH PRIVATE ${MCALIB_OPENCL_INCLUDE_DIR})
                set(MCALIB_BEV_OPENCL_FOUND TRUE)
                message(STATUS "qManualCalib bev_stitch: OpenCL enabled via ${MCALIB_OPENCL_LIB}")
            elseif(MCALIB_OPENCL_LIB)
                message(STATUS "qManualCalib bev_stitch: OpenCL library found but OpenCL header missing, OpenCL backend disabled")
            else()
                message(STATUS "qManualCalib bev_stitch: OpenCL not found")
            endif()
        endif()
    endif()
    if(MCALIB_BEV_OPENCL_FOUND)
        target_sources(MCALIB_BEV_STITCH PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/BevRemapOcl.cpp)
        target_compile_definitions(MCALIB_BEV_STITCH PRIVATE MCALIB_BEV_OPENCL_ENABLED)
    endif()
endif()
