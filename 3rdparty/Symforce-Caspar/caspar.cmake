if (NOT RECONSTRUCTION_CASPAR_ENABLED)
    return()
endif ()

if (NOT BUILD_CUDA_MODULE)
    message(FATAL_ERROR "RECONSTRUCTION_CASPAR_ENABLED requires BUILD_CUDA_MODULE=ON")
endif ()

foreach (_caspar_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^([0-9]+)" _caspar_arch_num "${_caspar_arch}")
    if (_caspar_arch_num AND _caspar_arch_num LESS 70)
        message(FATAL_ERROR
            "RECONSTRUCTION_CASPAR_ENABLED requires CUDA architecture >= 70, "
            "but CMAKE_CUDA_ARCHITECTURES contains '${_caspar_arch}'.")
    endif ()
endforeach ()

if (RECONSTRUCTION_CASPAR_USE_DOUBLE)
    set(_caspar_precision f64)
else ()
    set(_caspar_precision f32)
endif ()
set(_caspar_generated_dir
    "${CloudViewer_3RDPARTY_DIR}/Symforce-Caspar/generated/${_caspar_precision}")

if (NOT EXISTS "${_caspar_generated_dir}/CMakeLists.txt")
    message(FATAL_ERROR
        "Caspar generated ${_caspar_precision} kernels are missing at "
        "${_caspar_generated_dir}. Regenerate or restore the pinned source tree.")
endif ()

# Generated CMake declares its own project() and CUDA language, so give it a
# dedicated binary directory. The interface target is the only dependency
# exposed to Reconstruction.
add_subdirectory("${_caspar_generated_dir}" "${CMAKE_BINARY_DIR}/caspar/${_caspar_precision}")
add_library(3rdparty_caspar INTERFACE)
target_link_libraries(3rdparty_caspar INTERFACE caspar_lib_core)
target_include_directories(3rdparty_caspar INTERFACE
    "$<BUILD_INTERFACE:${_caspar_generated_dir}>")
target_compile_definitions(3rdparty_caspar INTERFACE CASPAR_ENABLED)
if (RECONSTRUCTION_CASPAR_USE_DOUBLE)
    target_compile_definitions(3rdparty_caspar INTERFACE CASPAR_USE_DOUBLE)
endif ()
