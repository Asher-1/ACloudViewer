# BundleGgmlCudaRuntime.cmake
# Invoke with cmake -P; required variables:
#   GGML_CUDA_MODULE  path to libggml-cuda shared module
#   DEST_DIR          output directory (lib/cuda-runtime)
#   PACK_SCRIPTS_PATH repo scripts/platforms root
# Optional:
#   EXTRA_LIB_DIRS    semicolon-separated search paths (CUDA toolkit lib dirs)

# execute_process passes -D values without a shell; quoted -DVAR=\"path\" leaves
# literal quote characters in GGML_CUDA_MODULE and breaks EXISTS checks.
if(GGML_CUDA_MODULE MATCHES "^\"(.*)\"$")
    set(GGML_CUDA_MODULE "${CMAKE_MATCH_1}")
endif()

if(NOT GGML_CUDA_MODULE OR NOT EXISTS "${GGML_CUDA_MODULE}")
    message(FATAL_ERROR "BundleGgmlCudaRuntime: GGML_CUDA_MODULE missing: ${GGML_CUDA_MODULE}")
endif()
if(NOT DEST_DIR)
    message(FATAL_ERROR "BundleGgmlCudaRuntime: DEST_DIR is required")
endif()
if(NOT PACK_SCRIPTS_PATH)
    message(FATAL_ERROR "BundleGgmlCudaRuntime: PACK_SCRIPTS_PATH is required")
endif()

file(MAKE_DIRECTORY "${DEST_DIR}")

if(UNIX AND NOT APPLE)
    set(_bundle_script "${PACK_SCRIPTS_PATH}/linux/bundle_cuda_runtime.sh")
    if(NOT EXISTS "${_bundle_script}")
        message(FATAL_ERROR "BundleGgmlCudaRuntime: missing ${_bundle_script}")
    endif()
    set(_extra_colon "")
    if(EXTRA_LIB_DIRS)
        string(REPLACE ";" ":" _extra_colon "${EXTRA_LIB_DIRS}")
    endif()
    execute_process(
        COMMAND bash "${_bundle_script}" "${GGML_CUDA_MODULE}" "${DEST_DIR}" "${_extra_colon}"
        RESULT_VARIABLE _bundle_result
        OUTPUT_VARIABLE _bundle_out
        ERROR_VARIABLE _bundle_err
    )
    if(_bundle_out)
        message(STATUS "${_bundle_out}")
    endif()
    if(NOT _bundle_result EQUAL 0)
        message(FATAL_ERROR "BundleGgmlCudaRuntime failed: ${_bundle_err}")
    endif()
elseif(WIN32)
    set(_bundle_script "${PACK_SCRIPTS_PATH}/windows/bundle_cuda_runtime.ps1")
    if(NOT EXISTS "${_bundle_script}")
        message(FATAL_ERROR "BundleGgmlCudaRuntime: missing ${_bundle_script}")
    endif()
    find_program(_POWERSHELL_PATH NAMES powershell pwsh REQUIRED)
    set(_ps_args
        -ExecutionPolicy Bypass
        -File "${_bundle_script}"
        "${GGML_CUDA_MODULE}"
        "${DEST_DIR}"
    )
    if(EXTRA_LIB_DIRS)
        foreach(_dir IN LISTS EXTRA_LIB_DIRS)
            list(APPEND _ps_args "${_dir}")
        endforeach()
    endif()
    execute_process(
        COMMAND "${_POWERSHELL_PATH}" ${_ps_args}
        RESULT_VARIABLE _bundle_result
        OUTPUT_VARIABLE _bundle_out
        ERROR_VARIABLE _bundle_err
    )
    if(_bundle_out)
        message(STATUS "${_bundle_out}")
    endif()
    if(NOT _bundle_result EQUAL 0)
        message(FATAL_ERROR "BundleGgmlCudaRuntime failed: ${_bundle_err}")
    endif()
else()
    message(STATUS "BundleGgmlCudaRuntime: skipped on ${CMAKE_SYSTEM_NAME} (CUDA ggml bundle is Linux/Windows only)")
endif()
