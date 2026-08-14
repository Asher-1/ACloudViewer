# BundleGgmlCudaRuntime.cmake
# Invoke with cmake -P; required variables:
#   GGML_CUDA_BASE_DIR  build output base directory (e.g. build/bin)
#   GGML_CUDA_NAME      module filename (e.g. ggml-cuda.dll or libggml-cuda.so)
#   DEST_DIR            output directory (lib/cuda-runtime)
#   PACK_SCRIPTS_PATH   repo scripts/platforms root
# Optional:
#   GGML_CUDA_EXT_DIR   ggml ExternalProject install runtime dir (fallback search)
#   EXTRA_LIB_DIRS      semicolon-separated search paths (CUDA toolkit lib dirs)
#   GGML_CUDA_MODULE    legacy: explicit full path (backward compat)

# --- Locate the ggml-cuda module ---
# Search candidate paths: multi-config generators put it in <base>/<Config>/,
# single-config generators put it directly in <base>/.
if(GGML_CUDA_MODULE)
    # Legacy mode: caller provided the full path directly.
    if(GGML_CUDA_MODULE MATCHES "^\"(.*)\"$")
        set(GGML_CUDA_MODULE "${CMAKE_MATCH_1}")
    endif()
elseif(GGML_CUDA_BASE_DIR AND GGML_CUDA_NAME)
    set(_candidate_dirs "${GGML_CUDA_BASE_DIR}")
    foreach(_cfg IN ITEMS Release Debug RelWithDebInfo MinSizeRel)
        list(APPEND _candidate_dirs "${GGML_CUDA_BASE_DIR}/${_cfg}")
    endforeach()
    # ggml ExternalProject install dir (covers CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING=OFF
    # on Linux where AICore lands in build/lib/<Config>/ instead of build/bin/)
    if(GGML_CUDA_EXT_DIR)
        list(APPEND _candidate_dirs "${GGML_CUDA_EXT_DIR}")
    endif()
    set(GGML_CUDA_MODULE "")
    foreach(_dir IN LISTS _candidate_dirs)
        set(_candidate "${_dir}/${GGML_CUDA_NAME}")
        if(EXISTS "${_candidate}")
            set(GGML_CUDA_MODULE "${_candidate}")
            message(STATUS "BundleGgmlCudaRuntime: found ${_candidate}")
            break()
        endif()
    endforeach()
endif()

if(NOT GGML_CUDA_MODULE OR NOT EXISTS "${GGML_CUDA_MODULE}")
    message(FATAL_ERROR "BundleGgmlCudaRuntime: ggml-cuda module not found.\n"
        "  Searched in: ${_candidate_dirs}\n"
        "  GGML_CUDA_BASE_DIR=${GGML_CUDA_BASE_DIR}\n"
        "  GGML_CUDA_NAME=${GGML_CUDA_NAME}\n"
        "  GGML_CUDA_MODULE=${GGML_CUDA_MODULE}")
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
