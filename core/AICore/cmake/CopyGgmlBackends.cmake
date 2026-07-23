# Copy ggml dynamic backend libraries to the application output directory.
# Invoked as a POST_BUILD script so the glob runs AFTER ExternalProject builds.
#
# Expected variables (passed via -D):
#   GGML_BACKEND_DIR  — source directory containing ggml .dylib/.so files
#   DEST_DIR          — destination directory (typically the bin/ folder)
#   LIB_PREFIX        — platform shared library prefix (e.g. "lib")
#   LIB_SUFFIX        — platform shared library suffix (e.g. ".dylib" or ".so")

if(NOT EXISTS "${GGML_BACKEND_DIR}")
    message(FATAL_ERROR
        "CopyGgmlBackends: source dir does not exist: ${GGML_BACKEND_DIR}")
endif()

if(REQUIRED_FILE AND NOT EXISTS "${GGML_BACKEND_DIR}/${REQUIRED_FILE}")
    message(FATAL_ERROR
        "CopyGgmlBackends: required runtime file is missing: "
        "${GGML_BACKEND_DIR}/${REQUIRED_FILE}")
endif()

if(REQUESTED_FILES_PIPE)
    string(REPLACE "|" ";" _requested_files "${REQUESTED_FILES_PIPE}")
    file(MAKE_DIRECTORY "${DEST_DIR}")
    # Remove every known backend module first, including modules that are no
    # longer supported. This prevents an incremental build from loading a stale
    # libggml-blas left by an older configuration.
    foreach(_backend IN ITEMS cpu blas cuda metal vulkan opencl sycl)
        file(GLOB _stale
            "${DEST_DIR}/${LIB_PREFIX}ggml-${_backend}${LIB_SUFFIX}*")
        if(_stale)
            file(REMOVE ${_stale})
        endif()
    endforeach()
    foreach(_name IN LISTS _requested_files)
        set(_source "${GGML_BACKEND_DIR}/${_name}")
        if(NOT EXISTS "${_source}")
            message(FATAL_ERROR
                "CopyGgmlBackends: configured backend is missing: ${_source}")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_source}" "${DEST_DIR}/${_name}")
        message(STATUS "CopyGgmlBackends: ${_name}")
    endforeach()
    return()
endif()

file(GLOB _ggml_libs "${GGML_BACKEND_DIR}/${LIB_PREFIX}ggml*${LIB_SUFFIX}*")
if(NOT _ggml_libs)
    message(FATAL_ERROR
        "CopyGgmlBackends: no ggml libs found in ${GGML_BACKEND_DIR}")
endif()

file(MAKE_DIRECTORY "${DEST_DIR}")

foreach(_lib ${_ggml_libs})
    get_filename_component(_name "${_lib}" NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" "${DEST_DIR}/${_name}"
    )
    message(STATUS "CopyGgmlBackends: ${_name}")
endforeach()
