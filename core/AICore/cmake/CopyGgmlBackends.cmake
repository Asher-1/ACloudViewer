# Copy ggml dynamic backend libraries to the application output directory.
# Invoked as a POST_BUILD script so the glob runs AFTER ExternalProject builds.
#
# Expected variables (passed via -D):
#   GGML_BACKEND_DIR  — source directory containing ggml .dylib/.so files
#   DEST_DIR          — destination directory (typically the bin/ folder)
#   LIB_PREFIX        — platform shared library prefix (e.g. "lib")
#   LIB_SUFFIX        — platform shared library suffix (e.g. ".dylib" or ".so")

if(NOT EXISTS "${GGML_BACKEND_DIR}")
    message(STATUS "CopyGgmlBackends: source dir does not exist: ${GGML_BACKEND_DIR}")
    return()
endif()

file(GLOB _ggml_libs "${GGML_BACKEND_DIR}/${LIB_PREFIX}ggml*${LIB_SUFFIX}*")
if(NOT _ggml_libs)
    message(STATUS "CopyGgmlBackends: no ggml libs found in ${GGML_BACKEND_DIR}")
    return()
endif()

file(MAKE_DIRECTORY "${DEST_DIR}")

foreach(_lib ${_ggml_libs})
    get_filename_component(_name "${_lib}" NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" "${DEST_DIR}/${_name}"
    )
    message(STATUS "CopyGgmlBackends: ${_name}")
endforeach()
