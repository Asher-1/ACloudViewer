if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE OR NOT DEFINED GIT_EXECUTABLE)
    message(FATAL_ERROR "SOURCE_DIR, PATCH_FILE, and GIT_EXECUTABLE are required")
endif()

# git apply resolves patch paths against the enclosing repository's work tree
# whenever repository discovery finds one above SOURCE_DIR — and the
# ExternalProject source directory lives inside the ACloudViewer checkout on
# CI. From inside such a repo the patch call exits 0 while changing nothing,
# leaving OIIO unpatched; OCIO's installed config then aborts the OIIO
# configure at its `find_dependency(pystring 1.1.4)` re-find. Run from
# SOURCE_DIR and ceiling repository discovery at the discovered work-tree root
# so git behaves like a plain patcher wherever the build tree sits.
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _openimageio_toplevel_result
    OUTPUT_VARIABLE _openimageio_work_tree
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(_openimageio_toplevel_result EQUAL 0 AND _openimageio_work_tree)
    set(ENV{GIT_CEILING_DIRECTORIES} "${_openimageio_work_tree}")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _openimageio_patch_check
    OUTPUT_QUIET
    ERROR_QUIET)
if(_openimageio_patch_check EQUAL 0)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _openimageio_patch_apply
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NOT _openimageio_patch_apply EQUAL 0)
        message(FATAL_ERROR "Failed to apply OpenImageIO compatibility patch")
    endif()
    return()
endif()

# Reused build trees have already consumed the patch. Verify that it is the
# expected patch instead of silently accepting a changed upstream source tree.
execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _openimageio_reverse_check
    OUTPUT_QUIET
    ERROR_QUIET)
if(NOT _openimageio_reverse_check EQUAL 0)
    message(FATAL_ERROR "OpenImageIO compatibility patch cannot be applied or verified")
endif()
