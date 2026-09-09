if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_DIR OR NOT DEFINED GIT_EXECUTABLE)
    message(FATAL_ERROR "SOURCE_DIR, PATCH_DIR, and GIT_EXECUTABLE are required")
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

# Apply every numbered patch in file(GLOB) order. Each patch is idempotent:
# a freshly extracted source tree gets it applied, while a reused build tree
# must already carry it — verified via the reverse check instead of silently
# accepting a changed upstream source tree.
file(GLOB _openimageio_patches "${PATCH_DIR}/*.patch")
if(NOT _openimageio_patches)
    message(FATAL_ERROR "No OpenImageIO patches found in ${PATCH_DIR}")
endif()
foreach(_openimageio_patch IN LISTS _openimageio_patches)
    get_filename_component(_openimageio_patch_name "${_openimageio_patch}" NAME)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check "${_openimageio_patch}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _openimageio_patch_check
        OUTPUT_QUIET
        ERROR_QUIET)
    if(_openimageio_patch_check EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" apply "${_openimageio_patch}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE _openimageio_patch_apply
            OUTPUT_QUIET
            ERROR_QUIET)
        if(NOT _openimageio_patch_apply EQUAL 0)
            message(FATAL_ERROR
                "Failed to apply OpenImageIO patch ${_openimageio_patch_name}")
        endif()
        continue()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_openimageio_patch}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _openimageio_reverse_check
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NOT _openimageio_reverse_check EQUAL 0)
        message(FATAL_ERROR
            "OpenImageIO patch ${_openimageio_patch_name} cannot be applied or verified. "
            "The extracted source tree no longer matches the patch set in "
            "${PATCH_DIR} (the patch set changed after this tree was patched). "
            "Delete the ext_openimageio source and build directories and rebuild "
            "to re-extract and re-patch")
    endif()
endforeach()
