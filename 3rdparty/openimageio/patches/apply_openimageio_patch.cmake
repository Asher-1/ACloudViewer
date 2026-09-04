if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE OR NOT DEFINED GIT_EXECUTABLE)
    message(FATAL_ERROR "SOURCE_DIR, PATCH_FILE, and GIT_EXECUTABLE are required")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --unsafe-paths --directory=${SOURCE_DIR} --check "${PATCH_FILE}"
    RESULT_VARIABLE _openimageio_patch_check
    OUTPUT_QUIET
    ERROR_QUIET)
if(_openimageio_patch_check EQUAL 0)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --unsafe-paths --directory=${SOURCE_DIR} "${PATCH_FILE}"
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
    COMMAND "${GIT_EXECUTABLE}" apply --unsafe-paths --directory=${SOURCE_DIR} --reverse --check "${PATCH_FILE}"
    RESULT_VARIABLE _openimageio_reverse_check
    OUTPUT_QUIET
    ERROR_QUIET)
if(NOT _openimageio_reverse_check EQUAL 0)
    message(FATAL_ERROR "OpenImageIO compatibility patch cannot be applied or verified")
endif()
