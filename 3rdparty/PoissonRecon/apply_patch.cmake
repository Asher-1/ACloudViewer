# apply_patch.cmake -- Apply a patch, tolerating "already applied".
#
# patch -N returns exit code 1 when the patch is already applied, even though
# it safely skips.  ExternalProject treats non-zero exit as fatal, so this
# wrapper checks whether the failure is benign.
#
# Required variables (passed via -D):
#   PATCH_EXECUTABLE  -- path to the patch program
#   PATCH_FILE        -- path to the .patch file

if(NOT PATCH_EXECUTABLE OR NOT PATCH_FILE)
    message(FATAL_ERROR "PATCH_EXECUTABLE and PATCH_FILE must be set")
endif()

execute_process(
    COMMAND ${PATCH_EXECUTABLE} -p1 -N -i ${PATCH_FILE}
    RESULT_VARIABLE _patch_result
    OUTPUT_VARIABLE _patch_output
    ERROR_VARIABLE  _patch_output
)

if(_patch_result EQUAL 0)
    message(STATUS "PoissonRecon: patch applied successfully")
elseif(_patch_output MATCHES "previously applied" OR _patch_output MATCHES "Reversed")
    message(STATUS "PoissonRecon: patch already applied, skipping")
else()
    message(FATAL_ERROR "PoissonRecon: patch failed (exit ${_patch_result}):\n${_patch_output}")
endif()
