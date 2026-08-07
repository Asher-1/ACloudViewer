# Cross-platform FFmpeg 8.x compatibility patch applier for OpenCV.
# Usage: cmake -DSRC_DIR=<opencv_source> -DPATCH_FILE=<patch_path> -P apply_ffmpeg8_patch.cmake
#
# Idempotent: performs a dry-run first; if the patch is already applied the
# real apply is skipped.  Works on Linux, macOS, and Windows (the `patch`
# utility is bundled with Git for Windows and available via package managers
# on Linux/macOS — all platforms that build ACloudViewer).

if(NOT DEFINED SRC_DIR OR NOT DEFINED PATCH_FILE)
    message(FATAL_ERROR "SRC_DIR and PATCH_FILE must be defined")
endif()

if(NOT EXISTS "${PATCH_FILE}")
    message(STATUS "[opencv-patch] patch file not found: ${PATCH_FILE} (skipping)")
    return()
endif()

# Dry-run: check whether the patch can still be applied.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E chdir "${SRC_DIR}"
            patch -p1 -N --dry-run -i "${PATCH_FILE}"
    RESULT_VARIABLE _dry_rc
    OUTPUT_QUIET ERROR_QUIET
)

if(NOT _dry_rc EQUAL 0)
    message(STATUS "[opencv-patch] already applied or not applicable: ${PATCH_FILE}")
    return()
endif()

# Apply for real.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E chdir "${SRC_DIR}"
            patch -p1 -N -i "${PATCH_FILE}"
    RESULT_VARIABLE _apply_rc
    OUTPUT_VARIABLE _apply_out
    ERROR_VARIABLE  _apply_out
)

if(NOT _apply_rc EQUAL 0)
    message(FATAL_ERROR "[opencv-patch] FAILED to apply ${PATCH_FILE}:\n${_apply_out}")
endif()

message(STATUS "[opencv-patch] applied: ${PATCH_FILE}")
