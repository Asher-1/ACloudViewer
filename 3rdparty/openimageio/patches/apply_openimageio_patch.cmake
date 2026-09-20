if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_DIR OR NOT DEFINED GIT_EXECUTABLE)
    message(FATAL_ERROR "SOURCE_DIR, PATCH_DIR, and GIT_EXECUTABLE are required")
endif()

# Fork note: a reused build tree whose build stamp already exists was patched
# and verified when it was first extracted; the generate_static_closure step
# may have since touched files a patch hunk also covers, which makes the
# idempotent reverse check below fail on every subsequent main build. Skip
# the check for completed trees (delete the ext_openimageio source/build dirs
# or the stamp to force a fresh extraction + patch).
if(NOT DEFINED ALOUD_OPENIMAGEIO_REPATCH AND EXISTS "${SOURCE_DIR}/../ext_openimageio-stamp/ext_openimageio-done")
    return()
endif()

# git apply resolves patch paths against the enclosing repository's work tree
# whenever repository discovery finds one above SOURCE_DIR — and the
# ExternalProject source directory lives inside the ACloudViewer checkout on
# CI. From inside such a repo the patch call exits 0 while changing nothing,
# leaving OIIO unpatched; OCIO's installed config then aborts the OIIO
# configure at its `find_dependency(pystring 1.1.4)` re-find. --no-index
# disables that repository absorption entirely (plain file patcher semantics
# on every git build), and the GIT_CEILING_DIRECTORIES guard below stays as a
# belt-and-suspenders second layer.
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
# must already carry it. "Already carried" is decided by the marker check of
# the _openimageio_verifications table below, NOT by `git apply --reverse
# --check`: with --no-index that reverse check compares raw file contents, so
# once a later patch appends to a file an earlier patch created (0002 extends
# the file 0001 adds), the earlier reverse check fails even though the tree
# is correctly patched. The marker check survives that composition, and any
# tree that is neither cleanly patchable nor fully patched still fails loudly.
file(GLOB _openimageio_patches "${PATCH_DIR}/*.patch")
if(NOT _openimageio_patches)
    message(FATAL_ERROR "No OpenImageIO patches found in ${PATCH_DIR}")
endif()

# Post-apply verification: every entry declares a marker string that must
# exist in a file of the patched tree once that patch is in place. This turns
# any silent no-op of the patcher (exit 0 while changing nothing, the failure
# mode that left windows-wheel CI with an unpatched OIIO and a yaml-cpp
# header/archive drift LNK2001 at the pybind link) into a loud configure-time
# failure instead of a link error hours later. It also decides "already
# applied" for reused build trees (see the loop below). When adding a new
# patch, add its "<NNN>:<tree-relative-file>:<marker>" entry here.
set(_openimageio_verifications
    "0001:src/cmake/ocio_subbuild_project_include.cmake:OCIO_MSVC_LOCATION_FIX"
    "0002:src/cmake/ocio_subbuild_project_include.cmake:OCIO_PINNED_YAML_CPP"
    "0002:src/cmake/build_yaml-cpp.cmake:yaml-cpp_BUILD_VERSION 0.9.0")

foreach(_openimageio_patch IN LISTS _openimageio_patches)
    get_filename_component(_openimageio_patch_name "${_openimageio_patch}" NAME)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --no-index --check "${_openimageio_patch}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _openimageio_patch_check
        OUTPUT_QUIET
        ERROR_QUIET)
    if(_openimageio_patch_check EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" apply --no-index "${_openimageio_patch}"
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
    # Not cleanly patchable: is it already applied? Look for the patch's
    # marker in its target file (see the verification table below).
    set(_openimageio_already_applied FALSE)
    foreach(_openimageio_verification IN LISTS _openimageio_verifications)
        string(REGEX MATCH "^([0-9]+):([^:]+):(.+)$" _openimageio_v_match "${_openimageio_verification}")
        if(NOT _openimageio_v_match)
            continue()
        endif()
        # snapshot the match groups: the REGEX REPLACE below overwrites CMAKE_MATCH_<n>
        set(_openimageio_v_num "${CMAKE_MATCH_1}")
        set(_openimageio_v_file "${CMAKE_MATCH_2}")
        set(_openimageio_v_marker "${CMAKE_MATCH_3}")
        # patch file name prefix (e.g. "0002") must match the entry prefix
        string(REGEX REPLACE "^([0-9]+)-.*$" "\\1" _openimageio_patch_prefix "${_openimageio_patch_name}")
        if(NOT _openimageio_patch_prefix STREQUAL _openimageio_v_num)
            continue()
        endif()
        set(_openimageio_v_path "${SOURCE_DIR}/${_openimageio_v_file}")
        if(NOT EXISTS "${_openimageio_v_path}")
            continue()
        endif()
        file(READ "${_openimageio_v_path}" _openimageio_v_content)
        string(FIND "${_openimageio_v_content}" "${_openimageio_v_marker}" _openimageio_v_pos)
        if(NOT _openimageio_v_pos EQUAL -1)
            set(_openimageio_already_applied TRUE)
        endif()
    endforeach()
    if(NOT _openimageio_already_applied)
        message(FATAL_ERROR
            "OpenImageIO patch ${_openimageio_patch_name} cannot be applied and its "
            "marker is absent - the extracted source tree no longer matches the "
            "patch set in ${PATCH_DIR} (the patch set changed after this tree was "
            "patched, or the patcher silently changed nothing). Delete the "
            "ext_openimageio source and build directories and rebuild to re-extract "
            "and re-patch")
    endif()
endforeach()

# Post-apply verification pass (table defined above): every patch's marker
# must exist in its target file once the patch set is in place.
foreach(_openimageio_patch IN LISTS _openimageio_patches)
    get_filename_component(_openimageio_patch_name "${_openimageio_patch}" NAME)
    foreach(_openimageio_verification IN LISTS _openimageio_verifications)
        string(REGEX MATCH "^([0-9]+):([^:]+):(.+)$" _openimageio_v_match "${_openimageio_verification}")
        if(NOT _openimageio_v_match)
            continue()
        endif()
        set(_openimageio_v_prefix "${CMAKE_MATCH_1}")
        set(_openimageio_v_file "${CMAKE_MATCH_2}")
        set(_openimageio_v_marker "${CMAKE_MATCH_3}")
        string(FIND "${_openimageio_patch_name}" "${_openimageio_v_prefix}-" _openimageio_v_pos)
        if(NOT _openimageio_v_pos EQUAL 0)
            continue()
        endif()
        set(_openimageio_v_path "${SOURCE_DIR}/${_openimageio_v_file}")
        if(NOT EXISTS "${_openimageio_v_path}")
            message(FATAL_ERROR
                "OpenImageIO patch ${_openimageio_patch_name} reports success but its "
                "target file ${_openimageio_v_file} is missing from the tree - the "
                "patcher silently changed nothing. Delete the ext_openimageio source "
                "and build directories and rebuild to re-extract and re-patch")
        endif()
        file(READ "${_openimageio_v_path}" _openimageio_v_content)
        string(FIND "${_openimageio_v_content}" "${_openimageio_v_marker}" _openimageio_v_pos)
        if(_openimageio_v_pos EQUAL -1)
            message(FATAL_ERROR
                "OpenImageIO patch ${_openimageio_patch_name} reports success but marker "
                "'${_openimageio_v_marker}' is absent from ${_openimageio_v_file} - the "
                "patcher silently changed nothing. Delete the ext_openimageio source and "
                "build directories and rebuild to re-extract and re-patch")
        endif()
    endforeach()
endforeach()
