# Generate a linker response file listing the complete static closure of the
# source-built OpenImageIO after its ExternalProject install finished.
#
# Static archives record no dependency information, so linking libOpenImageIO.a
# into a shared deliverable (pybind module, app executable) requires every
# static library OIIO was built against: the OIIO install tree, OIIO's local
# dependencies (deps/dist), OpenColorIO's ext dependencies (pystring, expat,
# minizip-ng under deps/OpenColorIO-build/ext/dist), and the repo-pinned
# ext_zlib archive. Enumerating those names by hand is fragile -- several
# libraries carry OIIO-internal versioned suffixes (e.g.
# libImath_v_3_1_10_OIIO.a) and the ext set moves between dist roots -- so
# this script globs the actual build products after the build and writes them
# into a response file consumed by the linker through `-Wl,@<rsp>`. The file
# is regenerated on every OIIO (re)build, so version upgrades never require
# manual list maintenance.
#
# Inputs:
#   OIIO_INSTALL_LIB   <install-prefix>/lib of the OIIO ExternalProject
#   OIIO_BINARY_DIR    binary dir of the OIIO ExternalProject
#   LIB_PREFIX         e.g. "lib" (or empty under MSVC)
#   LIB_SUFFIX         e.g. ".a" (or ".lib" under MSVC; with the all-static
#                      dep policy no DLL/import-lib pair exists, so every
#                      archive in these directories is a static library)
#   EXTRA_LIBS         optional additional archives (repo-pinned ext_zlib)
#   OUT_RSP            response file to write
#
# The dependency list is emitted twice: the linker scans archives in order and
# pulls members only for still-unresolved symbols, so an archive must come
# after every archive that depends on it. The dependency graph is not a
# topologically-sorted list under any single natural ordering (e.g. the
# alphabetical order of deps/dist puts libImath before the OpenColorIO archive
# that references it), and duplicating the list is the standard robust fix
# that costs nothing: unused scans pull no members.
#
# MSVC (MERGE_LIB_TOOL set to lib.exe by openimageio.cmake) cannot consume
# this rsp: link.exe expands no @file inside a response file, and MSBuild and
# the Ninja generator both wrap the link line in their own response file, so
# an @rsp link item would arrive already nested and fail with
# "LNK1104: cannot open file '@...rsp.lib'". For MSVC this script therefore
# MERGES every closure archive into one static library (oiio_static_closure.lib)
# which the consumer links as a plain library item. MSVC resolves archive
# members with repeated scans, so the merged archive needs neither the
# doubled entries nor any ordering.

# This script runs via `cmake -P` (no project() scope), so policies default
# to OLD. IN_LIST (CMP0057) is required by the MSVC dedup below.
cmake_policy(SET CMP0057 NEW)

if(NOT EXISTS "${OIIO_INSTALL_LIB}/${LIB_PREFIX}OpenImageIO${LIB_SUFFIX}")
    message(FATAL_ERROR
        "generate_static_closure: OpenImageIO static library missing: "
        "${OIIO_INSTALL_LIB}/${LIB_PREFIX}OpenImageIO${LIB_SUFFIX}")
endif()
if(NOT EXISTS "${OIIO_INSTALL_LIB}/${LIB_PREFIX}OpenImageIO_Util${LIB_SUFFIX}")
    message(FATAL_ERROR
        "generate_static_closure: OpenImageIO_Util static library missing: "
        "${OIIO_INSTALL_LIB}/${LIB_PREFIX}OpenImageIO_Util${LIB_SUFFIX}")
endif()

set(_rsp_entries
    "${OIIO_INSTALL_LIB}/${LIB_PREFIX}OpenImageIO${LIB_SUFFIX}"
    "${OIIO_INSTALL_LIB}/${LIB_PREFIX}OpenImageIO_Util${LIB_SUFFIX}")

foreach(_deps_dir IN ITEMS "${OIIO_BINARY_DIR}/deps/dist/lib"
                           "${OIIO_BINARY_DIR}/deps/OpenColorIO-build/ext/dist/lib")
    if(IS_DIRECTORY "${_deps_dir}")
        file(GLOB _dep_archives "${_deps_dir}/*${LIB_SUFFIX}")
        list(APPEND _rsp_entries ${_dep_archives})
        list(APPEND _rsp_entries ${_dep_archives}) # second resolution pass
    endif()
endforeach()

foreach(_extra IN LISTS EXTRA_LIBS)
    if(NOT EXISTS "${_extra}")
        message(FATAL_ERROR
            "generate_static_closure: pinned static library missing: ${_extra}")
    endif()
    list(APPEND _rsp_entries "${_extra}")
endforeach()

# NB: deliberately NO REMOVE_DUPLICATES — the doubled entries ARE the two
# resolution passes. The linker scans archives in order and pulls members only
# for still-unresolved symbols; a single pass breaks whenever the natural
# (alphabetical) order places a dependency before its dependents — deps/dist
# puts libIex before the libOpenEXR that references it, which is exactly what
# an end-to-end link test caught. Repeated paths cost nothing: a pass that
# satisfies no new symbols pulls no members. Same-basename archives from
# different directories (two libz.a copies) are distinct paths and both stay.
list(LENGTH _rsp_entries _rsp_count)
list(JOIN _rsp_entries "\n" _rsp_content)
file(WRITE "${OUT_RSP}" "${_rsp_content}\n")
message(STATUS
    "generate_static_closure: wrote ${OUT_RSP} (${_rsp_count} archives)")

if(NOT DEFINED MERGE_LIB_TOOL OR MERGE_LIB_TOOL STREQUAL "")
    return()
endif()

if(NOT EXISTS "${MERGE_LIB_TOOL}")
    message(FATAL_ERROR
        "generate_static_closure: archive merge tool not found: "
        "${MERGE_LIB_TOOL}")
endif()

# MSVC path: merge the closure into one archive. The OIIO install-tree
# archives (OpenImageIO, OpenImageIO_Util) are EXCLUDED: the consumer already
# links them as standalone items through import_3rdparty_library, and having
# the same archive enter the link through two paths would only add
# LNK4006-style duplicate-member warnings and dead weight to the merge.
#
# Inputs are deduplicated by basename -- the doubled two-pass list would
# embed duplicate object members, and the same basename can legitimately
# appear twice through different roots (deps/dist's zlibstatic.lib vs the
# repo-pinned ext_zlib zlibstatic.lib in EXTRA_LIBS). Both copies are the
# same pinned zlib 1.3.1, so keeping the first and dropping the rest loses
# no symbol: MSVC resolves every consumer's zlib references against the
# single surviving member set. MSVC resolves archive members with repeated
# scans, so the merged archive needs neither ordering nor repeats.
set(_install_tree "${OIIO_INSTALL_LIB}/")
set(_merged_entries "")
set(_seen_basenames "")
foreach(_entry IN LISTS _rsp_entries)
    # string(FIND), not MATCHES: the install path can contain regex
    # metacharacters (version dirs like openimageio-3.1.17.0).
    string(FIND "${_entry}" "${_install_tree}" _install_pos)
    if(_install_pos EQUAL 0)
        continue()
    endif()
    get_filename_component(_base "${_entry}" NAME)
    if(NOT _base IN_LIST _seen_basenames)
        list(APPEND _seen_basenames "${_base}")
        list(APPEND _merged_entries "${_entry}")
    endif()
endforeach()
list(LENGTH _merged_entries _merged_count)
if(_merged_count EQUAL 0)
    message(FATAL_ERROR
        "generate_static_closure: no dependency archives discovered under "
        "${OIIO_BINARY_DIR}/deps; the static closure would be empty")
endif()

set(_merged_lib "${OIIO_INSTALL_LIB}/oiio_static_closure.lib")
file(REMOVE "${_merged_lib}")
execute_process(
    COMMAND "${MERGE_LIB_TOOL}" "/OUT:${_merged_lib}" ${_merged_entries}
    RESULT_VARIABLE _merge_result
    OUTPUT_VARIABLE _merge_output
    ERROR_VARIABLE _merge_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    )
if(NOT _merge_result EQUAL 0)
    message(STATUS "${_merge_output}")
    message(FATAL_ERROR
        "generate_static_closure: failed to merge the OIIO static closure "
        "into ${_merged_lib} with ${MERGE_LIB_TOOL} (exit ${_merge_result})")
endif()
if(NOT EXISTS "${_merged_lib}")
    message(FATAL_ERROR
        "generate_static_closure: ${MERGE_LIB_TOOL} reported success but "
        "produced no output at ${_merged_lib}")
endif()
message(STATUS
    "generate_static_closure: merged ${_merged_count} archives into "
    "${_merged_lib}")
