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
