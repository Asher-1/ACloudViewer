# ------------------------------------------------------------------------------
# CGAL BGL iterator.h compatibility patch (affects CGAL <= 5.6.1).
#
# Single-file design (function library + enable wrapper + self test):
#   - Consumers include this file and call acv_enable_cgal_bgl_iterator_patch()
#     right after their find_package(CGAL); everything else is encapsulated.
#   - `cmake -P cmake/CGALBglIteratorPatch.cmake` runs the embedded regression
#     tests (guarded by CMAKE_SCRIPT_MODE_FILE, skipped when included).
#
# Why the bug exists: CGAL's BGL Halfedge_around_{source,target,face}_iterator
# classes always carried a boolean-conversion operator calling a `base()`
# member that does not exist on these non-inheriting classes ("safe bool"
# residue):
#   CGAL <= 5.5:        operator bool_type() const { return (! (this->base() == nullptr)) ? ... ; }
#   CGAL 5.6.0/5.6.1:   explicit operator bool() const { return (! (this->base() == nullptr)); }
# Per the C++ standard ([temp.res]) a member lookup failure on the current
# instantiation must be diagnosed at template-definition time, and clang 19+
# (including AppleClang 21 / Xcode 26) does exactly that: every translation
# unit that merely *includes* <CGAL/boost/graph/iterator.h> fails with
#   error: no member named 'base' in 'Halfedge_around_source_iterator<Graph>'
# even though the operator is never used.  GCC and MSVC only diagnose at
# instantiation time and nothing ever instantiates it, which is why the
# defect went unnoticed upstream.  CGAL 5.6.2 rewrote the operators without
# the dangling base() call; 6.0 finished the cleanup.
#
# What the patch does: detects the broken header (content-based, independent
# of the CGAL version scheme), writes a patched copy under the build tree
# with the broken boolean-conversion operators removed, and prepends an
# include directory that shadows the vendored header.  CGAL >= 5.6.2 / 6.x
# headers contain no dangling base() call and are left untouched, so 5.x and
# 6.x installs are supported side by side.  The generated header is plain C++
# that compiles everywhere, so the shadow is harmless on toolchains that
# never trigger the diagnostic (GCC, MSVC).
#
# Platform notes:
#   - Linux / macOS (Makefile, Ninja, Xcode) and Windows (Visual Studio,
#     Ninja): directory-level `-I` entries are searched before every
#     `-isystem` / `/external:I` entry, so the shadow wins over CGAL's
#     imported-target include dir regardless of how it is propagated.
#   - Line endings: the matcher tolerates CRLF (Windows installs) and any
#     indentation.
#
# Why this lives in its own file instead of core/cmake/CGALSupport.cmake:
# that file is CVCoreLib's CGAL *integration* config with side effects
# (find_package, Windows GMP install steps, SEND_ERROR when CGAL is missing)
# and it is only included when CVCORELIB_USE_CGAL is ON, while
# libs/Reconstruction runs its own find_package(CGAL) and needs the patch in
# every CGAL-enabled configuration.  include() pastes text, so a reusable
# helper must be a side-effect-free file of its own.
# ------------------------------------------------------------------------------

function(acv_patch_cgal_bgl_iterator)
    set(one_value_args OUT_PATCH_DIR)
    set(multi_value_args CGAL_INCLUDE_CANDIDATES)
    cmake_parse_arguments(PATCH "" "${one_value_args}" "${multi_value_args}"
                          ${ARGN})

    if(NOT PATCH_OUT_PATCH_DIR)
        message(FATAL_ERROR
            "acv_patch_cgal_bgl_iterator: OUT_PATCH_DIR is required")
    endif()
    set(${PATCH_OUT_PATCH_DIR} "" PARENT_SCOPE)

    # Normalized candidate list (drop empty entries and duplicates; normalize
    # so that e.g. ${CGAL_DIR}/../../../include collapses onto the same string
    # as ${CONDA_PREFIX}/include before dedup).
    set(_dirs "")
    foreach(_dir IN LISTS PATCH_CGAL_INCLUDE_CANDIDATES)
        if(_dir)
            get_filename_component(_norm_dir "${_dir}" ABSOLUTE)
            list(APPEND _dirs "${_norm_dir}")
        endif()
    endforeach()
    if(_dirs)
        list(REMOVE_DUPLICATES _dirs)
    endif()

    # Find every candidate that ships the broken header (content-based check,
    # independent of the CGAL version scheme).
    set(_bad_dirs "")
    foreach(_dir IN LISTS _dirs)
        if(EXISTS "${_dir}/CGAL/boost/graph/iterator.h")
            file(READ "${_dir}/CGAL/boost/graph/iterator.h" _probe)
            string(FIND "${_probe}" "this->base() == nullptr" _bad_pos)
            if(NOT _bad_pos EQUAL -1)
                list(APPEND _bad_dirs "${_dir}")
            endif()
            unset(_probe)
        endif()
    endforeach()
    if(NOT _bad_dirs)
        return()
    endif()

    # Shadow the first broken install; warn if several coexist (rare).
    list(GET _bad_dirs 0 _cgal_include_dir)
    list(LENGTH _bad_dirs _num_bad)
    if(_num_bad GREATER 1)
        message(WARNING
            "Multiple CGAL installs carry the broken BGL iterator header; "
            "shadowing ${_cgal_include_dir} only (found: ${_bad_dirs}).")
    endif()

    file(READ "${_cgal_include_dir}/CGAL/version.h" _version_h)
    string(REGEX MATCH "CGAL_VERSION_NR[ \t]+([0-9]+)" _match "${_version_h}")
    set(_cgal_version_nr "${CMAKE_MATCH_1}")

    file(READ "${_cgal_include_dir}/CGAL/boost/graph/iterator.h" _header)

    # Remove the broken boolean-conversion operators.  Both historical forms
    # reference the non-existent base() member; dropping them mirrors the
    # upstream 5.6.2/6.0 state and keeps every other member intact.  The
    # whitespace classes accept CRLF and arbitrary indentation.
    string(REGEX REPLACE
        "explicit operator bool[(][)] const[ \t\r\n]*[{][ \t\r\n]*return [(]![ ]?[(]this->base[(][)] == nullptr[)][)];[ \t\r\n]*[}][ \t\r\n]*"
        ""
        _patched "${_header}")
    string(REGEX REPLACE
        "operator bool_type[(][)] const[ \t\r\n]*[{][ \t\r\n]*return [(]![ ]?[(]this->base[(][)] == nullptr[)][)][ \t]*[?][ \t\r\n]*&[A-Za-z_][A-Za-z0-9_]*::this_type_does_not_support_comparisons : 0;[ \t\r\n]*[}][ \t\r\n]*"
        ""
        _patched "${_patched}")

    # Safety net: the copy must no longer contain the dangling call at all.
    string(FIND "${_patched}" "this->base() == nullptr" _bad_after)
    if(NOT _bad_after EQUAL -1)
        message(WARNING
            "CGAL BGL iterator patch: unexpected header layout at "
            "${_cgal_include_dir}; leaving it untouched. Please report this.")
        return()
    endif()

    set(_patch_dir "${CMAKE_BINARY_DIR}/cgal_bgl_iterator_patch/include")
    set(_dst_header "${_patch_dir}/CGAL/boost/graph/iterator.h")
    # Rewrite only on content change to avoid bumping the mtime (and thereby
    # recompiling dependents) on every re-configure.
    set(_need_write TRUE)
    if(EXISTS "${_dst_header}")
        file(READ "${_dst_header}" _existing)
        if(_existing STREQUAL _patched)
            set(_need_write FALSE)
        endif()
    endif()
    if(_need_write)
        file(MAKE_DIRECTORY "${_patch_dir}/CGAL/boost/graph")
        file(WRITE "${_dst_header}" "${_patched}")
    endif()

    message(STATUS
        "Patched CGAL ${_cgal_version_nr} BGL iterator.h (dangling base() in operator bool) -> ${_dst_header}")

    set(${PATCH_OUT_PATCH_DIR} "${_patch_dir}" PARENT_SCOPE)
endfunction()

# One-call wiring for consumers: run right after find_package(CGAL).  The
# candidate list encodes where CGAL headers are found across the platforms we
# build (conda prefix, vcpkg/system prefixes, CGAL_DIR layout); the
# include_directories call acts on the caller's directory scope, prepending
# the shadow dir ahead of CGAL's imported-target `-isystem` entry.
function(acv_enable_cgal_bgl_iterator_patch)
    acv_patch_cgal_bgl_iterator(
        CGAL_INCLUDE_CANDIDATES
            ${CGAL_INCLUDE_DIRS}
            ${CGAL_INCLUDE_DIR}
            ${CGAL_DIR}/../../../include
            ${CONDA_PREFIX}/include
        OUT_PATCH_DIR ACV_CGAL_BGL_PATCH_DIR)
    if(ACV_CGAL_BGL_PATCH_DIR)
        include_directories(BEFORE ${ACV_CGAL_BGL_PATCH_DIR})
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Self test: `cmake -P cmake/CGALBglIteratorPatch.cmake`
# Runs only in script mode; consumers that include() this file skip it.
# Covers every known header form (CGAL <= 5.5 safe-bool, 5.6.0/5.6.1 explicit
# operator, >= 5.6.2 rewritten operators), CRLF line endings, duplicate
# include candidates and multiple broken installs.  Exits non-zero on failure.
#
# Note: samples are built with string(APPEND) on purpose -- round-tripping
# C++ text through a CMake list would eat the semicolons inside the code.
# ------------------------------------------------------------------------------
if(CMAKE_SCRIPT_MODE_FILE)
    get_filename_component(_test_here "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
    set(_tmp "${_test_here}/../.cache/cgal_patch_test_tmp")
    file(REMOVE_RECURSE "${_tmp}")
    file(MAKE_DIRECTORY "${_tmp}")
    set(CMAKE_BINARY_DIR "${_tmp}/build")

    set(_pass 0)
    set(_fail 0)

    function(_make_cgal out_dir name header_content)
        set(dir "${_tmp}/${name}/include")
        file(MAKE_DIRECTORY "${dir}/CGAL/boost/graph")
        file(WRITE "${dir}/CGAL/boost/graph/iterator.h" "${header_content}")
        file(WRITE "${dir}/CGAL/version.h"
            "#define CGAL_VERSION 9.9.9\n#define CGAL_VERSION_NR 1050611000\n")
        set(${out_dir} "${dir}" PARENT_SCOPE)
    endfunction()

    function(_count str needle out_var)
        string(REGEX MATCHALL "${needle}" _matches "${str}")
        list(LENGTH _matches _n)
        set(${out_var} "${_n}" PARENT_SCOPE)
    endfunction()

    string(APPEND _sample_explicit
        "class Halfedge_around_source_iterator {\n"
        "  explicit operator bool() const\n"
        "  {\n"
        "    return (! (this->base() == nullptr));\n"
        "  }\n"
        "};\n"
        "class Halfedge_around_target_iterator {\n"
        "  explicit operator bool() const\n"
        "  {\n"
        "    return (! (this->base() == nullptr));\n"
        "  }\n"
        "};\n"
        "class Halfedge_around_face_iterator {\n"
        "  explicit operator bool() const\n"
        "  {\n"
        "    return (! (this->base() == nullptr));\n"
        "  }\n"
        "};\n"
        "#endif /* CGAL_BGL_ITERATORS_H */\n")

    string(APPEND _sample_safebool
        "class Halfedge_around_source_iterator {\n"
        "  operator bool_type() const\n"
        "  {\n"
        "    return (! (this->base() == nullptr)) ?\n"
        "      &Halfedge_around_source_iterator::this_type_does_not_support_comparisons : 0;\n"
        "  }\n"
        "};\n"
        "class Halfedge_around_target_iterator {\n"
        "  operator bool_type() const\n"
        "  {\n"
        "    return (! (this->base() == nullptr)) ?\n"
        "      &Halfedge_around_target_iterator::this_type_does_not_support_comparisons : 0;\n"
        "  }\n"
        "};\n"
        "class Halfedge_around_face_iterator {\n"
        "  operator bool_type() const\n"
        "  {\n"
        "    return (! (this->base() == nullptr)) ?\n"
        "      &Halfedge_around_face_iterator::this_type_does_not_support_comparisons : 0;\n"
        "  }\n"
        "};\n"
        "#endif /* CGAL_BGL_ITERATORS_H */\n")

    string(APPEND _sample_fixed
        "class Halfedge_around_source_iterator {\n"
        "  explicit operator bool() const\n"
        "  {\n"
        "    return (! (this->base_reference() == nullptr));\n"
        "  }\n"
        "};\n"
        "#endif /* CGAL_BGL_ITERATORS_H */\n")

    # Case 1: explicit form is patched, nothing else is touched.
    _make_cgal(d1 explicit_561 "${_sample_explicit}")
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d1}" OUT_PATCH_DIR p1)
    set(ok FALSE)
    if(p1)
        file(READ "${p1}/CGAL/boost/graph/iterator.h" out1)
        _count("${out1}" "this->base[(][)] == nullptr" bad1)
        _count("${out1}" "operator bool" bools1)
        _count("${out1}" "class Halfedge" classes1)
        if(bad1 EQUAL 0 AND bools1 EQUAL 0 AND classes1 EQUAL 3)
            set(ok TRUE)
        endif()
    endif()
    if(ok)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case1 (explicit form): p1=${p1} bad=${bad1} bools=${bools1} classes=${classes1}")
    endif()

    # Case 2: safe-bool form is patched.
    _make_cgal(d2 safebool_55 "${_sample_safebool}")
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d2}" OUT_PATCH_DIR p2)
    set(ok FALSE)
    if(p2)
        file(READ "${p2}/CGAL/boost/graph/iterator.h" out2)
        _count("${out2}" "this->base[(][)] == nullptr" bad2)
        _count("${out2}" "operator bool_type" bools2)
        _count("${out2}" "class Halfedge" classes2)
        if(bad2 EQUAL 0 AND bools2 EQUAL 0 AND classes2 EQUAL 3)
            set(ok TRUE)
        endif()
    endif()
    if(ok)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case2 (safe-bool form): p2=${p2} bad=${bad2} bool_type=${bools2} classes=${classes2}")
    endif()

    # Case 3: mixed forms in one header (both matchers apply).
    _make_cgal(d3 mixed "${_sample_explicit}${_sample_safebool}")
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d3}" OUT_PATCH_DIR p3)
    set(ok FALSE)
    if(p3)
        file(READ "${p3}/CGAL/boost/graph/iterator.h" out3)
        _count("${out3}" "this->base[(][)] == nullptr" bad3)
        if(bad3 EQUAL 0)
            set(ok TRUE)
        endif()
    endif()
    if(ok)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case3 (mixed forms): p3=${p3} bad=${bad3}")
    endif()

    # Case 4: CRLF input (Windows checkouts) is handled.
    string(REPLACE "\n" "\r\n" _sample_crlf "${_sample_explicit}")
    _make_cgal(d4 crlf_561 "${_sample_crlf}")
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d4}" OUT_PATCH_DIR p4)
    set(ok FALSE)
    if(p4)
        file(READ "${p4}/CGAL/boost/graph/iterator.h" out4)
        _count("${out4}" "this->base[(][)] == nullptr" bad4)
        if(bad4 EQUAL 0)
            set(ok TRUE)
        endif()
    endif()
    if(ok)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case4 (CRLF): p4=${p4} bad=${bad4}")
    endif()

    # Case 5: healthy header (CGAL >= 5.6.2) is left alone.
    _make_cgal(d5 fixed_562 "${_sample_fixed}")
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d5}" OUT_PATCH_DIR p5)
    if(p5 STREQUAL "")
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case5 (healthy header): expected empty patch dir, got ${p5}")
    endif()

    # Case 6: duplicate candidates collapse into a single patch run.
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d1}" "${d1}"
        OUT_PATCH_DIR p6)
    if(p1 AND p6 STREQUAL p1)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case6 (duplicates): p6=${p6} expected ${p1}")
    endif()

    # Case 7: two broken installs -> first is shadowed (plus a WARNING).
    _make_cgal(d7 second_bad "${_sample_explicit}")
    acv_patch_cgal_bgl_iterator(CGAL_INCLUDE_CANDIDATES "${d7}" "${d1}"
        OUT_PATCH_DIR p7)
    if(p1 AND p7 STREQUAL p1)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case7 (multiple installs): p7=${p7} expected ${p1}")
    endif()

    # Case 8: the enable wrapper inside a real project() context -- this is
    # the production wiring.  include_directories() is not scriptable, so the
    # directory-scope behavior can only be verified via a child configure.
    _make_cgal(d8 enable_ok "${_sample_explicit}")
    set(_proj8 "${_tmp}/proj8")
    file(MAKE_DIRECTORY "${_proj8}")
    file(WRITE "${_proj8}/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.24)
project(Proj8 NONE)
include(\"${CMAKE_CURRENT_LIST_FILE}\")
set(CGAL_INCLUDE_DIRS \"${d8}\")
acv_enable_cgal_bgl_iterator_patch()
get_directory_property(_incs INCLUDE_DIRECTORIES)
list(FIND _incs \"\${CMAKE_BINARY_DIR}/cgal_bgl_iterator_patch/include\" _at)
if(NOT _at EQUAL 0)
    message(FATAL_ERROR \"enable wrapper did not prepend the patch dir: \${_incs}\")
endif()")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${_proj8}" -B "${_proj8}/build"
        RESULT_VARIABLE _rv
        ERROR_VARIABLE _err)
    if(_rv EQUAL 0)
        math(EXPR _pass "${_pass}+1")
    else()
        math(EXPR _fail "${_fail}+1")
        message("FAIL case8 (enable wrapper): ${_err}")
    endif()

    message("=====================================================")
    message("CGAL BGL iterator patch tests: ${_pass} passed, ${_fail} failed")
    message("=====================================================")
    if(NOT _fail EQUAL 0)
        message(FATAL_ERROR "CGAL BGL iterator patch tests FAILED")
    endif()
endif()
