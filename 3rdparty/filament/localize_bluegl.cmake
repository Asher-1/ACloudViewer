# Localize bluegl's GL entry points in the precompiled filament archive.
#
# bluegl ships real GL definitions (glGetString, ...) as global symbols in
# its assembly object. Linked into an executable they interpose Qt's own GL
# resolution: Qt 6.4 Gui binds glGetString to the still-uninitialized bluegl
# pointer table and segfaults at QApplication init (Qt >= 6.5 resolves GL
# itself and is unaffected). Localizing mirrors the GLEW hidden-visibility
# fix; references from the other filament archives still resolve at
# static-link time, the symbols just stop being exported dynamically.
#
# Invoked by the ext_filament localize-bluegl step with:
#   BLUEGL_ARCHIVE  path to libbluegl.a
#   WORK_DIR        scratch directory for the symbol list

if(NOT EXISTS "${BLUEGL_ARCHIVE}")
    message(FATAL_ERROR "localize_bluegl: archive not found: ${BLUEGL_ARCHIVE}")
endif()

find_program(NM_EXECUTABLE nm)
find_program(OBJCOPY_EXECUTABLE objcopy)
if(NOT NM_EXECUTABLE OR NOT OBJCOPY_EXECUTABLE)
    message(FATAL_ERROR "localize_bluegl: nm/objcopy (binutils) required")
endif()

execute_process(
    COMMAND "${NM_EXECUTABLE}" --defined-only "${BLUEGL_ARCHIVE}"
    OUTPUT_VARIABLE nm_output
    RESULT_VARIABLE nm_rc
)
if(NOT nm_rc EQUAL 0)
    message(FATAL_ERROR "localize_bluegl: nm failed on ${BLUEGL_ARCHIVE}")
endif()

set(gl_syms "")
string(REPLACE "\n" ";" nm_lines "${nm_output}")
foreach(line IN LISTS nm_lines)
    if(line MATCHES "^[0-9a-f]+ T (gl[A-Z][A-Za-z0-9_]*|glX[A-Za-z0-9_]*)$")
        list(APPEND gl_syms "${CMAKE_MATCH_1}")
    endif()
endforeach()

if(gl_syms)
    list(REMOVE_DUPLICATES gl_syms)
    list(SORT gl_syms)
    string(REPLACE ";" "\n" sym_text "${gl_syms}")
    file(WRITE "${WORK_DIR}/bluegl-gl-syms.txt" "${sym_text}\n")
    execute_process(
        COMMAND "${OBJCOPY_EXECUTABLE}" --localize-symbols=${WORK_DIR}/bluegl-gl-syms.txt "${BLUEGL_ARCHIVE}"
        RESULT_VARIABLE objcopy_rc
    )
    if(NOT objcopy_rc EQUAL 0)
        message(FATAL_ERROR "localize_bluegl: objcopy failed on ${BLUEGL_ARCHIVE}")
    endif()
    list(LENGTH gl_syms sym_count)
    message(STATUS "localize_bluegl: localized ${sym_count} GL symbols in ${BLUEGL_ARCHIVE}")
endif()
# An empty list means the archive is already localized (reused build tree);
# nothing to do. binutils objcopy rejects an empty symbol file, so skip it.