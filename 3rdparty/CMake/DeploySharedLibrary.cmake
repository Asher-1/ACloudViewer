# DeploySharedLibrary.cmake
#
# Copy a built shared library into the runtime directory under the name the
# dynamic loader will ask for.
#
#   cmake "-DLIBRARY=<libX.so>" "-DDESTINATION_DIR=<bin>" "-DREADELF=<readelf>"
#         -P DeploySharedLibrary.cmake
#
# "<libX.so>" in an install tree is the unversioned development symlink. On
# Linux the loader never asks for that name: DT_NEEDED carries the library's
# DT_SONAME (e.g. libopencv_world.so.407). "cmake -E copy_if_different <dir>/"
# dereferences the symlink but keeps the ".so" name, so the deployed file is
# unloadable. Read DT_SONAME and name the copy after it; a library without a
# SONAME keeps its own name (that is what the linker recorded for it).

foreach(_required LIBRARY DESTINATION_DIR)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "DeploySharedLibrary: ${_required} is required")
    endif()
endforeach()
if(NOT EXISTS "${LIBRARY}")
    message(FATAL_ERROR "DeploySharedLibrary: ${LIBRARY} does not exist")
endif()

get_filename_component(_filename "${LIBRARY}" NAME)
set(_deployed_name "${_filename}")
set(_soname_pos -1)
if(READELF)
    execute_process(COMMAND "${READELF}" -d "${LIBRARY}"
                    RESULT_VARIABLE _readelf_result
                    OUTPUT_VARIABLE _readelf_output
                    ERROR_QUIET)
    # Extract "[<soname>]" from the "(SONAME)" line positionally: a CMake
    # regex over the whole dump can span lines and pick up RUNPATH instead,
    # and the RUNPATH line holds brackets of its own, so cut at end of line.
    if(_readelf_result EQUAL 0)
        string(FIND "${_readelf_output}" "(SONAME)" _soname_pos)
    endif()
    if(NOT _soname_pos EQUAL -1)
        string(SUBSTRING "${_readelf_output}" ${_soname_pos} -1 _soname_tail)
        string(FIND "${_soname_tail}" "\n" _eol)
        string(SUBSTRING "${_soname_tail}" 0 ${_eol} _soname_line)
        string(FIND "${_soname_line}" "[" _open)
        string(FIND "${_soname_line}" "]" _close)
        if(NOT _open EQUAL -1 AND _close GREATER _open)
            math(EXPR _len "${_close} - ${_open} - 1")
            math(EXPR _open "${_open} + 1")
            string(SUBSTRING "${_soname_line}" ${_open} ${_len} _deployed_name)
        endif()
    endif()
endif()

file(MAKE_DIRECTORY "${DESTINATION_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${LIBRARY}" "${DESTINATION_DIR}/${_deployed_name}"
                RESULT_VARIABLE _copy_result)
if(NOT _copy_result EQUAL 0)
    message(FATAL_ERROR "DeploySharedLibrary: failed to deploy ${LIBRARY}")
endif()

# Trees built before the SONAME lookup hold an unloadable unversioned copy.
if(NOT _deployed_name STREQUAL _filename)
    file(REMOVE "${DESTINATION_DIR}/${_filename}")
endif()
