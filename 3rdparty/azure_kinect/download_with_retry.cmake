# Retry helper for third-party ExternalProject downloads hosted on CDNs that
# return transient errors (e.g. HTTP 504 from packages.microsoft.com). A full
# wheel build takes hours, so failing the whole build on one transient download
# error is unacceptable; this script retries with linear backoff and verifies
# the archive hash on every attempt.
#
# Used as an ExternalProject DOWNLOAD_COMMAND override. Note that supplying a
# custom DOWNLOAD_COMMAND also disables ExternalProject's automatic archive
# extraction, so this script extracts the downloaded archive into SOURCE_DIR
# itself (cmake -E tar handles .tar/.zip and .deb via libarchive).
#
# Usage:
#   cmake -DURL=<url>
#         -DFILE_PATH=<download destination>
#         -DEXPECTED_HASH=<ALGO>=<hex>
#         -DSOURCE_DIR=<directory to extract into>
#         -DMAX_RETRIES=<N>
#         -P download_with_retry.cmake

set(_max_retries 3)
if(MAX_RETRIES)
    set(_max_retries ${MAX_RETRIES})
endif()

get_filename_component(_download_dir "${FILE_PATH}" DIRECTORY)
file(MAKE_DIRECTORY "${_download_dir}")
file(MAKE_DIRECTORY "${SOURCE_DIR}")

# Reuse a previously downloaded archive when it still matches the pinned hash.
string(REPLACE "=" ";" _hash_parts "${EXPECTED_HASH}")
list(GET _hash_parts 0 _hash_algo)
list(GET _hash_parts 1 _expected_hex)
if(EXISTS "${FILE_PATH}")
    file(${_hash_algo} "${FILE_PATH}" _existing_hash)
    if(_existing_hash STREQUAL _expected_hex)
        message(STATUS "Reusing cached download: ${FILE_PATH}")
        set(_download_ok TRUE)
    else()
        message(STATUS "Cached download failed hash check, re-downloading: ${FILE_PATH}")
        file(REMOVE "${FILE_PATH}")
        set(_download_ok FALSE)
    endif()
else()
    set(_download_ok FALSE)
endif()

set(_attempt 0)
while(NOT _download_ok AND _attempt LESS _max_retries)
    math(EXPR _attempt "${_attempt} + 1")
    message(STATUS "Downloading (attempt ${_attempt}/${_max_retries}): ${URL}")
    # Deliberately do NOT pass EXPECTED_HASH here: on a failed download CMake
    # aborts with "file DOWNLOAD cannot compute hash on failed download"
    # before the STATUS handling can trigger the retry path. Verify manually
    # below instead.
    file(DOWNLOAD "${URL}" "${FILE_PATH}" STATUS _status)
    list(GET _status 0 _code)
    if(_code EQUAL 0)
        file(${_hash_algo} "${FILE_PATH}" _actual_hash)
        if(_actual_hash STREQUAL _expected_hex)
            set(_download_ok TRUE)
            break()
        endif()
        message(WARNING "Hash mismatch for attempt ${_attempt}/${_max_retries}: "
                        "got ${_actual_hash}, expected ${_expected_hex}")
    else()
        list(GET _status 1 _message)
        message(WARNING "Download attempt ${_attempt}/${_max_retries} failed: ${_message}")
    endif()
    file(REMOVE "${FILE_PATH}")
    if(_attempt LESS _max_retries)
        math(EXPR _backoff "10 * ${_attempt}")
        message(STATUS "Retrying in ${_backoff}s ...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E sleep "${_backoff}")
    endif()
endwhile()

if(NOT _download_ok)
    message(FATAL_ERROR "Download failed after ${_max_retries} attempts: ${URL}")
endif()

# ExternalProject's automatic extract is disabled by the custom download
# command, so extract here to reproduce the default URL-download behavior.
message(STATUS "Extracting ${FILE_PATH} -> ${SOURCE_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${FILE_PATH}"
                WORKING_DIRECTORY "${SOURCE_DIR}"
                RESULT_VARIABLE _extract_result)
if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${FILE_PATH} into ${SOURCE_DIR}")
endif()
