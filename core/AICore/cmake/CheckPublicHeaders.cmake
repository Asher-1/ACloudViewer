if(NOT IS_DIRECTORY "${AICORE_PUBLIC_INCLUDE_DIR}")
    message(FATAL_ERROR
        "AICore public include directory is missing: ${AICORE_PUBLIC_INCLUDE_DIR}")
endif()

file(GLOB_RECURSE _aicore_public_headers
    "${AICORE_PUBLIC_INCLUDE_DIR}/*.h"
    "${AICORE_PUBLIC_INCLUDE_DIR}/*.hpp")
foreach(_header IN LISTS _aicore_public_headers)
    file(READ "${_header}" _contents)
    if(_contents MATCHES "#[ \t]*include[^\n]*(ggml|gguf)")
        message(FATAL_ERROR
            "AICore public header exposes a private ggml include: ${_header}")
    endif()
endforeach()

message(STATUS "AICore public headers do not expose ggml")
