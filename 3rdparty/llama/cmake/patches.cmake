# Patch dependencies and pristine replay for the selected source owner.
# Included by llama_runtime.cmake before ExternalProject_Add.

# The entry selects the source-version/backend manifest. The common runner
# validates and applies its ordered unified diffs using git. Disabled backend
# sections stay inert; candidate manifests must first port those sections.
set(_GGML_PATCH_SCRIPT
    "${_LLAMA_RUNTIME_DIR}/llama-patches/apply_ggml_patches.py")
set(_GGML_PATCH_MANIFEST
    "${_GGML_OWNER_PATCH_MANIFEST}")
file(GLOB_RECURSE _GGML_PATCH_FILES CONFIGURE_DEPENDS
    "${_GGML_OWNER_PATCH_DIR}/*.patch")
list(SORT _GGML_PATCH_FILES)
set(_GGML_PATCH_INPUTS
    "${_GGML_PATCH_SCRIPT}"
    "${_GGML_PATCH_MANIFEST}"
    ${_GGML_PATCH_FILES})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_GGML_PATCH_INPUTS})

# ExternalProject's patch stamp does not depend on patch file contents. Track a
# content signature explicitly; when it changes, discard only this owner's generated
# source/build/stamp trees so the verified archive is extracted and every patch
# is replayed from a pristine source tree.
set(_GGML_PATCH_SIGNATURE_PAYLOAD
    "ggml=${GGML_VERSION};archive=${GGML_SHA256}")
if(NOT _GGML_OWNER_PREFIX STREQUAL "ggml")
    string(APPEND _GGML_PATCH_SIGNATURE_PAYLOAD
        ";owner=${_GGML_OWNER_PREFIX};subdir=${_GGML_OWNER_SUBDIR}")
endif()
foreach(_patch_input IN LISTS _GGML_PATCH_INPUTS)
    file(SHA256 "${_patch_input}" _patch_input_sha256)
    string(APPEND _GGML_PATCH_SIGNATURE_PAYLOAD
        ";${_patch_input}=${_patch_input_sha256}")
endforeach()
string(SHA256 _GGML_PATCH_SIGNATURE "${_GGML_PATCH_SIGNATURE_PAYLOAD}")
set(_GGML_PATCH_SIGNATURE_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/${_GGML_OWNER_PREFIX}-patch-signature.txt")
set(_GGML_PREVIOUS_PATCH_SIGNATURE "")
if(EXISTS "${_GGML_PATCH_SIGNATURE_FILE}")
    file(READ "${_GGML_PATCH_SIGNATURE_FILE}"
        _GGML_PREVIOUS_PATCH_SIGNATURE)
    string(STRIP "${_GGML_PREVIOUS_PATCH_SIGNATURE}"
        _GGML_PREVIOUS_PATCH_SIGNATURE)
endif()
if(NOT _GGML_PREVIOUS_PATCH_SIGNATURE STREQUAL _GGML_PATCH_SIGNATURE)
    if(_GGML_PREVIOUS_PATCH_SIGNATURE)
        message(STATUS
            "ggml: patch inputs changed; refreshing fetched source tree")
    elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${_GGML_OWNER_PREFIX}/src/${_GGML_OWNER_TARGET}")
        message(STATUS
            "ggml: initializing patch signature; refreshing fetched source tree")
    endif()
    file(REMOVE_RECURSE
        "${CMAKE_CURRENT_BINARY_DIR}/${_GGML_OWNER_PREFIX}/src/${_GGML_OWNER_TARGET}"
        "${CMAKE_CURRENT_BINARY_DIR}/${_GGML_OWNER_PREFIX}/src/${_GGML_OWNER_TARGET}-build"
        "${CMAKE_CURRENT_BINARY_DIR}/${_GGML_OWNER_PREFIX}/src/${_GGML_OWNER_TARGET}-stamp")
    file(REMOVE
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_GGML_OWNER_TARGET}-complete")
    file(WRITE "${_GGML_PATCH_SIGNATURE_FILE}"
        "${_GGML_PATCH_SIGNATURE}\n")
endif()

if(NOT Python3_EXECUTABLE)
    find_package(Python3 QUIET COMPONENTS Interpreter)
endif()
if(Python3_EXECUTABLE)
    set(_ggml_patch_python "${Python3_EXECUTABLE}")
else()
    find_program(_ggml_patch_python NAMES python3 python)
endif()
if(NOT _ggml_patch_python)
    message(FATAL_ERROR
        "ggml: Python is required to apply patches/manifest.yaml")
endif()
find_package(Git REQUIRED)
set(_GGML_PATCH_SOURCE "<SOURCE_DIR>")
if(_GGML_OWNER_SUBDIR)
    string(APPEND _GGML_PATCH_SOURCE "/${_GGML_OWNER_SUBDIR}")
endif()
set(_GGML_PATCH_COMMAND
    "${CMAKE_COMMAND}" -E env "ACV_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "${_ggml_patch_python}" "${_GGML_PATCH_SCRIPT}" "${_GGML_PATCH_SOURCE}"
    "${_GGML_PATCH_MANIFEST}")

set(_GGML_EXTERNAL_DEPENDS_ARGS)
if(_GGML_EXTERNAL_DEPENDS)
    set(_GGML_EXTERNAL_DEPENDS_ARGS DEPENDS ${_GGML_EXTERNAL_DEPENDS})
endif()

