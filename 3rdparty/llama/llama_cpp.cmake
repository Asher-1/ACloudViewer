# llama.cpp source owner for AICore. Backend selection, archive verification,
# patch invalidation, installation and GGML_* exports use the same integration
# as ggml.cmake. Only one of the two source owners may be included.
include_guard(DIRECTORY)
if(DEFINED GGML_ENABLED AND NOT GGML_ENABLED)
    return()
endif()

# Pinned parent archive: its embedded ggml is the only ggml in this build.
# Keep the revision, digest and reported versions together when upgrading.
set(LLAMA_REVISION "707616feb5f379e99ff7442680a55b4a1704c7c9")
set(LLAMA_URL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_REVISION}.tar.gz")
set(LLAMA_SHA256 "a532d0601f65cee41446cfbdc6872b98e689aa727eb366c6b6a61a70e9965759")
set(LLAMA_VERSION "0.7.0-dev")
set(GGML_VERSION "0.27.0")
set(LLAMA_SOURCE_ARCHIVE "" CACHE FILEPATH
    "Optional local llama.cpp source archive; its pinned SHA-256 is still verified")
if(LLAMA_SOURCE_ARCHIVE)
    get_filename_component(LLAMA_URL "${LLAMA_SOURCE_ARCHIVE}" ABSOLUTE)
endif()

# CPU, CUDA and Vulkan have rebased operator manifests. Build support does not certify
# model numerics/performance: production promotion still needs the full gate.
# Reject unported accelerators before fetching rather than disabling them.
if(GGML_USE_METAL OR AICore_USE_METAL)
    if(NOT LLAMA_METAL_MIGRATION_VALIDATION)
        message(FATAL_ERROR
            "llama.cpp embedded ggml: METAL patch migration/validation is incomplete; "
            "use the isolated experimental driver for Metal device validation")
    endif()
    if(NOT APPLE)
        message(FATAL_ERROR "Metal migration validation requires an Apple platform and SDK")
    endif()
    if(GGML_USE_CUDA OR GGML_USE_VULKAN)
        message(FATAL_ERROR "Validate Metal separately from CUDA/Vulkan migration builds")
    endif()
endif()
foreach(_backend IN ITEMS SYCL OPENCL)
    if(GGML_USE_${_backend} OR AICore_USE_${_backend})
        message(FATAL_ERROR
            "llama.cpp embedded ggml: ${_backend} patch migration/validation is incomplete; "
            "no registered migration manifest provides the requested backend")
    endif()
endforeach()
if(NOT DEFINED GGML_USE_VULKAN OR NOT DEFINED GGML_USE_METAL)
    message(FATAL_ERROR
        "Synchronize AICore backend options before including llama_cpp.cmake")
endif()

set(GGML_URL "${LLAMA_URL}")
set(GGML_SHA256 "${LLAMA_SHA256}")
set(_GGML_OWNER_TARGET ext_llama)
set(_GGML_OWNER_PREFIX llama)
set(_GGML_OWNER_SUBDIR "ggml/")
set(_GGML_OWNER_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/experimental/patches")
set(_GGML_OWNER_PATCH_MANIFEST "${_GGML_OWNER_PATCH_DIR}/manifest.yaml")
if(GGML_USE_CUDA)
    set(_GGML_OWNER_PATCH_MANIFEST "${_GGML_OWNER_PATCH_DIR}/cuda-manifest.yaml")
    message(STATUS "llama.cpp: CUDA migration build enabled; full model regression remains required")
endif()
if(GGML_USE_VULKAN)
    set(_GGML_OWNER_PATCH_MANIFEST "${_GGML_OWNER_PATCH_DIR}/vulkan-manifest.yaml")
    if(GGML_USE_CUDA)
        set(_GGML_OWNER_PATCH_MANIFEST "${_GGML_OWNER_PATCH_DIR}/cuda-vulkan-manifest.yaml")
    endif()
    message(STATUS "llama.cpp: Vulkan migration build enabled; full model regression remains required")
endif()
if(GGML_USE_METAL)
    set(_GGML_OWNER_PATCH_MANIFEST "${_GGML_OWNER_PATCH_DIR}/metal-manifest.yaml")
    message(STATUS "llama.cpp: isolated Metal validation; full model regression remains required")
endif()
set(_GGML_OWNER_CMAKE_ARGS
    -DLLAMA_USE_SYSTEM_GGML=OFF
    -DLLAMA_BUILD_COMMIT=${LLAMA_REVISION}
    -DGGML_BUILD_COMMIT=${LLAMA_REVISION}
    -DLLAMA_BUILD_COMMON=OFF
    -DLLAMA_BUILD_TOOLS=OFF
    -DLLAMA_BUILD_MTMD=ON
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_SUBPROCESS=OFF
    -DMTMD_VIDEO=OFF
    -DCMAKE_INSTALL_LIBDIR=${CloudViewer_INSTALL_LIB_DIR})
if(DEFINED GGML_BUILD_SHARED AND NOT GGML_BUILD_SHARED)
    set(_llama_link_prefix "${CMAKE_STATIC_LIBRARY_PREFIX}")
    set(_llama_link_suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
elseif(WIN32)
    set(_llama_link_prefix "${CMAKE_IMPORT_LIBRARY_PREFIX}")
    set(_llama_link_suffix "${CMAKE_IMPORT_LIBRARY_SUFFIX}")
else()
    set(_llama_link_prefix "${CMAKE_SHARED_LIBRARY_PREFIX}")
    set(_llama_link_suffix "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
set(_GGML_OWNER_BUILD_BYPRODUCTS
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_llama_link_prefix}llama${_llama_link_suffix}
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_llama_link_prefix}mtmd${_llama_link_suffix})
if(WIN32 AND (NOT DEFINED GGML_BUILD_SHARED OR GGML_BUILD_SHARED))
    list(APPEND _GGML_OWNER_BUILD_BYPRODUCTS
        <INSTALL_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}llama${CMAKE_SHARED_LIBRARY_SUFFIX}
        <INSTALL_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}mtmd${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

include("${CMAKE_CURRENT_LIST_DIR}/llama_runtime.cmake")

set(LLAMA_INCLUDE_DIRS "${GGML_INCLUDE_DIRS}")
set(LLAMA_LIB_DIR "${GGML_LIB_DIR}")
set(LLAMA_RUNTIME_LIB_DIR "${GGML_RUNTIME_LIB_DIR}")
set(LLAMA_CORE_LINK_LIBS
    "${LLAMA_LIB_DIR}/${_llama_link_prefix}mtmd${_llama_link_suffix}"
    "${LLAMA_LIB_DIR}/${_llama_link_prefix}llama${_llama_link_suffix}")
# The runtime helper created this owner's single interface, including embedded
# ggml. Extend it with llama/mtmd rather than manufacturing a second ggml target.
target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE ${LLAMA_CORE_LINK_LIBS})
set(LLAMA_LIBRARIES ${GGML_DEPENDENCY_TARGET})
set(LLAMA_FOUND TRUE)
message(STATUS "llama.cpp ${LLAMA_VERSION}: source owner ext_llama, embedded ggml ${GGML_VERSION}")
