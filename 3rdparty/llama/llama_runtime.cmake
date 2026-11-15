# Shared backend/install contract for the ggml and llama source owners.
# Owned by llama; the old ../ggml/ggml.cmake entry was retired when every
# task re-anchored its numeric baseline on the 0.27 embedded ggml.
# Exports: ${GGML_INCLUDE_DIRS}
# Exports: ${GGML_LIB_DIR}
# Exports: ${GGML_LIBRARIES}
# Exports: ${GGML_DEPENDENCY_TARGET}, ${GGML_EXTERNAL_TARGET}

include(ExternalProject)
set(_LLAMA_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}")

# GGML_ENABLED and backend toggles are owned by cmake/AICoreOptions.cmake (AICore_*).
# aicore_sync_options_to_ggml() must run before including this file.
macro(_ggml_read_synced_cache var type default help)
    if(NOT DEFINED CACHE{${var}})
        set(${var} ${default} CACHE ${type} "${help}" FORCE)
        mark_as_advanced(${var})
    endif()
endmacro()

_ggml_read_synced_cache(GGML_ENABLED BOOL ON
    "Internal: synced from AICore_ENABLED")

if(NOT GGML_ENABLED)
    message(STATUS "ggml is disabled")
    return()
endif()

if(TARGET 3rdparty_ggml OR TARGET 3rdparty_llama)
    message(FATAL_ERROR "AICore requires exactly one ggml source owner; do not include both dependency entries")
endif()

# Targets do not survive a second CMake configure. Bind both generated outputs
# and the shared dependency prefix to one archive so switching entries cannot
# leave incompatible, identically named core libraries or backend modules.
function(_ggml_bind_source_identity)
    set(_identity "${_GGML_OWNER_TARGET}\nSHA256=${GGML_SHA256}\n")
    set(_build_marker "${CMAKE_BINARY_DIR}/.acloudviewer-ggml-source-owner")
    set(_install_marker "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/.acloudviewer-ggml-source-owner")
    file(MAKE_DIRECTORY "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}")
    file(LOCK "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/.acloudviewer-ggml-source-owner.lock"
        GUARD FUNCTION TIMEOUT 10)
    foreach(_marker IN ITEMS "${_build_marker}" "${_install_marker}")
        if(EXISTS "${_marker}")
            file(READ "${_marker}" _previous_identity)
            if(NOT "${_previous_identity}" STREQUAL "${_identity}")
                message(FATAL_ERROR
                    "ggml source identity differs at ${_marker}; "
                    "use separate build and dependency installation directories "
                    "for each source owner/archive")
            endif()
        endif()
    endforeach()
    # Older production ggml builds have no receipt. They may keep their old
    # entry, but the llama candidate must start from an attributable prefix.
    if(_GGML_OWNER_TARGET STREQUAL "ext_llama" AND NOT EXISTS "${_install_marker}")
        file(GLOB _unidentified_ggml
            "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include/ggml*.h"
            "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}/*ggml*"
            "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/bin/*ggml*")
        if(_unidentified_ggml)
            message(FATAL_ERROR
                "llama.cpp refuses an unidentified ggml installation at "
                "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}; use a clean, separate "
                "dependency installation directory")
        endif()
    endif()
    foreach(_marker IN ITEMS "${_build_marker}" "${_install_marker}")
        if(NOT EXISTS "${_marker}")
            file(WRITE "${_marker}" "${_identity}")
        endif()
    endforeach()
endfunction()
_ggml_bind_source_identity()


# Dynamic backend mode: accelerator backends are built as separate shared libraries
# loaded at runtime (GGML_BACKEND_DL). Fixed ON for AICore — see AICoreOptions sync.
_ggml_read_synced_cache(GGML_BUILD_SHARED BOOL ON
    "Internal: AICore requires dynamic ggml backend modules")

# Reuse project-wide ExternalProject_CMAKE_ARGS_hidden (includes compiler,
# build type, MSVC runtime, PIC, visibility, CUDA compiler, etc.)
set(GGML_CMAKE_ARGS
    ${ExternalProject_CMAKE_ARGS_hidden}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DGGML_BUILD_TESTS=OFF
    -DGGML_BUILD_EXAMPLES=OFF
    ${_GGML_OWNER_CMAKE_ARGS}
)

if(GGML_BUILD_SHARED)
    list(APPEND GGML_CMAKE_ARGS -DBUILD_SHARED_LIBS=ON -DGGML_STATIC=OFF)
    list(APPEND GGML_CMAKE_ARGS -DGGML_BACKEND_DL=ON)
    if(WIN32)
        # MSVC links import libraries from lib/, while the corresponding
        # runtime DLLs are installed to bin/.
        set(_GGML_LIB_PREFIX ${CMAKE_IMPORT_LIBRARY_PREFIX})
        set(_GGML_LIB_SUFFIX ${CMAKE_IMPORT_LIBRARY_SUFFIX})
        set(_GGML_RUNTIME_SUBDIR bin)
    else()
        set(_GGML_LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
        set(_GGML_LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(_GGML_RUNTIME_SUBDIR ${CloudViewer_INSTALL_LIB_DIR})
    endif()
else()
    list(APPEND GGML_CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON)
    set(_GGML_LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(_GGML_LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(_GGML_RUNTIME_SUBDIR ${CloudViewer_INSTALL_LIB_DIR})
endif()

include("${_LLAMA_RUNTIME_DIR}/cmake/backends.cmake")

# --- Build byproducts ---
# Link artifacts are import libraries on Windows shared builds, otherwise the
# shared/static libraries themselves.
set(GGML_BUILD_BYPRODUCTS
    ${_GGML_OWNER_BUILD_BYPRODUCTS}
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml${_GGML_LIB_SUFFIX}
    <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-base${_GGML_LIB_SUFFIX}
)
if(GGML_BUILD_SHARED)
    # GGML_BACKEND_DL mode: all backends (including ggml-cpu) are MODULE
    # libraries installed to bin/ with .so extension (even on macOS).
    set(_GGML_MODULE_SUFFIX ".so")
    if(WIN32)
        set(_GGML_MODULE_SUFFIX ".dll")
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}ggml${CMAKE_SHARED_LIBRARY_SUFFIX}
            <INSTALL_DIR>/bin/${CMAKE_SHARED_LIBRARY_PREFIX}ggml-base${CMAKE_SHARED_LIBRARY_SUFFIX}
        )
    endif()
    if(GGML_CPU_ALL_VARIANTS)
        # Baseline CPU module always built; extra ISA variants are discovered at
        # packaging time via libggml-cpu-* glob (compiler-adaptive patch).
        if(APPLE)
            set(_GGML_CPU_BYPRODUCT ggml-cpu-apple_m1)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)$")
            set(_GGML_CPU_BYPRODUCT ggml-cpu-armv8.0_1)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            set(_GGML_CPU_BYPRODUCT ggml-cpu-x64)
        else()
            set(_GGML_CPU_BYPRODUCT ggml-cpu)
        endif()
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}${_GGML_CPU_BYPRODUCT}${_GGML_MODULE_SUFFIX})
    else()
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-cpu${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_CUDA_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-cuda${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_METAL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-metal${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_VULKAN_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-vulkan${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_OPENCL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-opencl${_GGML_MODULE_SUFFIX})
    endif()
    if(_GGML_SYCL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/bin/${_GGML_LIB_PREFIX}ggml-sycl${_GGML_MODULE_SUFFIX})
    endif()
else()
    # Static mode: all backends are regular static libs in lib/.
    list(APPEND GGML_BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cpu${_GGML_LIB_SUFFIX}
    )
    if(_GGML_CUDA_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cuda${_GGML_LIB_SUFFIX})
    endif()
    if(_GGML_METAL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-metal${_GGML_LIB_SUFFIX})
    endif()
    if(_GGML_VULKAN_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-vulkan${_GGML_LIB_SUFFIX})
    endif()
    if(_GGML_OPENCL_ENABLED)
        list(APPEND GGML_BUILD_BYPRODUCTS
            <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-opencl${_GGML_LIB_SUFFIX})
    endif()
endif()

include("${_LLAMA_RUNTIME_DIR}/cmake/patches.cmake")

ExternalProject_Add(${_GGML_OWNER_TARGET}
    PREFIX ${_GGML_OWNER_PREFIX}
    INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
    URL ${GGML_URL}
    URL_HASH SHA256=${GGML_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/${_GGML_OWNER_PREFIX}"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${_GGML_PATCH_COMMAND}
    CMAKE_ARGS ${GGML_CMAKE_ARGS}
    BUILD_BYPRODUCTS ${GGML_BUILD_BYPRODUCTS}
    ${_GGML_EXTERNAL_DEPENDS_ARGS}
)
ExternalProject_Get_Property(${_GGML_OWNER_TARGET} BINARY_DIR)
set(GGML_BUILD_DIR "${BINARY_DIR}" CACHE INTERNAL
    "Selected ggml source owner's configure/build directory" FORCE)

# ALIKED patch (aliked_merged/0001-vulkan-aliked.patch) adds include/ggml-vulkan-aliked.h
# and adds it to GGML_PUBLIC_HEADERS.
# Copy after install so existing build trees pick up the header without a full ggml refetch.
if(_GGML_VULKAN_ENABLED)
    # Generate a helper script at configure time (before the ExternalProject runs)
    # so we can check file existence.  <SOURCE_DIR>/<INSTALL_DIR> are
    # ExternalProject build-time tokens that cannot be used inside -P scripts,
    # so we compute the concrete paths here.
    set(_aliked_src "${CMAKE_CURRENT_BINARY_DIR}/${_GGML_OWNER_PREFIX}/src/${_GGML_OWNER_TARGET}/${_GGML_OWNER_SUBDIR}include/ggml-vulkan-aliked.h")
    set(_aliked_dst "${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include/ggml-vulkan-aliked.h")
    set(_aliked_script "${CMAKE_CURRENT_BINARY_DIR}/ggml_install_aliked_header.cmake")
    file(WRITE "${_aliked_script}"
        "if(EXISTS \"${_aliked_src}\")\n"
        "    get_filename_component(_aliked_dst_dir \"${_aliked_dst}\" DIRECTORY)\n"
        "    file(MAKE_DIRECTORY \"${_aliked_dst_dir}\")\n"
        "    execute_process(COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different\n"
        "        \"${_aliked_src}\" \"${_aliked_dst}\")\n"
        "    message(STATUS \"Installed ggml-vulkan-aliked.h\")\n"
        "else()\n"
        "    message(STATUS \"ggml-vulkan-aliked.h not found in ggml source; skipping (ALIKED Vulkan fallback to CPU)\")\n"
        "endif()\n")
    ExternalProject_Add_Step(${_GGML_OWNER_TARGET} install_aliked_vulkan_header
        COMMAND "${CMAKE_COMMAND}" -P "${_aliked_script}"
        DEPENDEES install
        COMMENT "Installing ggml-vulkan-aliked.h for AICore ALIKED Vulkan"
    )
endif()

set(GGML_INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR})
set(GGML_INCLUDE_DIRS ${GGML_INSTALL_DIR}/include)
set(GGML_LIB_DIR ${GGML_INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(GGML_RUNTIME_LIB_DIR ${GGML_INSTALL_DIR}/${_GGML_RUNTIME_SUBDIR})
set(GGML_MODULE_DIR ${GGML_INSTALL_DIR}/bin)

# --- macOS: fix rpath for backend MODULE libraries ---
# Backend .so modules are dlopen'd at runtime.  On macOS, MODULE targets get
# no LC_RPATH by default.  All backends need @loader_path so they can resolve
# @rpath/libggml-base.0.dylib regardless of who does the dlopen.
if(APPLE AND GGML_BUILD_SHARED)
    set(_GGML_FIX_RPATH_SCRIPT
        "${CMAKE_CURRENT_BINARY_DIR}/fix_macos_backend_rpath.sh")
    file(WRITE "${_GGML_FIX_RPATH_SCRIPT}" "\
#!/bin/bash\n\
set -e\n\
for f in \"$1\"/libggml-*${_GGML_MODULE_SUFFIX}; do\n\
    [ -f \"$f\" ] || continue\n\
    install_name_tool -add_rpath @loader_path \"$f\" 2>/dev/null || true\n\
done\n")
    ExternalProject_Add_Step(${_GGML_OWNER_TARGET} fix_macos_rpath
        COMMAND bash "${_GGML_FIX_RPATH_SCRIPT}" "${GGML_MODULE_DIR}"
        DEPENDEES install
        COMMENT "Fixing macOS rpath for ggml backend modules"
    )
endif()

# --- Create the source owner's interface target ---
set(GGML_DEPENDENCY_TARGET "3rdparty_${_GGML_OWNER_PREFIX}")
set(GGML_EXTERNAL_TARGET "${_GGML_OWNER_TARGET}")
add_library(${GGML_DEPENDENCY_TARGET} INTERFACE)
add_dependencies(${GGML_DEPENDENCY_TARGET} ${GGML_EXTERNAL_TARGET})

target_include_directories(${GGML_DEPENDENCY_TARGET} INTERFACE ${GGML_INCLUDE_DIRS})

# In SHARED + GGML_BACKEND_DL mode, ggml-cpu is a MODULE library loaded at
# runtime by ggml_backend_load_all_from_path().  We only link the core shared
# libs (ggml + ggml-base).  In STATIC mode, ggml-cpu is a regular static lib
# and must be linked explicitly.
if(GGML_BUILD_SHARED)
    set(GGML_CORE_LINK_LIBS
        "${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml${_GGML_LIB_SUFFIX}"
        "${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-base${_GGML_LIB_SUFFIX}"
    )
else()
    set(GGML_CORE_LINK_LIBS
        "${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml${_GGML_LIB_SUFFIX}"
        "${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cpu${_GGML_LIB_SUFFIX}"
        "${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-base${_GGML_LIB_SUFFIX}"
    )
endif()
mark_as_advanced(GGML_CORE_LINK_LIBS)

target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE ${GGML_CORE_LINK_LIBS})

# --- GPU backend linking ---
# In SHARED mode, GPU backends are separate .so/.dylib files loaded at runtime
# via ggml_backend_load_all(). They are not linked into the owner target, so
# libAICore.so has no hard GPU runtime dependencies (no DT_NEEDED for
# libcuda.so.1, etc.). This is what enables the Python wheel to be imported
# on machines without GPU drivers.
#
# In explicit STATIC mode, backends are linked into AICore and all GPU symbols
# must resolve at process load time. Release app and wheel builds use dynamic
# backends so optional accelerators can fail without breaking CPU inference.

if(NOT GGML_BUILD_SHARED)
    # CUDA: reuse CUDAToolkit already found by the parent project
    if(_GGML_CUDA_ENABLED)
        target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-cuda${_GGML_LIB_SUFFIX}
            CUDA::cuda_driver
            CUDA::cudart
        )
        if(TARGET CUDA::cublas)
            target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE CUDA::cublas)
        endif()
        if(TARGET CUDA::cublasLt)
            target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE CUDA::cublasLt)
        endif()
    endif()

    # Metal (macOS/iOS)
    if(_GGML_METAL_ENABLED)
        target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-metal${_GGML_LIB_SUFFIX}
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
        )
    endif()

    # Vulkan
    if(_GGML_VULKAN_ENABLED)
        target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-vulkan${_GGML_LIB_SUFFIX}
            Vulkan::Vulkan
        )
    endif()

    # OpenCL
    if(_GGML_OPENCL_ENABLED)
        if(NOT TARGET OpenCL::OpenCL)
            if(OpenCL_LIBRARIES AND OpenCL_INCLUDE_DIRS)
                add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                set_target_properties(OpenCL::OpenCL PROPERTIES
                    IMPORTED_LOCATION "${OpenCL_LIBRARIES}"
                    INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}")
            else()
                find_library(_GGML_OPENCL_LIB OpenCL)
                if(_GGML_OPENCL_LIB)
                    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
                    set_target_properties(OpenCL::OpenCL PROPERTIES
                        IMPORTED_LOCATION "${_GGML_OPENCL_LIB}")
                endif()
            endif()
        endif()
        target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE
            ${GGML_LIB_DIR}/${_GGML_LIB_PREFIX}ggml-opencl${_GGML_LIB_SUFFIX}
        )
        if(TARGET OpenCL::OpenCL)
            target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE OpenCL::OpenCL)
        elseif(OpenCL_LIBRARIES)
            target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE ${OpenCL_LIBRARIES})
        endif()
    endif()

else()
    # SHARED + GGML_BACKEND_DL mode: all configured backends (Metal, CUDA, OpenCL,
    # Vulkan) are standalone dynamic modules loaded at runtime via
    # ggml_backend_load_all_from_path(). They are NOT linked into
    # the owner target. Each backend module links its own platform dependencies.
    # CopyGgmlBackends.cmake copies them alongside libAICore, and
    # load_backends_once() resolves the search directory at runtime.
endif()

# OpenMP (ggml defaults GGML_OPENMP=ON; link the runtime when available)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE OpenMP::OpenMP_CXX)
endif()

# --- Platform dependencies ---
find_package(Threads REQUIRED)
target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(${GGML_DEPENDENCY_TARGET} INTERFACE dl m)
endif()

# --- Export variables for downstream consumers ---
set(GGML_FOUND TRUE CACHE BOOL "ggml found" FORCE)
set(GGML_INCLUDE_DIRS ${GGML_INCLUDE_DIRS} CACHE PATH "ggml include dirs" FORCE)
set(GGML_LIBRARIES ${GGML_DEPENDENCY_TARGET} CACHE STRING "ggml libraries" FORCE)
set(GGML_DEPENDENCY_TARGET ${GGML_DEPENDENCY_TARGET} CACHE INTERNAL
    "Interface target of the unique ggml source owner" FORCE)
set(GGML_EXTERNAL_TARGET ${GGML_EXTERNAL_TARGET} CACHE INTERNAL
    "ExternalProject target of the unique ggml source owner" FORCE)
set(GGML_CUDA_ENABLED ${_GGML_CUDA_ENABLED} CACHE BOOL "ggml CUDA backend built" FORCE)
set(GGML_BUNDLE_CUDA_RUNTIME ${GGML_BUNDLE_CUDA_RUNTIME} CACHE BOOL
    "Bundle CUDA runtime into installer for driver-only deployment" FORCE)
set(GGML_VULKAN_ENABLED ${_GGML_VULKAN_ENABLED} CACHE BOOL
    "Read-only result: ON only when GGML_USE_VULKAN=ON and glslc/Vulkan/SPIR-V deps were found" FORCE)
mark_as_advanced(GGML_VULKAN_ENABLED)
set(GGML_OPENCL_ENABLED ${_GGML_OPENCL_ENABLED} CACHE BOOL "ggml OpenCL backend built" FORCE)
set(GGML_SYCL_ENABLED ${_GGML_SYCL_ENABLED} CACHE BOOL "ggml SYCL backend built" FORCE)
set(GGML_METAL_ENABLED ${_GGML_METAL_ENABLED} CACHE BOOL "ggml Metal backend built" FORCE)
set(GGML_DYNAMIC_BACKENDS ${GGML_BUILD_SHARED} CACHE BOOL "ggml backends are dynamic modules" FORCE)
set(GGML_RUNTIME_LIB_DIR "${GGML_RUNTIME_LIB_DIR}" CACHE PATH
    "ggml core runtime library directory" FORCE)
set(GGML_BACKEND_DIR "${GGML_MODULE_DIR}" CACHE PATH
    "ggml dynamic backend module directory" FORCE)
set(GGML_MODULE_SUFFIX "${_GGML_MODULE_SUFFIX}" CACHE STRING
    "ggml dynamic backend module suffix" FORCE)

set(GGML_CPU_ALL_VARIANTS ${GGML_CPU_ALL_VARIANTS} CACHE BOOL
    "ggml CPU micro-architecture variants built" FORCE)
set(_ggml_gpu_backend_module_files)
foreach(_backend IN ITEMS cuda metal vulkan opencl sycl)
    string(TOUPPER "${_backend}" _backend_upper)
    if(_GGML_${_backend_upper}_ENABLED)
        list(APPEND _ggml_gpu_backend_module_files
            "${CMAKE_SHARED_LIBRARY_PREFIX}ggml-${_backend}${_GGML_MODULE_SUFFIX}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _ggml_gpu_backend_module_files)
set(GGML_GPU_BACKEND_MODULE_FILES "${_ggml_gpu_backend_module_files}" CACHE STRING
    "Configured ggml GPU backend module files" FORCE)
set(_ggml_backend_module_files)
if(GGML_CPU_ALL_VARIANTS)
    set(_ggml_backend_module_files ${_ggml_gpu_backend_module_files})
else()
    list(APPEND _ggml_backend_module_files
        "${CMAKE_SHARED_LIBRARY_PREFIX}ggml-cpu${_GGML_MODULE_SUFFIX}")
    list(APPEND _ggml_backend_module_files ${_ggml_gpu_backend_module_files})
endif()
list(REMOVE_DUPLICATES _ggml_backend_module_files)
set(GGML_BACKEND_MODULE_FILES "${_ggml_backend_module_files}" CACHE STRING
    "Configured ggml runtime backend module files (CPU via glob when ALL_VARIANTS)" FORCE)

set(_GGML_BACKEND_LIST "CPU")
if(_GGML_CUDA_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "CUDA")
endif()
if(_GGML_OPENCL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "OpenCL")
endif()
if(_GGML_SYCL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "SYCL")
endif()
if(_GGML_VULKAN_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "Vulkan")
endif()
if(_GGML_METAL_ENABLED)
    list(APPEND _GGML_BACKEND_LIST "Metal")
endif()
set(_GGML_AUTO_ORDER_LIST)
if(_GGML_CUDA_ENABLED)
    list(APPEND _GGML_AUTO_ORDER_LIST "CUDA")
endif()
if(APPLE)
    if(_GGML_METAL_ENABLED)
        list(APPEND _GGML_AUTO_ORDER_LIST "Metal")
    endif()
else()
    if(_GGML_VULKAN_ENABLED)
        list(APPEND _GGML_AUTO_ORDER_LIST "Vulkan")
    endif()
endif()
list(APPEND _GGML_AUTO_ORDER_LIST "CPU")
string(JOIN " -> " _GGML_AUTO_ORDER ${_GGML_AUTO_ORDER_LIST})
string(JOIN ", " _GGML_BACKEND_LIST_TEXT ${_GGML_BACKEND_LIST})
set(GGML_BACKEND_LIST "${_GGML_BACKEND_LIST_TEXT}" CACHE STRING
    "ggml backends built at configure time" FORCE)
set(GGML_AUTO_BACKEND_ORDER "${_GGML_AUTO_ORDER}" CACHE STRING
    "ggml Auto device selection order at runtime" FORCE)
if(GGML_BUILD_SHARED)
    set(_GGML_LINK_MODE "shared (dynamic backend loading)")
else()
    set(_GGML_LINK_MODE "static")
endif()
set(_GGML_STATUS_SUFFIX "")
if(GGML_CPU_ALL_VARIANTS)
    set(_GGML_STATUS_SUFFIX ", CPU all variants ON")
endif()
message(STATUS "ggml ${GGML_VERSION}: backends = ${_GGML_BACKEND_LIST} "
                "(${_GGML_LINK_MODE}, auto runtime order: ${_GGML_AUTO_ORDER}${_GGML_STATUS_SUFFIX})")
message(STATUS "ggml ${GGML_VERSION}: install=${GGML_INSTALL_DIR}, download_dir=${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/${_GGML_OWNER_PREFIX}")
