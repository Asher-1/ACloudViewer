# Trellis CuMesh GPU chart clustering — task-private build wiring.
#
# Upstream trellis2cpp third_party/cumesh_glue integration (JeffreyXiang/
# CuMesh, MIT, vendored under src/tasks/trellis/third_party/CuMesh): GPU
# normal-cone chart clustering feeds xatlas' faceMaterialData as hard chart
# boundaries for the UV-atlas GLB bake (mesh_export.cpp's UnwrapMode::Auto).
#
# The user-facing switch and optional TorchConfig.cmake path are declared in
# cmake/AICoreOptions.cmake; everything below is task-owned. Upstream facts
# (trellis-ggml CMakeLists.txt, "CuMesh GPU mesh
# processing"): TRELLIS2_CUMESH defaults ON and CuMesh "links libtorch
# internally; exposes a torch-free C API through cumesh_glue.h" — the
# libtorch dependency is inherent to the vendored CuMesh core (ATen tensor
# plumbing plus the pybind-layer dtype aliases in hash/hash.cu), not an
# ACloudViewer addition. AICore mirrors the upstream default (ON) but degrades
# gracefully — with a loud warning per missing prerequisite, instead of
# upstream's silent disable or a hard configure failure:
#   CUDA off | CUDA toolkit < 12.3 | CUDA-enabled libtorch missing
#   Python development files missing
#     → GPU clustering off; mesh_export tries CPU cone clustering, then
#       simple_unwrap if CPU clustering fails.
# Point CUMESH_TORCH_DIR at the directory containing TorchConfig.cmake
# (or leave it empty to discover Torch from Python3_EXECUTABLE).

include_guard(GLOBAL)

function(aicore_trellis_wire_cumesh target trellis_src_dir)
    set(AICORE_CUMESH_ENABLED ${AICore_USE_CUMESH})
    # TorchConfig disables CMake's architecture generation for its legacy
    # FindCUDA flags. Preserve the consumer's native-CUDA target contract.
    get_target_property(_cumesh_cuda_architectures ${target} CUDA_ARCHITECTURES)
    if(NOT _cumesh_cuda_architectures
       OR _cumesh_cuda_architectures MATCHES "-NOTFOUND$")
        set(_cumesh_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Guard ladder: each step may demote AICORE_CUMESH_ENABLED with a message.
    if(AICORE_CUMESH_ENABLED AND NOT AICore_CUDA_ENABLED)
        message(
            STATUS
                "AICore: CuMesh GPU chart clustering disabled (AICore_USE_CUDA=OFF)"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED)
        find_package(CUDAToolkit QUIET)
        if(NOT CUDAToolkit_FOUND)
            message(
                WARNING
                    "AICore: CuMesh needs the CUDA Toolkit — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
        elseif(CUDAToolkit_VERSION VERSION_LESS 12.3)
            message(
                WARNING
                    "AICore: CuMesh needs CUDA Toolkit 12.3 or newer (found ${CUDAToolkit_VERSION}; vendored CUB code uses in-place DeviceScan added in 12.3) — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
        endif()
    endif()
    if(AICORE_CUMESH_ENABLED
       AND NOT EXISTS "${trellis_src_dir}/third_party/CuMesh/src/cumesh.h")
        message(
            WARNING
                "AICore: missing vendored CuMesh at ${trellis_src_dir}/third_party/CuMesh — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED)
        if(CUMESH_TORCH_DIR)
            if(EXISTS "${CUMESH_TORCH_DIR}/TorchConfig.cmake")
                set(Torch_DIR "${CUMESH_TORCH_DIR}")
            elseif(EXISTS "${CUMESH_TORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
                set(Torch_DIR "${CUMESH_TORCH_DIR}/share/cmake/Torch")
            elseif(EXISTS "${CUMESH_TORCH_DIR}/Torch/TorchConfig.cmake")
                set(Torch_DIR "${CUMESH_TORCH_DIR}/Torch")
            else()
                message(
                    WARNING
                        "AICore: CUMESH_TORCH_DIR does not contain TorchConfig.cmake: ${CUMESH_TORCH_DIR} — building without GPU chart clustering"
                )
                set(AICORE_CUMESH_ENABLED OFF)
            endif()
        elseif(NOT Torch_DIR OR Torch_DIR MATCHES "-NOTFOUND$")
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -c
                        "import torch; print(torch.utils.cmake_prefix_path)"
                RESULT_VARIABLE _cumesh_torch_probe_result
                OUTPUT_VARIABLE _cumesh_torch_cmake_prefix
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(_cumesh_torch_probe_result EQUAL 0
               AND EXISTS "${_cumesh_torch_cmake_prefix}/Torch/TorchConfig.cmake")
                set(Torch_DIR "${_cumesh_torch_cmake_prefix}/Torch")
            endif()
        endif()
        if(AICORE_CUMESH_ENABLED)
            find_package(Torch QUIET CONFIG)
        endif()
    endif()
    if(AICORE_CUMESH_ENABLED AND NOT Torch_FOUND)
        message(
            WARNING
                "AICore: CuMesh needs libtorch (find_package(Torch) failed; install Torch into the selected Python environment or set CUMESH_TORCH_DIR to the directory containing TorchConfig.cmake) — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED AND NOT CAFFE2_USE_CUDA)
        message(
            WARNING
                "AICore: CuMesh needs a CUDA-enabled Torch package — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    # The vendored glue keeps the upstream <torch/extension.h> includes (the
    # upstream glue CMakeLists does the same): it needs the pybind-layer dtype
    # aliases (torch::kUInt32, ...) that do not exist in the ATen header set.
    # Python development files must match the interpreter used for discovery.
    if(AICORE_CUMESH_ENABLED
       AND (NOT Python3_Development_FOUND OR NOT TARGET Python3::Python))
        message(
            WARNING
                "AICore: CuMesh needs Python development headers and a Python import library for ${Python3_EXECUTABLE} — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()

    if(AICORE_CUMESH_ENABLED)
        set(_cumesh_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib")
        set(_cumesh_torch_version_header
            "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include/torch/version.h")
        if(NOT IS_DIRECTORY "${_cumesh_torch_lib_dir}"
           OR NOT EXISTS "${_cumesh_torch_version_header}")
            message(
                WARNING
                    "AICore: Torch package is missing its library directory or version header (prefix=${TORCH_INSTALL_PREFIX}) — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
        endif()
        if(AICORE_CUMESH_ENABLED)
            find_library(_cumesh_torch_python_library
                         NAMES torch_python
                         PATHS "${_cumesh_torch_lib_dir}"
                         NO_DEFAULT_PATH
                         NO_CACHE)
            if(NOT _cumesh_torch_python_library)
                message(
                    WARNING
                        "AICore: Torch package is missing torch_python's link library under ${_cumesh_torch_lib_dir} — building without GPU chart clustering"
                )
                set(AICORE_CUMESH_ENABLED OFF)
            endif()
        endif()
    endif()

    if(AICORE_CUMESH_ENABLED AND WIN32)
        file(GLOB _cumesh_torch_runtime_dlls
             LIST_DIRECTORIES FALSE "${_cumesh_torch_lib_dir}/*.dll")
        if(NOT _cumesh_torch_runtime_dlls)
            message(
                WARNING
                    "AICore: Torch package has no runtime DLLs under ${_cumesh_torch_lib_dir} — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
        endif()
        if(AICORE_CUMESH_ENABLED)
            set(_cumesh_python_runtime_dlls "")
            set(_cumesh_python_runtime_dirs_found "")
            # FindPython already resolves the selected ABI's DLL, including
            # debug/free-threaded names. Do not reconstruct its filename.
            get_target_property(_cumesh_python_configs Python3::Python
                                IMPORTED_CONFIGURATIONS)
            set(_cumesh_python_locations IMPORTED_LOCATION)
            foreach(_cumesh_python_config IN LISTS _cumesh_python_configs)
                string(TOUPPER "${_cumesh_python_config}" _cumesh_python_config)
                list(APPEND _cumesh_python_locations
                     "IMPORTED_LOCATION_${_cumesh_python_config}")
            endforeach()
            foreach(_cumesh_python_location IN LISTS _cumesh_python_locations)
                get_target_property(_cumesh_python_dll Python3::Python
                                    "${_cumesh_python_location}")
                if(_cumesh_python_dll MATCHES "\\.[dD][lL][lL]$"
                   AND EXISTS "${_cumesh_python_dll}")
                    list(APPEND _cumesh_python_runtime_dlls "${_cumesh_python_dll}")
                    get_filename_component(_cumesh_python_runtime_dir
                                           "${_cumesh_python_dll}" DIRECTORY)
                    list(APPEND _cumesh_python_runtime_dirs_found
                         "${_cumesh_python_runtime_dir}")
                endif()
            endforeach()
            list(REMOVE_DUPLICATES _cumesh_python_runtime_dlls)
            list(REMOVE_DUPLICATES _cumesh_python_runtime_dirs_found)
            if(NOT _cumesh_python_runtime_dlls)
                message(
                    WARNING
                        "AICore: Python3::Python does not provide a runtime DLL — building without GPU chart clustering"
                )
                set(AICORE_CUMESH_ENABLED OFF)
            endif()
        endif()
    endif()

    # hash/hash.cu is part of the upstream CUMESH_SRC compile set but needs the
    # ATen UInt32 ScalarType (introduced in torch 2.3 — absent from c10/core/
    # ScalarType.h in 2.2, so "namespace torch has no member kUInt32"). Nothing
    # in the chart-clustering set calls the hashmap, so on older torch the file
    # is skipped instead of failing the build.
    set(_cumesh_hash_src "")
    if(AICORE_CUMESH_ENABLED)
        file(READ
             "${_cumesh_torch_version_header}"
             _cumesh_torch_version_contents)
        string(REGEX MATCH "#define TORCH_VERSION_MAJOR ([0-9]+)" _cumesh_maj
                      "${_cumesh_torch_version_contents}")
        set(_cumesh_torch_maj "${CMAKE_MATCH_1}")
        string(REGEX MATCH "#define TORCH_VERSION_MINOR ([0-9]+)" _cumesh_min
                      "${_cumesh_torch_version_contents}")
        set(_cumesh_torch_min "${CMAKE_MATCH_1}")
        if(_cumesh_torch_maj GREATER 2
           OR (_cumesh_torch_maj EQUAL 2 AND _cumesh_torch_min GREATER_EQUAL 3))
            set(_cumesh_hash_src
                ${trellis_src_dir}/third_party/CuMesh/src/hash/hash.cu)
        else()
            message(
                STATUS
                    "AICore: CuMesh hash/hash.cu skipped (ATen UInt32 needs torch >= 2.3; found ${_cumesh_torch_maj}.${_cumesh_torch_min}) — unused by chart clustering"
            )
        endif()
    endif()

    if(NOT AICORE_CUMESH_ENABLED)
        return()
    endif()

    add_library(
        aicore_cumesh SHARED
        ${trellis_src_dir}/third_party/CuMesh/src/atlas.cu
        ${trellis_src_dir}/third_party/CuMesh/src/clean_up.cu
        ${trellis_src_dir}/third_party/CuMesh/src/connectivity.cu
        ${trellis_src_dir}/third_party/CuMesh/src/cumesh.cu
        ${trellis_src_dir}/third_party/CuMesh/src/geometry.cu
        # See the _cumesh_hash_src version gate above (upstream CUMESH_SRC
        # compiles hash/hash.cu on torch >= 2.3; empty on older torch).
        ${_cumesh_hash_src}
        ${trellis_src_dir}/third_party/CuMesh/src/io.cu
        ${trellis_src_dir}/third_party/CuMesh/src/shared.cu
        ${trellis_src_dir}/third_party/cumesh_glue/cumesh_glue.cu)
    target_compile_definitions(aicore_cumesh PRIVATE CUMESH_GLUE_BUILD=1)
    target_include_directories(
        aicore_cumesh
        PUBLIC "$<BUILD_INTERFACE:${trellis_src_dir}/../../../include>"
        PRIVATE ${trellis_src_dir}
                ${trellis_src_dir}/../..
                ${trellis_src_dir}/third_party/CuMesh/src
                ${trellis_src_dir}/third_party/cumesh_glue
                ${TORCH_INCLUDE_DIRS})
    target_link_libraries(aicore_cumesh PRIVATE Python3::Python)
    set_target_properties(
        aicore_cumesh PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                                 POSITION_INDEPENDENT_CODE ON)
    if(NOT "${_cumesh_cuda_architectures}" STREQUAL "")
        set_target_properties(aicore_cumesh PROPERTIES
                              CUDA_ARCHITECTURES "${_cumesh_cuda_architectures}")
    endif()
    # ATen CUDA helpers rely on relaxed constexpr / extended lambdas.
    target_compile_options(
        aicore_cumesh PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    # TORCH_LIBRARIES is TorchConfig's portable link contract (.so on Unix,
    # import .lib on Windows). Filter only the broken optional entry emitted
    # by some conda packages.
    separate_arguments(_cumesh_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(aicore_cumesh PRIVATE ${_cumesh_torch_cxx_flags})
    set(_cumesh_torch_libraries ${TORCH_LIBRARIES})
    list(FILTER _cumesh_torch_libraries EXCLUDE REGEX "^-lFALSE$")
    target_link_libraries(aicore_cumesh PRIVATE
        ${_cumesh_torch_libraries}
        "${_cumesh_torch_python_library}")

    get_target_property(_cumesh_runtime_dir ${target} RUNTIME_OUTPUT_DIRECTORY)
    get_target_property(_cumesh_library_dir ${target} LIBRARY_OUTPUT_DIRECTORY)
    if(NOT _cumesh_runtime_dir OR _cumesh_runtime_dir MATCHES "-NOTFOUND$")
        set(_cumesh_runtime_dir "${CMAKE_BINARY_DIR}/bin")
    endif()
    if(NOT _cumesh_library_dir OR _cumesh_library_dir MATCHES "-NOTFOUND$")
        set(_cumesh_library_dir "${CMAKE_BINARY_DIR}/bin")
    endif()
    set_target_properties(
        aicore_cumesh PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_cumesh_runtime_dir}"
        LIBRARY_OUTPUT_DIRECTORY "${_cumesh_library_dir}")
    if(UNIX AND NOT APPLE)
        # The installed bridge must resolve bundled Torch/Python libraries
        # beside itself; AICore's RUNPATH does not cover indirect dependencies.
        set_target_properties(aicore_cumesh PROPERTIES
                              INSTALL_RPATH "$ORIGIN;$ORIGIN/cuda-runtime")
        if(TARGET CUDA::cudart)
            # CMake omits CUDA's implicit link directory from automatic
            # RUNPATHs, even when that toolkit is not registered with ldconfig.
            set_property(TARGET aicore_cumesh APPEND PROPERTY BUILD_RPATH
                         "$<TARGET_FILE_DIR:CUDA::cudart>")
        endif()
    endif()

    if(WIN32)
        set_property(GLOBAL PROPERTY AICORE_CUMESH_TORCH_RUNTIME_DIR
                     "${TORCH_INSTALL_PREFIX}/lib")
        set_property(GLOBAL PROPERTY AICORE_CUMESH_PYTHON_RUNTIME_DIRS
                     "${_cumesh_python_runtime_dirs_found}")
        # Keep Torch's shipped CUDA/delay-loaded DLLs in the installer too.
        # The general dependency scanner intentionally filters CUDA runtimes.
        set_property(GLOBAL PROPERTY AICORE_CUMESH_RUNTIME_DLLS
                     "${_cumesh_torch_runtime_dlls};${_cumesh_python_runtime_dlls}")
        add_custom_command(
            TARGET aicore_cumesh
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_cumesh_torch_runtime_dlls}
                    ${_cumesh_python_runtime_dlls}
                    "$<TARGET_RUNTIME_DLLS:aicore_cumesh>"
                    "$<TARGET_FILE_DIR:aicore_cumesh>"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${CLOUDVIEWER_OUTPUT_DIRECTORY}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_cumesh_torch_runtime_dlls}
                    ${_cumesh_python_runtime_dlls}
                    "$<TARGET_RUNTIME_DLLS:aicore_cumesh>"
                    "${CLOUDVIEWER_OUTPUT_DIRECTORY}"
            COMMAND_EXPAND_LISTS
            VERBATIM
            COMMENT "Deploying CuMesh runtime DLL dependencies")
    endif()

    target_compile_definitions(${target} PRIVATE TRELLIS2_HAVE_CUMESH=1)
    target_include_directories(
        ${target} PRIVATE ${trellis_src_dir}/third_party/cumesh_glue)
    target_link_libraries(${target} PRIVATE aicore_cumesh)
    if(NOT APPLE)
        cloudViewer_install_targets(aicore_cumesh)
        InstallSharedLibrary(TARGET aicore_cumesh)
    endif()
    if(UNIX AND NOT APPLE AND AICore_BUNDLE_CUDA_RUNTIME)
        # Reuse the CUDA bundler with Torch's larger runtime closure. The
        # general installer scanner deliberately excludes these libraries.
        install(CODE "
            execute_process(
                COMMAND bash
                    \"${trellis_src_dir}/../../../../../scripts/platforms/linux/bundle_cuda_runtime.sh\"
                    \"$<TARGET_FILE:aicore_cumesh>\"
                    \"\${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_LIB_DIR}/cuda-runtime\"
                    \"${CUDAToolkit_LIBRARY_DIR}\" torch
                RESULT_VARIABLE _cumesh_bundle_result)
            if(NOT _cumesh_bundle_result EQUAL 0)
                message(FATAL_ERROR \"Failed to bundle CuMesh CUDA runtime dependencies\")
            endif()
        " COMPONENT Runtime)
    endif()
    message(STATUS "AICore: CuMesh GPU chart clustering enabled")
endfunction()
