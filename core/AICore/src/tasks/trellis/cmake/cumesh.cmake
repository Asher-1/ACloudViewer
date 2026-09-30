# Trellis CuMesh GPU chart clustering — task-private build wiring.
#
# Upstream trellis2cpp third_party/cumesh_glue integration (JeffreyXiang/
# CuMesh, MIT, vendored under src/tasks/trellis/third_party/CuMesh): GPU
# normal-cone chart clustering feeds xatlas' faceMaterialData as hard chart
# boundaries for the UV-atlas GLB bake (mesh_export.cpp's UnwrapMode::Auto).
#
# The user-facing switch (AICore_USE_CUMESH) is DECLARED in
# cmake/AICoreOptions.cmake alongside CUMESH_TORCH_DIR; everything below is
# task-owned. Upstream facts (trellis-ggml CMakeLists.txt, "CuMesh GPU mesh
# processing"): TRELLIS2_CUMESH defaults ON and CuMesh "links libtorch
# internally; exposes a torch-free C API through cumesh_glue.h" — the
# libtorch dependency is inherent to the vendored CuMesh core (ATen tensor
# plumbing plus the pybind-layer dtype aliases in hash/hash.cu), not an
# ACloudViewer addition. AICore mirrors the upstream default (ON) but degrades
# gracefully — with a loud warning per missing prerequisite, instead of
# upstream's silent disable or a hard configure failure:
#   CUDA off | libtorch missing | Python headers missing
#     → chart clustering off; mesh_export falls back to simple_unwrap.
# Point CUMESH_TORCH_DIR at the directory containing TorchConfig.cmake
# (e.g. <conda-env>/lib/python3.*/site-packages/torch/share/cmake/Torch).

include_guard(GLOBAL)

function(aicore_trellis_wire_cumesh target trellis_src_dir)
    set(AICORE_CUMESH_ENABLED ${AICore_USE_CUMESH})
    # Torch_DIR = <torch package>/share/cmake/Torch → <torch package> = the
    # site-packages/torch directory itself.
    get_filename_component(_cumesh_torch_pkg "${Torch_DIR}/../../.." ABSOLUTE)
    set(_cumesh_torch_lib_dir "${_cumesh_torch_pkg}/lib")

    # Guard ladder: each step may demote AICORE_CUMESH_ENABLED with a message.
    if(AICORE_CUMESH_ENABLED AND NOT AICore_CUDA_ENABLED)
        message(
            STATUS
                "AICore: CuMesh GPU chart clustering disabled (AICore_USE_CUDA=OFF)"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED
       AND NOT EXISTS ${trellis_src_dir}/third_party/CuMesh/src/cumesh.h)
        message(
            WARNING
                "AICore: missing vendored CuMesh at ${trellis_src_dir}/third_party/CuMesh — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED)
        if(CUMESH_TORCH_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${CUMESH_TORCH_DIR}")
        endif()
        find_package(Torch QUIET)
    endif()
    if(AICORE_CUMESH_ENABLED AND NOT Torch_FOUND)
        message(
            WARNING
                "AICore: CuMesh needs libtorch (find_package(Torch) failed; set CUMESH_TORCH_DIR to <torch-env>/lib/python3.*/site-packages/torch/share/cmake/Torch) — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    if(AICORE_CUMESH_ENABLED
       AND NOT EXISTS "${_cumesh_torch_lib_dir}/libtorch.so")
        message(
            WARNING
                "AICore: CuMesh libtorch.so not found at ${_cumesh_torch_lib_dir} (Torch_DIR=${Torch_DIR}) — building without GPU chart clustering"
        )
        set(AICORE_CUMESH_ENABLED OFF)
    endif()
    # The vendored glue keeps the upstream <torch/extension.h> includes (the
    # upstream glue CMakeLists does the same): it needs the pybind-layer dtype
    # aliases (torch::kUInt32, ...) that do not exist in the ATen header set.
    # That drags in Python.h + libtorch_python, so resolve the Python
    # development bits from the conda/pip env that owns torch: <env>/lib/
    # python3.x/site-packages/torch → up 4 levels to <env>.
    if(AICORE_CUMESH_ENABLED)
        get_filename_component(_cumesh_env_root "${_cumesh_torch_pkg}/../../../.."
                               ABSOLUTE)
        set(_cumesh_py_inc "")
        file(GLOB _cumesh_py_incs "${_cumesh_env_root}/include/python*")
        foreach(_d IN LISTS _cumesh_py_incs)
            if(EXISTS "${_d}/Python.h")
                set(_cumesh_py_inc "${_d}")
                break()
            endif()
        endforeach()
        set(_cumesh_py_lib "")
        file(GLOB _cumesh_py_libs "${_cumesh_env_root}/lib/libpython3*.so")
        if(_cumesh_py_libs)
            list(GET _cumesh_py_libs 0 _cumesh_py_lib)
        endif()
        if(NOT _cumesh_py_inc OR NOT _cumesh_py_lib)
            message(
                WARNING
                    "AICore: CuMesh needs the Python development headers/library of the libtorch env root ${_cumesh_env_root} (include=${_cumesh_py_inc}, lib=${_cumesh_py_lib}) — building without GPU chart clustering"
            )
            set(AICORE_CUMESH_ENABLED OFF)
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
             "${_cumesh_torch_pkg}/include/torch/csrc/api/include/torch/version.h"
             _cumesh_torch_version_h)
        string(REGEX MATCH "#define TORCH_VERSION_MAJOR ([0-9]+)" _cumesh_maj
                      "${_cumesh_torch_version_h}")
        set(_cumesh_torch_maj "${CMAKE_MATCH_1}")
        string(REGEX MATCH "#define TORCH_VERSION_MINOR ([0-9]+)" _cumesh_min
                      "${_cumesh_torch_version_h}")
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
    target_include_directories(
        aicore_cumesh
        PRIVATE ${trellis_src_dir}
                ${trellis_src_dir}/third_party/CuMesh/src
                ${trellis_src_dir}/third_party/cumesh_glue
                ${TORCH_INCLUDE_DIRS}
                ${_cumesh_py_inc})
    set_target_properties(
        aicore_cumesh PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                                 POSITION_INDEPENDENT_CODE ON)
    # ATen CUDA helpers rely on relaxed constexpr / extended lambdas.
    target_compile_options(
        aicore_cumesh PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
        --expt-extended-lambda>
        # Match the libtorch ABI: torch publishes
        # TORCH_CXX_FLAGS = -D_GLIBCXX_USE_CXX11_ABI=0 for pre-2.6 wheels;
        # without it the glue's std::string mangles differ from libc10 and
        # c10::detail::torchCheckFail stays unresolved at dlopen time.
        ${TORCH_CXX_FLAGS})
    # Link the torch core libraries explicitly (conda's TORCH_LIBRARIES can
    # carry a broken "-lFALSE" entry; same workaround as the upstream glue).
    # torch/extension.h pulls the pybind layer, so libtorch_python and
    # libpython are required or the shared lib keeps undefined Py* symbols.
    target_link_directories(aicore_cumesh PRIVATE "${_cumesh_torch_lib_dir}")
    target_link_libraries(
        aicore_cumesh
        PRIVATE "${_cumesh_torch_lib_dir}/libtorch.so"
                "${_cumesh_torch_lib_dir}/libtorch_cpu.so"
                "${_cumesh_torch_lib_dir}/libtorch_cuda.so"
                "${_cumesh_torch_lib_dir}/libc10.so"
                "${_cumesh_torch_lib_dir}/libc10_cuda.so"
                "${_cumesh_torch_lib_dir}/libtorch_python.so"
                "${_cumesh_py_lib}"
                nvrtc)

    target_compile_definitions(${target} PRIVATE TRELLIS2_HAVE_CUMESH=1)
    target_include_directories(
        ${target} PRIVATE ${trellis_src_dir}/third_party/cumesh_glue)
    target_link_libraries(${target} PRIVATE aicore_cumesh)
    message(STATUS "AICore: CuMesh GPU chart clustering enabled")
endfunction()
