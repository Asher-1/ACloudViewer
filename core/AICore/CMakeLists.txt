# AICore: Unified AI inference core library
# Capability modules (not plugin names):
# Capability modules live under src/tasks/ (inference task implementations).
#   src/tasks/depth/      — multi-view monocular depth (DA3-family GGUF)
#   src/tasks/gaussian/   — image-to-3D Gaussian splatting (FreeSplatter-family GGUF)
#   src/tasks/lightglue/  — sparse feature matching (LightGlue GGUF)
#   src/tasks/aliked/     — ALIKED feature extraction
#   src/tasks/deeplsd/    — DeepLSD line extraction
#   src/tasks/facedetect/ — face detection, alignment, embedding
#   src/tasks/rfdetr/     — RF-DETR object detection / segmentation
#   src/tasks/rmbg/       — RMBG-2.0 background removal (foundational matte)
#   src/tasks/ocr/        — PP-OCR text detection/recognition catalog (runtime gates pending)
#   src/tasks/yolo/       — YOLO object detection / metric depth (ultralytics-ggml port)
#   src/tasks/sam3/       — SAM 2 / 2.1 / 3 segmentation (sam3-ggml port)
#   src/tasks/sam3d/      — SAM 3D Objects image-to-Gaussians/mesh (optional)
#   src/tasks/trellis/    — TRELLIS image-to-3D generation
#   src/tasks/gkd/        — general keypoint detection (GKDT, General-Keypoint-Detection-GGML port)
#   src/tasks/map_anything/ — universal feed-forward metric 3D reconstruction
#                           (map-anything-ggml cpp_ggml port: VGGT-Omega / VGGT /
#                            Pi3 / Pi3X / MapAnything / DUSt3R)
#   src/tasks/lingbot/    — LingBot-Map streaming RGB-D reconstruction
#   src/tasks/loma/       — LoMa feature extraction and matching
#   src/tasks/reid/       — appearance embeddings for multi-object tracking
#   src/common/           — shared utilities (data paths, ggml helpers, logging)
#
# Build:   libAICore.so
# Exports: all public headers under include/aicore/
# Depends: 3rdparty_ggml (PRIVATE build dep), CVCoreLib, Qt::Core + Qt::Gui

if(NOT AICore_ENABLED)
    message(STATUS "AICore module is disabled (set AICore_ENABLED=ON to enable)")
    return()
endif()

if(NOT TARGET 3rdparty_ggml)
    message(WARNING "AICore: 3rdparty_ggml not available. Enable AICore_ENABLED in CMake.")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

project(AICore
    DESCRIPTION "Unified AI inference core for ACloudViewer inference tasks"
    LANGUAGES CXX
)

set(AICORE_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AICORE_COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
set(AICORE_TASKS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks)
set(AICORE_DEPTH_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/depth)
set(AICORE_GAUSSIAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/gaussian)
set(AICORE_LIGHTGLUE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/lightglue)
set(AICORE_LOMA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/loma)
set(AICORE_DEEPLSD_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/deeplsd)
set(AICORE_ALIKED_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/aliked)
set(AICORE_FACEDETECT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/facedetect)
set(AICORE_RFDETR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/rfdetr)
set(AICORE_RMBG_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/rmbg)
set(AICORE_OCR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/ocr)
set(AICORE_TABLE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/table)
set(AICORE_YOLO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/yolo)
set(AICORE_SAM3_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/sam3)
set(AICORE_TRELLIS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/trellis)
set(AICORE_GKD_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/gkd)
set(AICORE_MAP_ANYTHING_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/map_anything)
set(AICORE_LINGBOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/lingbot)
set(AICORE_SAM3D_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/sam3d)

if(NOT EXISTS ${AICORE_DEPTH_SRC_DIR}/engine.cpp)
    message(WARNING "AICore: depth module not found at ${AICORE_DEPTH_SRC_DIR}")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

if(NOT EXISTS ${AICORE_GAUSSIAN_SRC_DIR}/model.cpp)
    message(WARNING "AICore: gaussian module not found at ${AICORE_GAUSSIAN_SRC_DIR}")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

if(NOT EXISTS ${AICORE_LIGHTGLUE_SRC_DIR}/matcher.cpp)
    message(WARNING "AICore: lightglue module not found at ${AICORE_LIGHTGLUE_SRC_DIR}")
    set(AICore_ENABLED OFF CACHE BOOL "" FORCE)
    return()
endif()

file(GLOB AICORE_COMMON_SOURCES CONFIGURE_DEPENDS ${AICORE_COMMON_SRC_DIR}/*.cpp)
file(GLOB AICORE_COMMON_HEADERS CONFIGURE_DEPENDS ${AICORE_COMMON_SRC_DIR}/*.hpp)
file(GLOB DEPTH_SOURCES CONFIGURE_DEPENDS ${AICORE_DEPTH_SRC_DIR}/*.cpp)
file(GLOB DEPTH_HEADERS CONFIGURE_DEPENDS ${AICORE_DEPTH_SRC_DIR}/*.hpp)
file(GLOB GAUSSIAN_SOURCES CONFIGURE_DEPENDS ${AICORE_GAUSSIAN_SRC_DIR}/*.cpp)
file(GLOB GAUSSIAN_HEADERS CONFIGURE_DEPENDS ${AICORE_GAUSSIAN_SRC_DIR}/*.hpp ${AICORE_GAUSSIAN_SRC_DIR}/*.h)
file(GLOB LIGHTGLUE_SOURCES CONFIGURE_DEPENDS ${AICORE_LIGHTGLUE_SRC_DIR}/*.cpp)
file(GLOB LIGHTGLUE_HEADERS CONFIGURE_DEPENDS ${AICORE_LIGHTGLUE_SRC_DIR}/*.hpp)
file(GLOB AICORE_LOMA_SOURCES CONFIGURE_DEPENDS ${AICORE_LOMA_SRC_DIR}/*.cpp)
file(GLOB AICORE_DEEPLSD_SOURCES CONFIGURE_DEPENDS ${AICORE_DEEPLSD_SRC_DIR}/*.cpp)
file(GLOB AICORE_ALIKED_SOURCES CONFIGURE_DEPENDS ${AICORE_ALIKED_SRC_DIR}/*.cpp)
file(GLOB AICORE_FACEDETECT_SOURCES CONFIGURE_DEPENDS ${AICORE_FACEDETECT_SRC_DIR}/*.cpp)
file(GLOB AICORE_RFDETR_SOURCES CONFIGURE_DEPENDS ${AICORE_RFDETR_SRC_DIR}/*.cpp)
file(GLOB AICORE_RMBG_SOURCES CONFIGURE_DEPENDS ${AICORE_RMBG_SRC_DIR}/*.cpp)
file(GLOB AICORE_OCR_SOURCES CONFIGURE_DEPENDS ${AICORE_OCR_SRC_DIR}/*.cpp)
file(GLOB AICORE_TABLE_SOURCES CONFIGURE_DEPENDS ${AICORE_TABLE_SRC_DIR}/*.cpp)
file(GLOB AICORE_YOLO_SOURCES CONFIGURE_DEPENDS ${AICORE_YOLO_SRC_DIR}/*.cpp)
set(AICORE_REID_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/reid)
file(GLOB AICORE_REID_SOURCES CONFIGURE_DEPENDS ${AICORE_REID_SRC_DIR}/*.cpp)
file(GLOB AICORE_SAM3_SOURCES CONFIGURE_DEPENDS ${AICORE_SAM3_SRC_DIR}/*.cpp)
file(GLOB AICORE_GKD_SOURCES CONFIGURE_DEPENDS ${AICORE_GKD_SRC_DIR}/*.cpp)
file(GLOB AICORE_MAP_ANYTHING_SOURCES CONFIGURE_DEPENDS
    ${AICORE_MAP_ANYTHING_SRC_DIR}/*.cpp)
file(GLOB AICORE_LINGBOT_SOURCES CONFIGURE_DEPENDS ${AICORE_LINGBOT_SRC_DIR}/*.cpp)
file(GLOB AICORE_SAM3D_SOURCES CONFIGURE_DEPENDS ${AICORE_SAM3D_SRC_DIR}/*.cpp)
file(GLOB AICORE_SAM3D_HEADERS CONFIGURE_DEPENDS
    ${AICORE_SAM3D_SRC_DIR}/*.hpp ${AICORE_SAM3D_SRC_DIR}/*.h)
file(GLOB AICORE_TRELLIS_SOURCES CONFIGURE_DEPENDS
    ${AICORE_TRELLIS_SRC_DIR}/*.cpp
    ${AICORE_TRELLIS_SRC_DIR}/third_party/xatlas/*.cpp
    ${AICORE_TRELLIS_SRC_DIR}/third_party/o-voxel-fdg/*.cpp)
file(GLOB AICORE_API_HEADERS CONFIGURE_DEPENDS ${AICORE_INC_DIR}/aicore/*.h)

if(NOT EXISTS ${AICORE_DEEPLSD_SRC_DIR}/deeplsd_extractor.cpp)
    message(FATAL_ERROR "AICore: missing DeepLSD module at ${AICORE_DEEPLSD_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_ALIKED_SRC_DIR}/aliked.cpp)
    message(FATAL_ERROR "AICore: missing ALIKED module at ${AICORE_ALIKED_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_FACEDETECT_SRC_DIR}/model.cpp)
    message(FATAL_ERROR "AICore: missing FaceDetect module at ${AICORE_FACEDETECT_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_RFDETR_SRC_DIR}/rfdetr_model.cpp)
    message(FATAL_ERROR "AICore: missing RF-DETR module at ${AICORE_RFDETR_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_RMBG_SRC_DIR}/rmbg_graph.cpp)
    message(FATAL_ERROR "AICore: missing RMBG module at ${AICORE_RMBG_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_YOLO_SRC_DIR}/yolo_graph.cpp)
    message(FATAL_ERROR "AICore: missing YOLO module at ${AICORE_YOLO_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_GKD_SRC_DIR}/gkd_graph.cpp)
    message(FATAL_ERROR "AICore: missing GKD module at ${AICORE_GKD_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_MAP_ANYTHING_SRC_DIR}/map_anything_graph.cpp)
    message(FATAL_ERROR "AICore: missing MapAnything module at ${AICORE_MAP_ANYTHING_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_SAM3_SRC_DIR}/sam3.cpp)
    message(FATAL_ERROR "AICore: missing SAM3 module at ${AICORE_SAM3_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_LINGBOT_SRC_DIR}/lingbot_model.cpp)
    message(FATAL_ERROR "AICore: missing LingBot-Map module at ${AICORE_LINGBOT_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_TRELLIS_SRC_DIR}/trellis2.cpp)
    message(FATAL_ERROR "AICore: missing TRELLIS module at ${AICORE_TRELLIS_SRC_DIR}")
endif()
if(NOT EXISTS ${AICORE_TRELLIS_SRC_DIR}/third_party/xatlas/xatlas.cpp)
    message(FATAL_ERROR "AICore: missing TRELLIS third_party (xatlas) at ${AICORE_TRELLIS_SRC_DIR}/third_party")
endif()
if(NOT EXISTS ${AICORE_SAM3D_SRC_DIR}/session.cpp)
    message(FATAL_ERROR "AICore: missing SAM3D module at ${AICORE_SAM3D_SRC_DIR}")
endif()

option(MAP_ANYTHING_ENABLE_DUMP
       "Compile qMapAnything parity stage dumps (debug-only, increases runtime I/O)"
       OFF)

add_library(${PROJECT_NAME} SHARED
    ${AICORE_COMMON_SOURCES} ${AICORE_COMMON_HEADERS}
    ${DEPTH_SOURCES} ${DEPTH_HEADERS}
    ${GAUSSIAN_SOURCES} ${GAUSSIAN_HEADERS}
    ${LIGHTGLUE_SOURCES} ${LIGHTGLUE_HEADERS}
    ${AICORE_LOMA_SOURCES}
    ${AICORE_DEEPLSD_SOURCES}
    ${AICORE_ALIKED_SOURCES}
    ${AICORE_FACEDETECT_SOURCES}
    ${AICORE_RFDETR_SOURCES}
    ${AICORE_RMBG_SOURCES}
    ${AICORE_OCR_SOURCES}
    ${AICORE_TABLE_SOURCES}
    ${AICORE_YOLO_SOURCES}
    ${AICORE_REID_SOURCES}
    ${AICORE_SAM3_SOURCES}
    ${AICORE_TRELLIS_SOURCES}
    ${AICORE_GKD_SOURCES}
    ${AICORE_MAP_ANYTHING_SOURCES}
    ${AICORE_LINGBOT_SOURCES}
    ${AICORE_SAM3D_SOURCES} ${AICORE_SAM3D_HEADERS}
    ${AICORE_API_HEADERS}
)

# LingBot host-side floating-point parity: the reference implementation
# (upstream cpp_ggml) compiles WITHOUT -march (x86-64 baseline, no FMA),
# while CMakeFindSSE adds -march=x86-64-v3 (AVX2+FMA) to this tree. With
# contraction enabled GCC fuses `a*b+c` into vfmadd inside the DINO
# positional-table interpolation and the Pillow resample kernels, and the
# resulting 1-ULP drift propagates through all 24 transformer blocks into
# visibly different depth/pose output. Disabling contraction on the
# lingbot TUs restores the reference bit-exact contract; ggml kernels are
# unaffected (they are compiled in ext_ggml without -march).
set_source_files_properties(
    ${AICORE_LINGBOT_SRC_DIR}/lingbot_model.cpp
    ${AICORE_LINGBOT_SRC_DIR}/lingbot_capi.cpp
    ${AICORE_LINGBOT_SRC_DIR}/lingbot_skyseg.cpp
    ${AICORE_LINGBOT_SRC_DIR}/lingbot_gguf_loader.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# MapAnything host-side floating-point parity: the reference implementation
# (upstream map-anything-ggml cpp_ggml) compiles WITHOUT -march (x86-64
# baseline, no FMA), while CMakeFindSSE adds -march=x86-64-v3 (AVX2+FMA) to
# this tree. With contraction enabled GCC fuses `a*b+c` into vfmadd inside
# the graph-builder's host-side float math (RoPE cos/sin tables, mean/std
# staging, DPT resample weight math), drifting 1-ULP through the 24
# transformer blocks and visibly changing depth/pose output versus the
# upstream parity feeds. Disabling contraction on the map_anything TUs
# restores the reference bit-exact contract; ggml kernels are unaffected
# (they are compiled in ext_ggml without -march).
set_source_files_properties(
    ${AICORE_MAP_ANYTHING_SRC_DIR}/map_anything_graph.cpp
    ${AICORE_MAP_ANYTHING_SRC_DIR}/map_anything_capi.cpp
    ${AICORE_MAP_ANYTHING_SRC_DIR}/map_anything_gguf_loader.cpp
    ${AICORE_MAP_ANYTHING_SRC_DIR}/map_anything_image_io.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# OCR det/rec geometry parity: ocr_rec_crop.cpp and ocr_db_postprocess.cpp
# port cv2.minAreaRect / contourArea / arcLength / warpPerspective(
# INTER_CUBIC) from OpenCV's x86-64 baseline build and the Clipper offset
# pyclipper ships, neither of which fuses a*b+c. A fused rotatingCalipers /
# perspective-matrix product moves box corners by 1 ULP (628 of the 720 rec
# crop golden boxes with -ffp-contract=fast) and flips cubic table cells, so
# these TUs keep separate roundings (tests/ocr/test_ocr_rec_crop and
# test_ocr_db_postprocess compare bytes). MSVC does not contract without
# /fp:contract.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        ${AICORE_OCR_SRC_DIR}/ocr_rec_crop.cpp
        ${AICORE_OCR_SRC_DIR}/ocr_db_postprocess.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

if(MAP_ANYTHING_ENABLE_DUMP)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MAP_ANYTHING_ENABLE_DUMP)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "core"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # Always beside ACloudViewer executable (bin/), never bin/plugins/ where Qt scans *.so.
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# AICore and its private ggml runtime files are always deployed together. Use
# only a loader-relative runtime path so build-machine directories cannot leak
# into installers or wheels.
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN")
endif()

if(UNIX AND NOT APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/aicore.exports.map")
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
        LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/aicore.exports.map)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${AICORE_INC_DIR}>
        $<INSTALL_INTERFACE:include/AICore>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${AICORE_COMMON_SRC_DIR}
        ${AICORE_TASKS_SRC_DIR}
        ${AICORE_DEPTH_SRC_DIR}
        ${AICORE_GAUSSIAN_SRC_DIR}
        ${AICORE_LIGHTGLUE_SRC_DIR}
        ${AICORE_LOMA_SRC_DIR}
        ${AICORE_DEEPLSD_SRC_DIR}
        ${AICORE_ALIKED_SRC_DIR}
        ${AICORE_ALIKED_SRC_DIR}/cuda
        ${AICORE_ALIKED_SRC_DIR}/vulkan
        ${AICORE_ALIKED_SRC_DIR}/include
        ${AICORE_FACEDETECT_SRC_DIR}
        ${AICORE_RFDETR_SRC_DIR}
        ${AICORE_RMBG_SRC_DIR}
        ${AICORE_OCR_SRC_DIR}
        ${AICORE_TABLE_SRC_DIR}
        ${AICORE_YOLO_SRC_DIR}
        ${AICORE_SAM3_SRC_DIR}
        ${AICORE_TRELLIS_SRC_DIR}
        ${AICORE_GKD_SRC_DIR}
        ${AICORE_MAP_ANYTHING_SRC_DIR}
        ${AICORE_LINGBOT_SRC_DIR}
        ${AICORE_TRELLIS_SRC_DIR}/third_party/xatlas
        ${AICORE_TRELLIS_SRC_DIR}/third_party/o-voxel-fdg
        ${AICORE_SAM3D_SRC_DIR}
        ${EIGEN_INCLUDE_DIRS}
)

add_dependencies(${PROJECT_NAME} 3rdparty_ggml)
add_dependencies(${PROJECT_NAME} 3rdparty_meshoptimizer)
# Build: ggml headers/targets stay PRIVATE (see AICore/README.md).
target_link_libraries(${PROJECT_NAME} PRIVATE 3rdparty_ggml 3rdparty_meshoptimizer 3rdparty_jsoncpp)
if(MSVC)
    # UTF-8 source & execution character sets: AICore sources carry Unicode
    # literals (em-dashes in UI strings, the multilingual-bridge docs) — match
    # the GCC/Clang default parsing instead of the host code page (C4819).
    target_compile_options(AICore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    if(MSVC)
        # MSVC's default /openmp is OpenMP 2.0, which does NOT support
        # '#pragma omp simd' or the 'collapse' clause (C7660/C4849). The
        # built-in /openmp:experimental enables them (no optional LLVM
        # component required), so the ALIKED SIMD/collapse kernels keep their
        # full performance on Windows instead of being disabled.
        #
        # ggml.cmake links OpenMP::OpenMP_CXX INTERFACE on 3rdparty_ggml,
        # propagating /openmp. Strip it here first — /openmp:experimental
        # subsumes standard /openmp, and having both triggers MSVC D9025.
        get_target_property(_ggml_omp 3rdparty_ggml INTERFACE_LINK_LIBRARIES)
        if(_ggml_omp MATCHES "OpenMP::OpenMP_CXX")
            list(REMOVE_ITEM _ggml_omp "OpenMP::OpenMP_CXX")
            set_target_properties(3rdparty_ggml PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_ggml_omp}")
        endif()
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
# Consumers link only AICore; propagate ggml core .so paths (no include dirs) so
# executables resolve libggml alongside libAICore. Backend MODULE libs remain
# dlopen'd from libAICore's directory at runtime.
if(GGML_CORE_LINK_LIBS)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${GGML_CORE_LINK_LIBS})
endif()
# GGML_BACKEND_DL: direct AICore C API executables need allow-shlib-undefined on
# Linux so libAICore.so can leave backend symbols unresolved until dlopen.
# Do NOT propagate via AICore INTERFACE — Colmap/VTK link DSOs that need crc32
# from libz while static libz.a is hidden via --exclude-libs,libz.a.
if(GGML_DYNAMIC_BACKENDS AND UNIX AND NOT APPLE AND NOT MSVC)
    add_library(AICore_capi_link INTERFACE)
    target_link_options(AICore_capi_link INTERFACE "-Wl,--allow-shlib-undefined")
endif()
if(DEFINED GGML_INCLUDE_DIRS)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GGML_INCLUDE_DIRS})
endif()

# Dynamic backend mode: GPU backends are separate modules loaded at runtime.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AICoreCompileDefinitions.cmake)
if(GGML_DYNAMIC_BACKENDS)
    message(STATUS "AICore: using dynamic ggml backends (AICORE_BACKEND_DL)")
endif()
aicore_apply_compile_definitions(${PROJECT_NAME})

if(AICore_CUDA_ENABLED)
    target_sources(${PROJECT_NAME} PRIVATE ${AICORE_ALIKED_SRC_DIR}/cuda/aliked_cuda.cu)
    # PyTorch-2.5.1-compatible CUDA normal RNG for the sam3d samplers
    # (same compile-definition contract as the source sam3dggml build:
    # SAM3D_USE_CUDA selects the device kernel over the Philox CPU path).
    target_sources(${PROJECT_NAME} PRIVATE ${AICORE_SAM3D_SRC_DIR}/pytorch_cuda_rng.cu)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SAM3D_USE_CUDA)
    # Static-link the CUDA runtime (cudart) on UNIX so libAICore.so has no
    # DT_NEEDED on libcudart.so.X and loads on machines without a CUDA toolkit.
    # This is the same strategy as the qSIBR plugin (see qSIBR/CMakeLists.txt).
    # On Windows, NVIDIA does not ship a static cudart, so fall through to the
    # dynamic CUDA::cudart target (the ALIKED CUDA path is guarded by a runtime
    # ggml_backend_is_cuda() check, so the dlopen-proof cudart dependency is
    # acceptable during developer bring-up).
    if(TARGET CUDA::cudart_static AND UNIX)
        target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
    elseif(TARGET CUDA::cudart)
        target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
    endif()
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# ── SAM3D official native PBR assembly ──────────────────────
# Upstream sam-3d-objects-ggml assemble_official_pbr_cuda chain (CUDA-only):
# VTK quadric simplify (0.95) → 1000-view nvdiffrast visibility → CPU min-cut
# filter → pymeshfix TMesh repair (GPL) → xatlas UV → angle-weighted normals
# → 100-view mip-splatting Gaussian render → nvdiffrast 2500-step Adam/TV
# texture bake → Telea inpaint → Y-up GLB.
# License isolation mirrors upstream: nvdiffrast (NVIDIA, non-commercial) and
# pymeshfix TMesh (GPL-3.0-or-commercial) are never fetched or linked without
# an explicit acknowledgment option; the mip-splatting rasterizer (Inria
# non-commercial) and the MIT xatlas snapshot are vendored but also compiled
# only under this gate. The default AICore build is untouched.
# The task-owned wiring (vendored third-party builds, missing-module
# diagnostics, pipeline source list) lives in src/tasks/sam3d/cmake/pbr.cmake
# (mirroring this module's core/AICore/cmake/ convention); the switch and its
# license acknowledgments are declared in cmake/AICoreOptions.cmake (before
# find_dependencies, so the bundled OpenCV build wires its photo module from
# the same switch). The CUDA-backend requirement is the only cross-module
# linkage kept here.
if(AICore_USE_SAM3D_PBR)
    if(NOT AICore_CUDA_ENABLED)
        message(
            FATAL_ERROR
                "AICore: AICore_USE_SAM3D_PBR requires AICore_USE_CUDA=ON: the official PBR chain is CUDA-only (same as upstream SAM3D_GGML_NATIVE_PBR). The auto-enable only fires with CUDA already ON, so this value was set explicitly — set AICore_USE_SAM3D_PBR=OFF to keep a CUDA-less build."
        )
    endif()
    include(${AICORE_SAM3D_SRC_DIR}/cmake/pbr.cmake)
    aicore_sam3d_wire_pbr(${PROJECT_NAME} ${AICORE_SAM3D_SRC_DIR})
endif()

if(AICore_VULKAN_ENABLED)
    target_sources(${PROJECT_NAME} PRIVATE
        ${AICORE_ALIKED_SRC_DIR}/vulkan/vulkan_aliked_dispatch.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${AICORE_ALIKED_SRC_DIR}/vulkan)
endif()

# ── Trellis CuMesh GPU chart clustering ──────────────────────
# Upstream trellis2cpp third_party/cumesh_glue integration (JeffreyXiang/
# CuMesh, MIT, vendored under src/tasks/trellis/third_party/CuMesh): GPU
# normal-cone chart clustering feeds xatlas' faceMaterialData as hard chart
# boundaries for the UV-atlas GLB bake (mesh_export.cpp's UnwrapMode::Auto).
# Upstream facts (trellis-ggml CMakeLists.txt, "CuMesh GPU mesh processing"):
# TRELLIS2_CUMESH defaults ON and CuMesh "links libtorch internally; exposes
# a torch-free C API through cumesh_glue.h" — the libtorch dependency is
# inherent to the vendored CuMesh core (ATen tensor plumbing plus the
# pybind-layer dtype aliases in hash/hash.cu), not an ACloudViewer addition.
# AICore mirrors the upstream default (ON) but degrades gracefully — with a
# loud warning per missing prerequisite, instead of upstream's silent
# disable or a hard configure failure:
#   CUDA off | libtorch missing | Python headers missing
#     → chart clustering off; mesh_export falls back to simple_unwrap.
# Point CUMESH_TORCH_DIR at the directory containing TorchConfig.cmake
# (e.g. <conda-env>/lib/python3.*/site-packages/torch/share/cmake/Torch).
# Task-owned wiring (graceful guard ladder, vendored SHARED build against
# libtorch/libpython, hash.cu torch version gate) lives in
# src/tasks/trellis/cmake/cumesh.cmake; the AICore_USE_CUMESH switch and
# CUMESH_TORCH_DIR are declared in cmake/AICoreOptions.cmake.
include(${AICORE_TRELLIS_SRC_DIR}/cmake/cumesh.cmake)
aicore_trellis_wire_cumesh(${PROJECT_NAME} ${AICORE_TRELLIS_SRC_DIR})
# Vulkan ALIKED custom compute: 3rdparty/ggml/patches/aliked/0001-vulkan-aliked-custom-compute.patch
# (AICORE_VULKAN_ALIKED when AICore_VULKAN_ENABLED — see AICoreCompileDefinitions.cmake)

# Static mode only: ggml-cuda static archives are embedded into libAICore.so
# and pull in CUDA driver API (libcuda.so) as a DT_NEEDED dep.  In Docker/CI
# environments the real driver isn't installed; the CUDA toolkit ships a stub at
# <toolkit>/lib64/stubs/.  Expose that path so downstream executables (colmap_exe,
# ACloudViewer) can resolve libcuda.so at link time.
if(NOT GGML_DYNAMIC_BACKENDS AND AICore_CUDA_ENABLED AND CUDAToolkit_FOUND)
    get_target_property(_cuda_driver_loc CUDA::cuda_driver IMPORTED_LOCATION)
    if(_cuda_driver_loc)
        get_filename_component(_cuda_driver_dir "${_cuda_driver_loc}" DIRECTORY)
        target_link_directories(${PROJECT_NAME} PUBLIC "${_cuda_driver_dir}")
    elseif(CUDAToolkit_LIBRARY_DIR)
        if(EXISTS "${CUDAToolkit_LIBRARY_DIR}/stubs")
            target_link_directories(${PROJECT_NAME} PUBLIC "${CUDAToolkit_LIBRARY_DIR}/stubs")
        endif()
    endif()
endif()

if(TARGET CVCoreLib)
    target_link_libraries(${PROJECT_NAME} PRIVATE CVCoreLib)
    target_compile_definitions(${PROJECT_NAME} PRIVATE AICore_HAS_CVLOG)
else()
    message(FATAL_ERROR "AICore requires CVCoreLib (CVTools for cross-platform paths)")
endif()
# xatlas (trellis GLB export) uses std::thread for parallel charting.
find_package(Threads REQUIRED)
if(Threads_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
endif()

if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(${PROJECT_NAME} PUBLIC Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Core Qt5::Gui)
endif()

target_compile_definitions(${PROJECT_NAME}
    PRIVATE AICore_BUILD AICore_SHARED
    PRIVATE TRELLIS2_HAVE_RMBG=1
    INTERFACE AICore_SHARED
    PUBLIC AICore_ENABLED
)
# SAM3 C++ symbols are implementation details.  The diagnostics that need
# them are built into a test-only target (see tests/CMakeLists.txt), keeping
# libAICore's ABI limited to the documented C API.
target_compile_definitions(${PROJECT_NAME} PRIVATE SAM3_BUILD)
if(MSVC)
    # M_PI and other math constants (pose.cpp, tests); same as CloudViewerSetGlobalProperties.cmake
    target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

include(CheckCXXCompilerFlag)
if(ENABLE_AVX512)
    if(NOT MSVC)
        check_cxx_compiler_flag("-mavx512f" AICORE_HAS_AVX512F)
        if(AICORE_HAS_AVX512F)
            set_source_files_properties(${AICORE_DEPTH_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma")
        endif()
    endif()
else()
    if(NOT MSVC)
        check_cxx_compiler_flag("-mavx2" AICORE_HAS_AVX2)
        check_cxx_compiler_flag("-mfma" AICORE_HAS_FMA)
        if(AICORE_HAS_AVX2 AND AICORE_HAS_FMA)
            set_source_files_properties(${AICORE_DEPTH_SRC_DIR}/winograd.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
            set_source_files_properties(
                ${AICORE_COMMON_SRC_DIR}/winograd.cpp
                ${AICORE_COMMON_SRC_DIR}/blocked_conv.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        endif()
    endif()
endif()

set(AICore_FOUND TRUE CACHE BOOL "AICore module found" FORCE)
set(AICore_INCLUDE_DIRS
    "${AICORE_INC_DIR}"
    CACHE PATH "AICore include dirs" FORCE)
set(AICore_LIBRARIES AICore CACHE STRING "AICore library target" FORCE)

if(NOT APPLE)
    cloudViewer_install_targets(${PROJECT_NAME})
    InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif()
install(DIRECTORY ${AICORE_INC_DIR}/aicore
        DESTINATION include/AICore
        FILES_MATCHING PATTERN "*.h")

# Ship ggml shared libs and dynamic backend modules alongside libAICore.
# file(GLOB) runs at configure time before ExternalProject builds, so we use a
# POST_BUILD script that globs at build time when the dylibs exist.
#
# IMPORTANT: In GGML_BACKEND_DL mode, ggml installs dynamic backend modules as
# .so files (MODULE libraries) into the bin/ subdirectory of the install prefix,
# even on macOS.  ggml's runtime loader (ggml-backend-reg.cpp) searches for .so
# extensions on all UNIX platforms.  We must copy both:
#   - .dylib from lib/  (shared core: ggml, ggml-base)
#   - .so    from bin/  (loadable modules for runtime: ggml-metal, ggml-vulkan, etc.)
if(GGML_DYNAMIC_BACKENDS AND GGML_RUNTIME_LIB_DIR AND GGML_BACKEND_DIR)
    string(REPLACE ";" "|" _aicore_gpu_backend_files_pipe
        "${GGML_GPU_BACKEND_MODULE_FILES}")
    set(_aicore_copy_ggml_script
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyGgmlBackends.cmake")
    set(_aicore_backend_copy_args
        -DCOPY_BACKEND_MODULES=ON
        -DGGML_BACKEND_DIR="${GGML_BACKEND_DIR}"
        -DDEST_DIR="$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        -DLIB_SUFFIX="${GGML_MODULE_SUFFIX}"
        -DGPU_FILES_PIPE="${_aicore_gpu_backend_files_pipe}")
    if(GGML_CPU_ALL_VARIANTS)
        list(APPEND _aicore_backend_copy_args -DCPU_ALL_VARIANTS=ON)
    endif()
    list(APPEND _aicore_backend_copy_args -P "${_aicore_copy_ggml_script}")
    # Copy shared libraries (.dylib / .so depending on platform)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DCOPY_CORE_SHARED=ON
            -DGGML_BACKEND_DIR="${GGML_RUNTIME_LIB_DIR}"
            -DDEST_DIR="$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
            -DLIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
            -P "${_aicore_copy_ggml_script}"
        COMMENT "Copying ggml shared libraries to libAICore directory"
    )
    # Copy loadable backend modules (.so on all UNIX platforms, .dll on Windows)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} ${_aicore_backend_copy_args}
        COMMENT "Copying ggml dynamic backend modules to libAICore directory"
    )
    # aicore_tests/ sits beside libAICore.so; ggml's backend loader may probe the
    # executable directory first when scripts set LD_LIBRARY_PATH=.../aicore_tests.
    if(AICore_BUILD_TESTS)
        set(_aicore_test_backend_copy_args
            -DCOPY_BACKEND_MODULES=ON
            -DGGML_BACKEND_DIR="${GGML_BACKEND_DIR}"
            -DDEST_DIR="${CMAKE_BINARY_DIR}/bin/aicore_tests"
            -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
            -DLIB_SUFFIX="${GGML_MODULE_SUFFIX}"
            -DGPU_FILES_PIPE="${_aicore_gpu_backend_files_pipe}")
        if(GGML_CPU_ALL_VARIANTS)
            list(APPEND _aicore_test_backend_copy_args -DCPU_ALL_VARIANTS=ON)
        endif()
        list(APPEND _aicore_test_backend_copy_args -P "${_aicore_copy_ggml_script}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} ${_aicore_test_backend_copy_args}
            COMMENT "Copying ggml dynamic backend modules to aicore_tests directory"
        )
    endif()
    # `cmake --install` only ships libAICore.so by default; backend MODULE libs
    # (libggml-cpu.so, libggml-cuda.so, …) must be installed alongside it or
    # ggml_backend_load_all_from_path() finds dev_count=0 at runtime.
    if(POLICY CMP0087)
        cmake_policy(SET CMP0087 NEW)
    endif()
    if(WIN32)
        set(_aicore_ggml_install_subdir "${ACloudViewer_DEST_FOLDER}")
    else()
        set(_aicore_ggml_install_subdir "${CloudViewer_INSTALL_LIB_DIR}")
    endif()
    if(GGML_CPU_ALL_VARIANTS)
        set(_aicore_install_cpu_all_variants ON)
    else()
        set(_aicore_install_cpu_all_variants OFF)
    endif()
    set(_AICORE_GPU_PIPE "${_aicore_gpu_backend_files_pipe}")
    set(_AICORE_CPU_ALL_VARIANTS "${_aicore_install_cpu_all_variants}")
    set(_AICORE_COPY_GGML_SCRIPT "${_aicore_copy_ggml_script}")
    set(_AICORE_GGML_INSTALL_SUBDIR "${_aicore_ggml_install_subdir}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/InstallGgmlBackends.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/InstallGgmlBackends.cmake"
        @ONLY)
    install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallGgmlBackends.cmake")
endif()

set(_aicore_enabled_task_names
    aliked deeplsd depth facedetect gaussian gkd lightglue lingbot loma ocr
    reid rfdetr rmbg sam3 sam3d table trellis yolo)
list(JOIN _aicore_enabled_task_names "," _aicore_enabled_tasks_csv)

if(AICore_BUILD_TESTS)
    if(AICore_BUILD_WHITEBOX_TESTS)
        set(AICORE_TEST_SOURCES
            ${AICORE_COMMON_SOURCES}
            ${DEPTH_SOURCES}
            ${GAUSSIAN_SOURCES}
            ${LIGHTGLUE_SOURCES}
            ${AICORE_LOMA_SOURCES}
            ${AICORE_FACEDETECT_SOURCES}
            ${AICORE_RFDETR_SOURCES}
            ${AICORE_RMBG_SOURCES}
            ${AICORE_OCR_SOURCES}
            ${AICORE_YOLO_SOURCES}
            ${AICORE_SAM3_SOURCES}
            ${AICORE_TRELLIS_SOURCES}
            ${AICORE_GKD_SOURCES}
            ${AICORE_LINGBOT_SOURCES}
            ${AICORE_REID_SOURCES}
            ${AICORE_SAM3D_SOURCES}
        )
        list(FILTER AICORE_TEST_SOURCES EXCLUDE REGEX
             ".*/(capi|image_depth)\\.cpp$")

        add_library(AICore_test STATIC
            ${AICORE_TEST_SOURCES}
            ${AICORE_COMMON_HEADERS} ${DEPTH_HEADERS} ${GAUSSIAN_HEADERS}
            ${LIGHTGLUE_HEADERS}
        )
        set_target_properties(AICore_test PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            FOLDER "core/tests"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
        target_include_directories(AICore_test PUBLIC
            $<BUILD_INTERFACE:${AICORE_INC_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${AICORE_COMMON_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_TASKS_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_DEPTH_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_GAUSSIAN_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_LIGHTGLUE_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_LOMA_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_FACEDETECT_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_RFDETR_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_RMBG_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_YOLO_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_SAM3_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_GKD_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_LINGBOT_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_TRELLIS_SRC_DIR}>
            $<BUILD_INTERFACE:${AICORE_TRELLIS_SRC_DIR}/third_party/xatlas>
            $<BUILD_INTERFACE:${AICORE_TRELLIS_SRC_DIR}/third_party/o-voxel-fdg>
            $<BUILD_INTERFACE:${AICORE_SAM3D_SRC_DIR}>
        )
        add_dependencies(AICore_test 3rdparty_ggml 3rdparty_meshoptimizer)
        target_link_libraries(AICore_test PUBLIC 3rdparty_ggml 3rdparty_meshoptimizer CVCoreLib 3rdparty_jsoncpp)
        if(DEFINED GGML_INCLUDE_DIRS)
            target_include_directories(AICore_test PUBLIC ${GGML_INCLUDE_DIRS})
        endif()
        if(TARGET Qt::Core AND TARGET Qt::Gui)
            target_link_libraries(AICore_test PUBLIC Qt::Core Qt::Gui)
        elseif(Qt5_FOUND)
            target_link_libraries(AICore_test PUBLIC Qt5::Core Qt5::Gui)
        endif()
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            if(MSVC)
                # Same as AICore: strip inherited /openmp from 3rdparty_ggml
                # to avoid D9025 before adding /openmp:experimental.
                get_target_property(_ggml_omp_t 3rdparty_ggml INTERFACE_LINK_LIBRARIES)
                if(_ggml_omp_t MATCHES "OpenMP::OpenMP_CXX")
                    list(REMOVE_ITEM _ggml_omp_t "OpenMP::OpenMP_CXX")
                    set_target_properties(3rdparty_ggml PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${_ggml_omp_t}")
                endif()
                target_compile_options(AICore_test PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
            else()
                target_link_libraries(AICore_test PUBLIC OpenMP::OpenMP_CXX)
            endif()
        endif()
        target_compile_definitions(AICore_test PUBLIC AICore_ENABLED)
        if(AICore_CUDA_ENABLED)
            # Same compile-definition contract as libAICore: the white-box
            # sam3d probes (stage bisect against upstream condition dumps)
            # need the PyTorch-compatible CUDA normal RNG so a fixed-conditions
            # replay generates the same seed noise as the production session.
            target_sources(AICore_test PRIVATE ${AICORE_SAM3D_SRC_DIR}/pytorch_cuda_rng.cu)
            target_compile_definitions(AICore_test PUBLIC SAM3D_USE_CUDA)
            if(TARGET CUDA::cudart)
                target_link_libraries(AICore_test PUBLIC CUDA::cudart)
            endif()
        endif()
        aicore_apply_compile_definitions(AICore_test)
        if(MSVC)
            target_compile_definitions(AICore_test PRIVATE _USE_MATH_DEFINES)
        endif()
    endif()

    add_subdirectory(tests)
    message(STATUS "AICore: contract tests enabled")
endif()

message(STATUS "AICore (SHARED): tasks=${_aicore_enabled_tasks_csv}; ggml linked once")

add_executable(aicore_gguf_quantize tools/aicore_gguf_quantize.cpp)
target_link_libraries(aicore_gguf_quantize PRIVATE AICore)

# End-to-end OCR benchmark (det+rec per image, p50/p95); the official
# PaddleOCR counterpart lives in scripts/bench_paddle_official.py.
add_executable(bench_ocr tools/bench_ocr.cpp)
target_link_libraries(bench_ocr PRIVATE AICore)

# Headless table reconstruction driver (OCR + table, one JSON line per
# image); scripts/run_table_gt_matrix.py scores its HTML against GT.
add_executable(bench_table tools/bench_table.cpp)
target_link_libraries(bench_table PRIVATE AICore)

add_executable(test_ocr_runtime tests/test_ocr_runtime.cpp
    src/tasks/ocr/ocr_runtime.cpp
    # Winograd conv custom op (routed by the CPU-session graph build).
    src/common/winograd.cpp
    # Blocked-layout island ops (routed by the CPU-session graph build).
    src/common/blocked_conv.cpp
    # Backend discovery helpers referenced by the probe (dynamic-loader
    # backend registration lives in the env bridge).
    src/common/ggml_env_bridge.cpp)
target_include_directories(test_ocr_runtime PRIVATE
    ${AICORE_INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/ocr
    ${JSONCPP_INCLUDE_DIRS})
# The probe is deliberately independent of the AICore shared library: the
# ocr interpreter sources are compiled into this target directly, so the
# parity ctest stays runnable while unrelated task sources (e.g. the sam3d
# vendor migration) break the main-library closure.
target_link_libraries(test_ocr_runtime PRIVATE 3rdparty_ggml 3rdparty_jsoncpp)
set_target_properties(test_ocr_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    # The backend .so modules (ggml-cuda/vulkan/cpu-*) live beside
    # libAICore.so; without $ORIGIN the DL loader cannot resolve them and
    # every device except the caller's own registrations disappears.
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
# Layout the probe under bin/aicore_tests/ together with the ggml backend
# copies (batch C test-layout fix; the probe does not go through the
# tests/CMakeLists.txt that already applies the layout function).
cloudviewer_set_aicore_test_runtime_layout(test_ocr_runtime)

# Table-structure parity probe: the SLANet encoder runs on the OCR
# interpreter, so it compiles the same closure plus the table runtime.
add_executable(test_table_runtime tests/test_table_runtime.cpp
    src/tasks/table/table_runtime.cpp
    src/tasks/ocr/ocr_runtime.cpp
    src/common/winograd.cpp
    src/common/blocked_conv.cpp
    src/common/ggml_env_bridge.cpp)
target_include_directories(test_table_runtime PRIVATE
    ${AICORE_INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/ocr
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks/table
    ${JSONCPP_INCLUDE_DIRS})
target_link_libraries(test_table_runtime PRIVATE 3rdparty_ggml 3rdparty_jsoncpp)
set_target_properties(test_table_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
cloudviewer_set_aicore_test_runtime_layout(test_table_runtime)
# The probe_ocr_* / probe_table CTests (tests/CMakeLists.txt, labels
# capi;model) run these binaries, but tests/ is configured before they
# exist, so the label-derived tier DEPENDS cannot see them: attach here.
if(TARGET aicore-model-tests)
    add_dependencies(aicore-model-tests test_ocr_runtime test_table_runtime)
endif()
if(TARGET aicore-contract-tests AND AICORE_TEST_ENABLE_MODEL_TESTS)
    add_dependencies(aicore-contract-tests test_ocr_runtime test_table_runtime)
endif()
if(TARGET AICore_capi_link)
    target_link_libraries(aicore_gguf_quantize PRIVATE AICore_capi_link)
endif()
set_target_properties(aicore_gguf_quantize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/aicore_tests"
    FOLDER "core/tools")
if(GGML_DYNAMIC_BACKENDS AND GGML_RUNTIME_LIB_DIR AND GGML_BACKEND_DIR)
        string(REPLACE ";" "|" _quantize_gpu_pipe "${GGML_GPU_BACKEND_MODULE_FILES}")
        set(_copy_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyGgmlBackends.cmake")
        add_custom_command(TARGET aicore_gguf_quantize POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DCOPY_CORE_SHARED=ON
                -DGGML_BACKEND_DIR="${GGML_RUNTIME_LIB_DIR}"
                -DDEST_DIR="$<TARGET_FILE_DIR:aicore_gguf_quantize>"
                -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
                -DLIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
                -P "${_copy_script}"
        )
        set(_quantize_backend_copy_args
            -DCOPY_BACKEND_MODULES=ON
            -DGGML_BACKEND_DIR="${GGML_BACKEND_DIR}"
            -DDEST_DIR="$<TARGET_FILE_DIR:aicore_gguf_quantize>"
            -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
            -DLIB_SUFFIX="${GGML_MODULE_SUFFIX}"
            -DGPU_FILES_PIPE="${_quantize_gpu_pipe}")
        if(GGML_CPU_ALL_VARIANTS)
            list(APPEND _quantize_backend_copy_args -DCPU_ALL_VARIANTS=ON)
        endif()
        list(APPEND _quantize_backend_copy_args -P "${_copy_script}")
        add_custom_command(TARGET aicore_gguf_quantize POST_BUILD
            COMMAND ${CMAKE_COMMAND} ${_quantize_backend_copy_args}
        )
endif()

# ReID validation probe (tools/reid_validate.py drives it against PyTorch /
# ONNX Runtime baselines; NOT a ctest — needs a classify GGUF):
#   reid_probe --gguf <file> [--device cuda|vulkan|cpu] [--output file.json]
add_executable(reid_probe tools/reid_probe.cpp)
target_link_libraries(reid_probe PRIVATE AICore)
if(TARGET AICore_capi_link)
    target_link_libraries(reid_probe PRIVATE AICore_capi_link)
endif()
# The probe includes <QImage>: match the capi-test PIC/no-PIE setup, required
# by Qt builds with -reduce-relocations.
set_target_properties(reid_probe PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/aicore_tests"
    FOLDER "core/tools")
if(NOT MSVC)
    target_compile_options(reid_probe PRIVATE -fPIC)
    target_link_options(reid_probe PRIVATE "-no-pie")
endif()
