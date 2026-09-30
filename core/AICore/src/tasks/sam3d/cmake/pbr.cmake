# SAM3D official native PBR assembly — task-private build wiring.
#
# Upstream sam-3d-objects-ggml assemble_official_pbr_cuda chain (CUDA-only):
# VTK quadric simplify (0.95) → 1000-view nvdiffrast visibility → CPU min-cut
# filter → pymeshfix TMesh repair (GPL) → xatlas UV → angle-weighted normals
# → 100-view mip-splatting Gaussian render → nvdiffrast 2500-step Adam/TV
# texture bake → Telea inpaint → Y-up GLB.
#
# The user-facing switch (AICore_USE_SAM3D_PBR) and its two license
# acknowledgment switches are DECLARED in cmake/AICoreOptions.cmake — declared
# before find_dependencies.cmake so the bundled OpenCV build can wire its
# photo module from the same switch. The AICore_USE_CUDA hard requirement is
# the only cross-module linkage and is enforced by core/AICore/CMakeLists.txt
# before this file is included. Everything below is task-owned: missing-module
# diagnostics, vendored third-party builds and the pipeline source list.
#
# License isolation notes (vendored, fixed-version source integration):
# all four third-party trees live under pbr/third_party/ and are compiled
# only under this gate — the default AICore build is untouched. nvdiffrast
# (NVIDIA, non-commercial; the csrc/common/cudaraster subtree of pinned
# 253ac4f) and pymeshfix TMesh (GPL-3.0-or-commercial, upstream src/ of
# pinned 79adef9) ship WITH their license texts in-tree; their sources are
# distributed with the repo but never compiled or linked unless the
# acknowledgment switches are set. mip-splatting (Inria non-commercial) and
# the MIT xatlas snapshot mirror the upstream sam3dggml vendoring.

include_guard(GLOBAL)

function(aicore_sam3d_wire_pbr target sam3d_src_dir)
    set(AICORE_SAM3D_PBR_SRC_DIR ${sam3d_src_dir}/pbr)
    if(NOT EXISTS ${AICORE_SAM3D_PBR_SRC_DIR}/native_pbr_pipeline.cpp)
        message(
            FATAL_ERROR
                "AICore: missing SAM3D PBR module at ${AICORE_SAM3D_PBR_SRC_DIR}"
        )
    endif()
    # The official chain consumes the project-managed 3rdparty_vtk target.
    # Auto-enabled builds may run on configurations where the VTK section of
    # find_dependencies never created it (no VTK backend, no PCL) — degrade
    # the same way CuMesh does (warning + drop) instead of failing the
    # build; info_json then reports "pbr":false and the plugin falls back
    # to the backend-agnostic CPU projection bake.
    if(NOT TARGET 3rdparty_vtk)
        message(WARNING
                "AICore: AICore_USE_SAM3D_PBR is ON but the project-managed "
                "3rdparty_vtk target does not exist in this configuration "
                "(no VTK backend / PCL consumer) — the official PBR chain is "
                "DROPPED from this build. Enable USE_VTK_BACKEND (or "
                "USE_SYSTEM_VTK) to keep it.")
        return()
    endif()
    # VTK comes from the project-managed 3rdparty_vtk target (self-contained:
    # a prebuilt 9.3 archive is downloaded, or VTK is built from source, or
    # USE_SYSTEM_VTK opt-ins the system package — see
    # 3rdparty/vtk/vtk_build_or_download.cmake; all three carry FiltersCore).
    # OpenCV rides the project-wide 3rdparty_opencv target when it exists
    # (bundled build or USE_SYSTEM_OPENCV — both create it in
    # find_dependencies.cmake, which runs before this wiring). Only builds
    # with OpenCV entirely out of the dependency graph fall back to a system
    # find_package for the Telea-inpaint stage.
    if(NOT TARGET 3rdparty_opencv)
        find_package(OpenCV 4.2 REQUIRED COMPONENTS core photo)
    endif()
    find_package(Threads REQUIRED)

    # MIT xatlas snapshot vendored from the upstream sam3dggml tree. Kept
    # separate from the TRELLIS xatlas (the snapshots differ) so mesh_uv.cpp
    # parameterizes with the exact upstream library for bit parity.
    add_library(AICore_sam3d_xatlas STATIC
        ${AICORE_SAM3D_PBR_SRC_DIR}/third_party/xatlas/xatlas.cpp)
    target_include_directories(AICore_sam3d_xatlas SYSTEM PUBLIC
        ${AICORE_SAM3D_PBR_SRC_DIR}/third_party/xatlas)
    target_link_libraries(AICore_sam3d_xatlas PUBLIC Threads::Threads)

    # Inria mip-splatting forward Gaussian rasterizer (non-commercial),
    # vendored identically to upstream third_party; compiled as-is except the
    # single gsplat-parity delta recorded in the fork's UPSTREAM.md.
    set(AICORE_SAM3D_MIP_DIR
        ${AICORE_SAM3D_PBR_SRC_DIR}/third_party/mip-splatting-diff-gaussian-rasterization)
    if(NOT EXISTS ${AICORE_SAM3D_MIP_DIR}/cuda_rasterizer/rasterizer_impl.cu)
        message(
            FATAL_ERROR
                "AICore: missing SAM3D PBR Gaussian rasterizer at ${AICORE_SAM3D_MIP_DIR}"
        )
    endif()
    add_library(AICore_sam3d_mip_gaussian STATIC
        ${AICORE_SAM3D_MIP_DIR}/cuda_rasterizer/backward.cu
        ${AICORE_SAM3D_MIP_DIR}/cuda_rasterizer/forward.cu
        ${AICORE_SAM3D_MIP_DIR}/cuda_rasterizer/rasterizer_impl.cu)
    target_include_directories(AICore_sam3d_mip_gaussian
        PUBLIC ${AICORE_SAM3D_MIP_DIR}/cuda_rasterizer
        PRIVATE ${AICORE_SAM3D_MIP_DIR}/third_party/glm)
    target_link_libraries(AICore_sam3d_mip_gaussian PUBLIC CUDA::cudart)
    set_target_properties(AICore_sam3d_mip_gaussian PROPERTIES
        CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_EXTENSIONS OFF)

    # nvdiffrast is the official optimized UV/face rasterizer; non-commercial,
    # so it is compiled and linked only when the acknowledgment option is ON.
    # Vendored subtree: the csrc/common/cudaraster core of pinned 253ac4f
    # (the wrapper sources include it as <csrc/common/cudaraster/impl/...>,
    # so the include path shape must be preserved) plus framework.h.
    if(AICore_USE_SAM3D_PBR_NVDIFFRAST)
        set(AICORE_SAM3D_NVDR_DIR
            ${AICORE_SAM3D_PBR_SRC_DIR}/third_party/nvdiffrast)
        if(NOT EXISTS ${AICORE_SAM3D_NVDR_DIR}/csrc/common/cudaraster/impl/CudaRaster.cpp)
            message(
                FATAL_ERROR
                    "AICore: missing vendored nvdiffrast cudaraster at ${AICORE_SAM3D_NVDR_DIR}"
            )
        endif()
        add_library(AICore_sam3d_nvdiffrast STATIC
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_buffer.cpp
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_cuda_raster.cpp
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_raster_impl.cpp
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_raster_kernel.cu
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_rasterize_shader.cu
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_interpolate_shader.cu
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_texture.cpp
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_texture_kernel.cu
            ${AICORE_SAM3D_PBR_SRC_DIR}/nvdiffrast_standalone/nvdiffrast_common.cpp)
        target_include_directories(AICore_sam3d_nvdiffrast
            PUBLIC ${AICORE_SAM3D_NVDR_DIR}/csrc/common/cudaraster
            PRIVATE ${AICORE_SAM3D_NVDR_DIR})
        target_link_libraries(AICore_sam3d_nvdiffrast PUBLIC CUDA::cudart)
        set_target_properties(AICore_sam3d_nvdiffrast PROPERTIES
            CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_EXTENSIONS OFF)
    endif()

    # The official postprocess invokes pymeshfix.PyTMesh directly; its TMesh
    # source is GPL-3.0-or-commercial, so the exact boundary repair is a
    # deliberate opt-in and never enters the default runtime. The vendored
    # src/ is the upstream tree of pinned 79adef9 (the Python extension
    # binding is not part of the library build).
    if(AICore_USE_SAM3D_PBR_MESHFIX)
        set(_aicore_sam3d_meshfix_dir
            ${AICORE_SAM3D_PBR_SRC_DIR}/third_party/pymeshfix/src)
        if(NOT EXISTS ${_aicore_sam3d_meshfix_dir}/checkAndRepair.cpp)
            message(
                FATAL_ERROR
                    "AICore: missing vendored pymeshfix sources at ${_aicore_sam3d_meshfix_dir}"
            )
        endif()
        add_library(AICore_sam3d_meshfix STATIC
            ${_aicore_sam3d_meshfix_dir}/checkAndRepair.cpp
            ${_aicore_sam3d_meshfix_dir}/coordinates.cpp
            ${_aicore_sam3d_meshfix_dir}/detectIntersections.cpp
            ${_aicore_sam3d_meshfix_dir}/edge.cpp
            ${_aicore_sam3d_meshfix_dir}/graph.cpp
            ${_aicore_sam3d_meshfix_dir}/heap.cpp
            ${_aicore_sam3d_meshfix_dir}/holeFilling.cpp
            ${_aicore_sam3d_meshfix_dir}/io.cpp
            ${_aicore_sam3d_meshfix_dir}/jqsort.cpp
            ${_aicore_sam3d_meshfix_dir}/list.cpp
            ${_aicore_sam3d_meshfix_dir}/marchIntersections.cpp
            ${_aicore_sam3d_meshfix_dir}/matrix.cpp
            ${_aicore_sam3d_meshfix_dir}/orientation.c
            ${_aicore_sam3d_meshfix_dir}/point.cpp
            ${_aicore_sam3d_meshfix_dir}/tin.cpp
            ${_aicore_sam3d_meshfix_dir}/tmesh.cpp
            ${_aicore_sam3d_meshfix_dir}/triangle.cpp
            ${_aicore_sam3d_meshfix_dir}/vertex.cpp)
        target_include_directories(AICore_sam3d_meshfix PUBLIC
            ${_aicore_sam3d_meshfix_dir})
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            target_compile_definitions(AICore_sam3d_meshfix PUBLIC
                IS64BITPLATFORM)
        endif()
        set_target_properties(AICore_sam3d_meshfix PROPERTIES
            CXX_STANDARD 17 CXX_EXTENSIONS OFF)
    endif()

    # Pipeline sources into libAICore. mesh_rasterizer_cuda.cu,
    # texture_baker_cuda.cu and native_pbr_pipeline.cpp need the nvdiffrast
    # headers (their include lines are unconditional, same as upstream), so
    # they only join the build under the nvdiffrast acknowledgment.
    list(APPEND AICORE_SAM3D_PBR_SOURCES
        ${AICORE_SAM3D_PBR_SRC_DIR}/mesh_postprocess.cpp
        ${AICORE_SAM3D_PBR_SRC_DIR}/mesh_uv.cpp
        ${AICORE_SAM3D_PBR_SRC_DIR}/mesh_rasterizer.cpp
        ${AICORE_SAM3D_PBR_SRC_DIR}/mesh_boundary_repair.cpp
        ${AICORE_SAM3D_PBR_SRC_DIR}/texture_inpaint.cpp
        ${AICORE_SAM3D_PBR_SRC_DIR}/gaussian_renderer.cpp
        ${AICORE_SAM3D_PBR_SRC_DIR}/gaussian_renderer.cu)
    if(AICore_USE_SAM3D_PBR_NVDIFFRAST)
        list(APPEND AICORE_SAM3D_PBR_SOURCES
            ${AICORE_SAM3D_PBR_SRC_DIR}/mesh_rasterizer_cuda.cu
            ${AICORE_SAM3D_PBR_SRC_DIR}/texture_baker_cuda.cu
            ${AICORE_SAM3D_PBR_SRC_DIR}/native_pbr_pipeline.cpp)
    endif()
    target_sources(${target} PRIVATE ${AICORE_SAM3D_PBR_SOURCES})
    target_include_directories(${target} PRIVATE
        ${AICORE_SAM3D_PBR_SRC_DIR})
    if(OpenCV_INCLUDE_DIRS)
        target_include_directories(${target} PRIVATE
            ${OpenCV_INCLUDE_DIRS})
    endif()
    target_compile_definitions(${target} PRIVATE SAM3D_NATIVE_PBR_CUDA)
    # Reuse the project-wide OpenCV wrapper (ext_opencv world build on
    # configure-time machines, system per-module libs otherwise) so the link
    # resolves on both layouts; q3DMASC/qAnyLabeling follow the same pattern.
    if(TARGET 3rdparty_opencv)
        target_link_libraries(${target} PRIVATE
            AICore_sam3d_xatlas AICore_sam3d_mip_gaussian 3rdparty_vtk
            3rdparty_opencv)
        add_dependencies(${target} 3rdparty_opencv)
    else()
        target_link_libraries(${target} PRIVATE
            AICore_sam3d_xatlas AICore_sam3d_mip_gaussian 3rdparty_vtk
            ${OpenCV_LIBS})
    endif()
    if(AICore_USE_SAM3D_PBR_NVDIFFRAST)
        target_link_libraries(${target} PRIVATE AICore_sam3d_nvdiffrast
                                                CUDA::cublas)
        target_include_directories(${target} PRIVATE
            ${AICORE_SAM3D_PBR_SRC_DIR}/third_party/nvdiffrast)
        target_compile_definitions(${target} PRIVATE
            SAM3D_USE_NVDIFFRAST_CUDARASTER)
    endif()
    if(AICore_USE_SAM3D_PBR_MESHFIX)
        target_link_libraries(${target} PRIVATE AICore_sam3d_meshfix)
        target_compile_definitions(${target} PRIVATE
            SAM3D_USE_MESHFIX_GPL)
    endif()
endfunction()
