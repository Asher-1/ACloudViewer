if (BUILD_CUDA_MODULE)
    message(STATUS "Building PyTorch ops with CUDA")
else ()
    message(STATUS "Building PyTorch ops")
endif ()

find_package(Pytorch REQUIRED)

if (Python3_VERSION VERSION_GREATER_EQUAL 3.9 AND Pytorch_VERSION VERSION_LESS 1.8.0)
    message(FATAL_ERROR "Please update to PyTorch 1.8.0+ to build PyTorch Ops "
            "with Python 3.9 to prevent a segmentation fault. See "
            "https://github.com/pytorch/pytorch/issues/50014 for details")
endif ()

add_library(cloudViewer_torch_ops SHARED)

target_sources(cloudViewer_torch_ops PRIVATE
        continuous_conv/ContinuousConvBackpropFilterOpKernel.cpp
        continuous_conv/ContinuousConvOpKernel.cpp
        continuous_conv/ContinuousConvOps.cpp
        continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cpp
        continuous_conv/ContinuousConvTransposeOpKernel.cpp
        continuous_conv/ContinuousConvTransposeOps.cpp
        )

target_sources(cloudViewer_torch_ops PRIVATE
        sparse_conv/SparseConvBackpropFilterOpKernel.cpp
        sparse_conv/SparseConvOpKernel.cpp
        sparse_conv/SparseConvOps.cpp
        sparse_conv/SparseConvTransposeBackpropFilterOpKernel.cpp
        sparse_conv/SparseConvTransposeOpKernel.cpp
        sparse_conv/SparseConvTransposeOps.cpp
        )

target_sources(cloudViewer_torch_ops PRIVATE
        misc/BuildSpatialHashTableOpKernel.cpp
        misc/BuildSpatialHashTableOps.cpp
        misc/FixedRadiusSearchOpKernel.cpp
        misc/FixedRadiusSearchOps.cpp
        misc/InvertNeighborsListOpKernel.cpp
        misc/InvertNeighborsListOps.cpp
        misc/KnnSearchOpKernel.cpp
        misc/KnnSearchOps.cpp
        misc/NmsOps.cpp
        misc/RadiusSearchOpKernel.cpp
        misc/RadiusSearchOps.cpp
        misc/RaggedToDenseOpKernel.cpp
        misc/RaggedToDenseOps.cpp
        misc/ReduceSubarraysSumOpKernel.cpp
        misc/ReduceSubarraysSumOps.cpp
        misc/RoiPoolOps.cpp
        misc/VoxelizeOpKernel.cpp
        misc/VoxelizeOps.cpp
        misc/VoxelPoolingOpKernel.cpp
        misc/VoxelPoolingOps.cpp
        )

target_sources(cloudViewer_torch_ops PRIVATE
        ragged_tensor/RaggedTensor.cpp)

target_sources(cloudViewer_torch_ops PRIVATE
        pointnet/BallQueryOps.cpp
        pointnet/InterpolateOps.cpp
        pointnet/SamplingOps.cpp
        pvcnn/TrilinearDevoxelizeOps.cpp
        )

target_sources(cloudViewer_torch_ops PRIVATE
        ../contrib/Nms.cpp
        )

if (BUILD_CUDA_MODULE)
    target_sources(cloudViewer_torch_ops PRIVATE
            continuous_conv/ContinuousConvBackpropFilterOpKernel.cu
            continuous_conv/ContinuousConvOpKernel.cu
            continuous_conv/ContinuousConvTransposeBackpropFilterOpKernel.cu
            continuous_conv/ContinuousConvTransposeOpKernel.cu
            )

    target_sources(cloudViewer_torch_ops PRIVATE
            sparse_conv/SparseConvBackpropFilterOpKernel.cu
            sparse_conv/SparseConvOpKernel.cu
            sparse_conv/SparseConvTransposeBackpropFilterOpKernel.cu
            sparse_conv/SparseConvTransposeOpKernel.cu
            )

    target_sources(cloudViewer_torch_ops PRIVATE
            misc/BuildSpatialHashTableOpKernel.cu
            misc/FixedRadiusSearchOpKernel.cu
            misc/InvertNeighborsListOpKernel.cu
            misc/RaggedToDenseOpKernel.cu
            misc/ReduceSubarraysSumOpKernel.cu
            misc/VoxelizeOpKernel.cu
            )

    target_sources(cloudViewer_torch_ops PRIVATE
            pointnet/BallQueryKernel.cu
            pointnet/InterpolateKernel.cu
            pointnet/SamplingKernel.cu
            pvcnn/TrilinearDevoxelizeKernel.cu
            )

    target_sources(cloudViewer_torch_ops PRIVATE
            ../impl/continuous_conv/ContinuousConvCUDAKernels.cu
            ../impl/sparse_conv/SparseConvCUDAKernels.cu
            )

    target_sources(cloudViewer_torch_ops PRIVATE
            ../contrib/BallQuery.cu
            ../contrib/InterpolatePoints.cu
            ../contrib/Nms.cu
            ../contrib/RoiPoolKernel.cu
            ../contrib/TrilinearDevoxelize.cu
            )
endif ()

cloudViewer_show_and_abort_on_warning(cloudViewer_torch_ops)
cloudViewer_set_global_properties(cloudViewer_torch_ops)

# Set output directory according to architecture (cpu/cuda)
get_target_property(TORCH_OPS_DIR cloudViewer_torch_ops LIBRARY_OUTPUT_DIRECTORY)
set(TORCH_OPS_ARCH_DIR
        "${TORCH_OPS_DIR}/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
# cloudViewer_torch_ops.{dylib|so|dll} the default location is build/lib/${CMAKE_BUILD_TYPE}/Python/{cpu|cuda}`
set(TORCH_OPS_RUNTIME_ARCH_DIR
        "${TORCH_OPS_DIR}/Python/$<IF:$<BOOL:${BUILD_CUDA_MODULE}>,cuda,cpu>")
set_target_properties(cloudViewer_torch_ops PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${TORCH_OPS_ARCH_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${TORCH_OPS_ARCH_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${TORCH_OPS_RUNTIME_ARCH_DIR}")

# Do not add "lib" prefix
set_target_properties(cloudViewer_torch_ops PROPERTIES PREFIX "")
set_target_properties(cloudViewer_torch_ops PROPERTIES DEBUG_POSTFIX "_debug")
# Set BUILD_RPATH to find tbb. We don't install through cmake.
# if (APPLE)
#     set_target_properties(cloudViewer_torch_ops PROPERTIES BUILD_RPATH "@loader_path/..;@loader_path/../lib")
# elseif (UNIX)
#     set_target_properties(cloudViewer_torch_ops PROPERTIES BUILD_RPATH "$ORIGIN/..;$ORIGIN/../lib")
# endif()

target_include_directories(cloudViewer_torch_ops SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/libs/cloudViewer
        ${TORCH_INCLUDE_DIRS}
        )

target_link_libraries(cloudViewer_torch_ops PRIVATE
        torch_cpu
        CloudViewer::CloudViewer
        3rdparty_eigen3
        3rdparty_fmt
        3rdparty_nanoflann
        3rdparty_tbb
        )

if (TARGET 3rdparty_parallelstl)
        target_link_libraries(cloudViewer_torch_ops PRIVATE
                3rdparty_parallelstl
        )
endif()

if (BUILD_CUDA_MODULE)
    target_link_libraries(cloudViewer_torch_ops PRIVATE
            3rdparty_cutlass
            ${TORCH_LIBRARIES}
            CUDA::cuda_driver
            )

    if (TARGET 3rdparty_cub)
        target_link_libraries(cloudViewer_torch_ops PRIVATE
                3rdparty_cub
                )
    endif ()
endif ()