# file(GLOB_RECURSE ALL_CORE_HEADER_FILES "*.h")
add_library(core OBJECT )

target_sources(core PRIVATE
        AdvancedIndexing.cpp
        CUDAUtils.cpp
        Device.cpp
        Dtype.cpp
        Gpu.cpp
        EigenConverter.cpp
        SYCLUtils.cpp
        Indexer.cpp
        MemoryManager.cpp
        MemoryManagerCached.cpp
        MemoryManagerCPU.cpp
        MemoryManagerStatistic.cpp
        ShapeUtil.cpp
        SizeVector.cpp
        SmallVector.cpp
        Tensor.cpp
        TensorCheck.cpp
        TensorFunction.cpp
        TensorKey.cpp
        TensorList.cpp
        )

target_sources(core PRIVATE
        hashmap/HashMap.cpp
        hashmap/HashSet.cpp
        hashmap/DeviceHashBackend.cpp
        hashmap/HashBackendBuffer.cpp
        hashmap/CPU/CPUHashBackendBuffer.cpp
        hashmap/CPU/CreateCPUHashBackend.cpp
)

target_sources(core PRIVATE
        kernel/Kernel.cpp
        kernel/Arange.cpp
        kernel/BinaryEW.cpp
        kernel/IndexGetSet.cpp
        kernel/IndexReduction.cpp
        kernel/NonZero.cpp
        kernel/Reduction.cpp
        kernel/UnaryEW.cpp
        kernel/ArangeCPU.cpp
        kernel/BinaryEWCPU.cpp
        kernel/IndexGetSetCPU.cpp
        kernel/IndexReductionCPU.cpp
        kernel/NonZeroCPU.cpp
        kernel/ReductionCPU.cpp
        kernel/UnaryEWCPU.cpp
)

target_sources(core PRIVATE
        linalg/AddMM.cpp
        linalg/AddMMCPU.cpp
        linalg/Det.cpp
        linalg/Inverse.cpp
        linalg/InverseCPU.cpp
        linalg/LeastSquares.cpp
        linalg/LeastSquaresCPU.cpp
        linalg/LU.cpp
        linalg/LUCPU.cpp
        linalg/Matmul.cpp
        linalg/MatmulCPU.cpp
        linalg/Solve.cpp
        linalg/SolveCPU.cpp
        linalg/SVD.cpp
        linalg/SVDCPU.cpp
        linalg/Tri.cpp
        linalg/TriCPU.cpp
        )

target_sources(core PRIVATE
        nns/FixedRadiusIndex.cpp
        nns/FixedRadiusSearchOps.cpp
        nns/KnnIndex.cpp
        nns/NanoFlannIndex.cpp
        nns/NearestNeighborSearch.cpp
        nns/NNSIndex.cpp
        )

if (BUILD_CUDA_MODULE)
    target_sources(core PRIVATE
            MemoryManagerCUDA.cpp
            )

    target_sources(core PRIVATE
            hashmap/CUDA/CreateCUDAHashBackend.cu
            hashmap/CUDA/CUDAHashBackendBuffer.cu
            hashmap/CUDA/SlabNodeManager.cu
            )

    target_sources(core PRIVATE
        kernel/ArangeCUDA.cu
        kernel/BinaryEWCUDA.cu
        kernel/IndexGetSetCUDA.cu
        kernel/IndexReductionCUDA.cu
        kernel/NonZeroCUDA.cu
        kernel/ReductionCUDA.cu
        kernel/UnaryEWCUDA.cu
     )

    target_sources(core PRIVATE
        linalg/AddMMCUDA.cpp
        linalg/InverseCUDA.cpp
        linalg/LeastSquaresCUDA.cpp
        linalg/LinalgUtils.cpp
        linalg/LUCUDA.cpp
        linalg/MatmulCUDA.cpp
        linalg/SolveCUDA.cpp
        linalg/SVDCUDA.cpp
        linalg/TriCUDA.cu
        )

    target_sources(core PRIVATE
            nns/FixedRadiusSearchOps.cu
            nns/kernel/BlockSelectFloat32.cu
            nns/kernel/BlockSelectFloat64.cu
            nns/KnnSearchOps.cu
            )
endif ()

cloudViewer_show_and_abort_on_warning(core)
cloudViewer_set_global_properties(core)
cloudViewer_set_cloudViewer_lib_properties(core)
cloudViewer_link_3rdparty_libraries(core)

if (BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()
