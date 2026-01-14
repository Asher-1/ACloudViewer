target_sources(tests PRIVATE
    # Blob.cpp  # TODO: Fix API compatibility
    CoreTest.cpp
    # CUDAState.cpp  # TODO: Fix CUDAState API (GetNumDevices not available)
    CUDAUtils.cpp
    Device.cpp
    EigenConverter.cpp
    FixedRadiusIndex.cpp
    # Hashmap.cpp  # TODO: Fix HashMap API changes
    # HashMap.cpp  # TODO: Fix HashMap API changes
    Indexer.cpp
    KnnIndex.cpp
    Linalg.cpp
    # MemoryManager.cpp  # TODO: Fix API
    NanoFlannIndex.cpp
    # NearestNeighborSearch.cpp  # TODO: Fix HybridSearch API
    # ParallelFor.cpp  # TODO: Fix OPEN3D_TEMPLATE_VECTORIZED macro
    # ParallelForSYCL.cpp  # TODO: Fix SYCL includes
    Scalar.cpp
    ShapeUtil.cpp
    # SizeVector.cpp  # TODO: Fix API
    # SYCLUtils.cpp  # TODO: Fix SYCL includes
    # TensorCheck.cpp  # TODO: Fix API
    # Tensor.cpp  # TODO: Fix Tensor.Copy() API
    TensorFunction.cpp
    # TensorList.cpp  # TODO: Fix API
    # TensorObject.cpp  # TODO: Fix API
)
