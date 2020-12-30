set(KERNEL_SRC
    kernel/IndexGetSet.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/NonZero.cpp
    kernel/NonZeroCPU.cpp
    kernel/UnaryEW.cpp
    kernel/UnaryEWCPU.cpp
    kernel/BinaryEW.cpp
    kernel/BinaryEWCPU.cpp
    kernel/GeneralEW.cpp
    kernel/GeneralEWCPU.cpp
    kernel/Reduction.cpp
    kernel/ReductionCPU.cpp
    kernel/Kernel.cpp
)

set(KERNEL_CUDA_SRC
    kernel/IndexGetSetCUDA.cu
    kernel/NonZeroCUDA.cu
    kernel/UnaryEWCUDA.cu
    kernel/BinaryEWCUDA.cu
    kernel/GeneralEWCUDA.cu
    kernel/ReductionCUDA.cu    
)

set(LINALG_SRC
    linalg/Matmul.cpp
    linalg/MatmulCPU.cpp
    linalg/LeastSquares.cpp
    linalg/LeastSquaresCPU.cpp
    linalg/Solve.cpp
    linalg/SolveCPU.cpp
    linalg/Inverse.cpp
    linalg/InverseCPU.cpp
    linalg/SVD.cpp
    linalg/SVDCPU.cpp
)

set(LINALG_CUDA_SRC
    linalg/LinalgUtils.cpp
    linalg/MatmulCUDA.cpp
    linalg/LeastSquaresCUDA.cpp
    linalg/SolveCUDA.cpp
    linalg/InverseCUDA.cpp
    linalg/SVDCUDA.cpp
)

set(CORE_SRC
    AdvancedIndexing.cpp
    ShapeUtil.cpp
    CUDAUtils.cpp
    Dtype.cpp
    EigenConverter.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    Tensor.cpp
    TensorKey.cpp
    TensorList.cpp
)

set(CORE_CUDA_SRC
    MemoryManagerCUDACached.cu
    MemoryManagerCUDASimple.cu
)

set(HASHMAP_SRC
  hashmap/Hashmap.cpp
  hashmap/DeviceHashmap.cpp
  hashmap/CPU/DefaultHashmapCPU.cpp
)

set(HASHMAP_CUDA_SRC
  hashmap/CUDA/DefaultHashmapCUDA.cu
  )

set(CORE_NNS_SRC
    nns/NNSIndex.cpp
    nns/NanoFlannIndex.cpp
    nns/NearestNeighborSearch.cpp
    nns/FixedRadiusIndex.cpp
)

if (WITH_FAISS)
    set(CORE_NNS_SRC ${CORE_NNS_SRC} nns/FaissIndex.cpp)
endif()

if (BUILD_CUDA_MODULE)
    set(CORE_NNS_SRC ${CORE_NNS_SRC} nns/FixedRadiusSearch.cu)
endif()

if (BUILD_CUDA_MODULE)
    list(APPEND CORE_NNS_SRC nns/FixedRadiusSearch.cu)
endif()

if(BUILD_CUDA_MODULE)
    set(ALL_CORE_SRC
        ${CORE_SRC}
        ${CORE_CUDA_SRC}
        ${LINALG_SRC}
        ${LINALG_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
        ${HASHMAP_SRC}
        ${HASHMAP_CUDA_SRC}
        ${CORE_NNS_SRC}
    )
else()
  set(ALL_CORE_SRC
        ${CORE_SRC}
        ${LINALG_SRC}
        ${KERNEL_SRC}
        ${HASHMAP_SRC}
        ${CORE_NNS_SRC}
    )
endif()

# Create object library
add_library(core OBJECT ${ALL_CORE_SRC} ${ALL_HEADER_FILES})
cloudViewer_set_global_properties(core)
cloudViewer_set_cloudViewer_lib_properties(core)
ShowAndAbortOnWarning(core)

target_link_libraries(core PRIVATE ${CloudViewer_3RDPARTY_PRIVATE_TARGETS} CVCoreLib)
target_link_libraries(core PUBLIC ${CloudViewer_3RDPARTY_PUBLIC_TARGETS} CVCoreLib)
foreach(dep IN LISTS CloudViewer_3RDPARTY_HEADER_TARGETS)
    if(TARGET ${dep})
        get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        if(inc)
            set_property(TARGET core APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
        endif()
        get_property(inc TARGET core PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
        if(inc)
            set_property(TARGET core APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
        endif()
        get_property(def TARGET core PROPERTY INTERFACE_COMPILE_DEFINITIONS)
        if(def)
            set_property(TARGET core APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
        endif()
    endif()
endforeach()
#cloudViewer_link_3rdparty_libraries(core)

if(BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    find_package(CUB REQUIRED)
    target_include_directories(core SYSTEM PRIVATE ${CUB_INCLUDE_DIR})
endif()
