if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-narrowing")
endif()

if(NOT SIMD_ENABLED)
    add_compile_definitions(DISABLE_CPU_NEON)
    add_compile_definitions(DISABLE_CPU_AVX)
    add_compile_definitions(DISABLE_CPU_SSE)
endif()

if(CUDA_ENABLED)
    COLMAP_ADD_STATIC_CUDA_LIBRARY(pba
        ConfigBA.cpp
        ConfigBA.h
        CuTexImage.cpp
        CuTexImage.h
        DataInterface.h
        pba.cpp
        pba.h
        ProgramCU.cu
        ProgramCU.h
        SparseBundleCPU.cpp
        SparseBundleCPU.h
        SparseBundleCU.cpp
        SparseBundleCU.h
        util.h
    )
else()
    add_compile_definitions(PBA_NO_GPU)

    COLMAP_ADD_STATIC_LIBRARY(pba
        ConfigBA.cpp
        ConfigBA.h
        DataInterface.h
        pba.cpp
        pba.h
        SparseBundleCPU.cpp
        SparseBundleCPU.h
        util.h
    )
endif()
