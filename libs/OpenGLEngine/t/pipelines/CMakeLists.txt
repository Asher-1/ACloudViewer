# Build
set(REGISTRATION_SRC
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
)

set(KERNEL_SRC
    kernel/TransformationConverter.cpp
    kernel/ComputePosePointToPlane.cpp
    kernel/ComputePosePointToPlaneCPU.cpp
)

set(KERNEL_CUDA_SRC
    kernel/TransformationConverter.cu
    kernel/ComputePosePointToPlaneCUDA.cu
)

set(ALL_PIPELINE_SRC
    ${REGISTRATION_SRC}
    ${KERNEL_SRC}
)

if(BUILD_CUDA_MODULE)
    set(ALL_PIPELINE_SRC
        ${ALL_PIPELINE_SRC}
        ${KERNEL_CUDA_SRC}
    )
endif()

# Create object library
add_library(tpipelines OBJECT ${ALL_PIPELINE_SRC})
cloudViewer_show_and_abort_on_warning(tpipelines)
cloudViewer_set_global_properties(tpipelines)
cloudViewer_set_cloudViewer_lib_properties(tpipelines)
cloudViewer_link_3rdparty_libraries(tpipelines)

if(BUILD_CUDA_MODULE)
    target_include_directories(tpipelines SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
