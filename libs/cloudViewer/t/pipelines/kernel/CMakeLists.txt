add_library(tpipelines_kernel OBJECT)

target_sources(tpipelines_kernel  PRIVATE
    Registration.cpp
    RegistrationCPU.cpp
    FillInLinearSystem.cpp
    FillInLinearSystemCPU.cpp
    RGBDOdometry.cpp
    RGBDOdometryCPU.cpp
    TransformationConverter.cpp
    Feature.cpp
    FeatureCPU.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tpipelines_kernel  PRIVATE
        RegistrationCUDA.cu
        FillInLinearSystemCUDA.cu
        RGBDOdometryCUDA.cu
        TransformationConverter.cu
        FeatureCUDA.cu
    )
endif()

cloudViewer_show_and_abort_on_warning(tpipelines_kernel)
cloudViewer_set_global_properties(tpipelines_kernel)
# The kernels are used in the unit tests, so they cannot be hidden for now.
cloudViewer_set_cloudViewer_lib_properties(tpipelines_kernel)
# cloudViewer_set_cloudViewer_lib_properties(tpipelines_kernel HIDDEN)
cloudViewer_link_3rdparty_libraries(tpipelines_kernel)

if(BUILD_CUDA_MODULE)
    target_include_directories(tpipelines_kernel SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
