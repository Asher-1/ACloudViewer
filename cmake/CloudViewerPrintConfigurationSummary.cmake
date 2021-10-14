# Internal helper function.
function(cloudViewer_aligned_print printed_name printed_valued)
    string(LENGTH "${printed_name}" PRINTED_NAME_LENGTH)
    math(EXPR PRINTED_DOTS_LENGTH "40 - ${PRINTED_NAME_LENGTH}")
    string(REPEAT "." ${PRINTED_DOTS_LENGTH} PRINTED_DOTS)
    message(STATUS "  ${printed_name} ${PRINTED_DOTS} ${printed_valued}")
endfunction()


# cloudViewer_print_configuration_summary()
#
# Prints a summary of the current configuration.
function(cloudViewer_print_configuration_summary)
    message(STATUS "========================================================================")
    message(STATUS "CloudViewer ${PROJECT_VERSION} Configuration Summary")
    message(STATUS "========================================================================")
    message(STATUS "Enabled Features:")
    cloudViewer_aligned_print("OpenMP" "${WITH_OPENMP}")
    cloudViewer_aligned_print("SIMD" "${USE_SIMD}")
    cloudViewer_aligned_print("Headless Rendering" "${ENABLE_HEADLESS_RENDERING}")
    cloudViewer_aligned_print("Azure Kinect Support" "${BUILD_AZURE_KINECT}")
    cloudViewer_aligned_print("Intel RealSense Support" "${BUILD_LIBREALSENSE}")
    cloudViewer_aligned_print("3D Reconstruction Support" "${BUILD_RECONSTRUCTION}")
    cloudViewer_aligned_print("CUDA Support" "${BUILD_CUDA_MODULE}")
    cloudViewer_aligned_print("Build GUI" "${BUILD_GUI}")
    cloudViewer_aligned_print("Build WebRTC visualizer" "${BUILD_WEBRTC}")
    cloudViewer_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
    if (WIN32)
        cloudViewer_aligned_print("Use Windows Static Runtime" "${STATIC_WINDOWS_RUNTIME}")
    endif ()
    cloudViewer_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
    cloudViewer_aligned_print("Build Examples" "${BUILD_EXAMPLES}")
    cloudViewer_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
    cloudViewer_aligned_print("Build Jupyter Extension" "${BUILD_JUPYTER_EXTENSION}")
    cloudViewer_aligned_print("Build Tensorflow Ops" "${BUILD_TENSORFLOW_OPS}")
    cloudViewer_aligned_print("Build Pytorch Ops" "${BUILD_PYTORCH_OPS}")
    if (BUILD_PYTORCH_OPS AND BUILD_CUDA_MODULE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
        message(WARNING
                "--------------------------------------------------------------------------------\n"
                "                                                                                \n"
                " You are compiling PyTorch ops with CUDA 11. This configuration may have        \n"
                " stability issues. See https://github.com/isl-org/Open3D/issues/3324 and        \n"
                " https://github.com/pytorch/pytorch/issues/52663 for more information on this   \n"
                " problem.                                                                       \n"
                "                                                                                \n"
                " We recommend to compile PyTorch from source with compile flags                 \n"
                "   '-Xcompiler -fno-gnu-unique'                                                 \n"
                "                                                                                \n"
                " or use the PyTorch wheels at                                                   \n"
                "   https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.1          \n"
                "                                                                                \n"
                "--------------------------------------------------------------------------------\n"
                )
    endif ()
    cloudViewer_aligned_print("Build Benchmarks" "${BUILD_BENCHMARKS}")
    cloudViewer_aligned_print("Bundle CloudViewer-ML" "${BUNDLE_CLOUDVIEWER_ML}")
    if (GLIBCXX_USE_CXX11_ABI)
        cloudViewer_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "1")
    else ()
        cloudViewer_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "0")
    endif ()

    message(STATUS "========================================================================")
    message(STATUS "Third-Party Dependencies:")
    set(3RDPARTY_DEPENDENCIES
            Eigen3
            faiss
            filament
            fmt
            GLEW
            GLFW
            imgui
            ippicv
            JPEG
            jsoncpp
            liblzf
            OpenGL
            PNG
            qhullcpp
            librealsense
            tinyfiledialogs
            TinyGLTF
            tinyobjloader
            WebRTC
            )

    foreach (dep IN LISTS 3RDPARTY_DEPENDENCIES)
        string(TOLOWER "${dep}" dep_lower)
        string(TOUPPER "${dep}" dep_upper)
        if (TARGET 3rdparty_${dep_lower})
            if (NOT USE_SYSTEM_${dep_upper})
                cloudViewer_aligned_print("${dep}" "yes (build from source)")
            else ()
                if (3rdparty_${dep_lower}_VERSION)
                    cloudViewer_aligned_print("${dep}" "yes (v${3rdparty_${dep_lower}_VERSION})")
                else ()
                    cloudViewer_aligned_print("${dep}" "yes")
                endif ()
            endif ()
        else ()
            cloudViewer_aligned_print("${dep}" "no")
        endif ()
    endforeach ()
    message(STATUS "================================================================================")

endfunction()
