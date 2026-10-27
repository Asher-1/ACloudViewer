cmake_minimum_required(VERSION 3.19)
project(ACloudViewerLlamaMigration LANGUAGES C CXX)

# Validation driver only: the dependency definition lives in llama_cpp.cmake.
set(CloudViewer_INSTALL_LIB_DIR lib)
set(CLOUDVIEWER_EXTERNAL_INSTALL_DIR "${CMAKE_BINARY_DIR}/install")
set(CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads" CACHE PATH
    "Pinned source/header cache, separate from the dependency install prefix")
set(GGML_ENABLED ON CACHE BOOL "Build inference runtime")
set(GGML_BUILD_SHARED ON CACHE BOOL "Dynamic backend modules")
foreach(_backend IN ITEMS CUDA VULKAN METAL SYCL OPENCL)
    set(GGML_USE_${_backend} OFF CACHE BOOL "Validation driver backend")
endforeach()
option(LLAMA_METAL_MIGRATION_VALIDATION
    "Opt into unqualified Metal patches in this isolated validation driver" OFF)
option(GGML_CPU_ALL_VARIANTS "Validation driver CPU variants" ON)
if(NOT GGML_CPU_ALL_VARIANTS)
    message(FATAL_ERROR
        "Controlled llama migration validation requires GGML_CPU_ALL_VARIANTS=ON; "
        "reconfigure with -DGGML_CPU_ALL_VARIANTS=ON to match the frozen baseline")
endif()
set(GGML_USE_SAM3D_OPS ON CACHE BOOL "Include the AICore SAM3D operator closure")
set(GGML_CUDA_FORCE_MMQ OFF CACHE BOOL "Match AICore CUDA default")
set(GGML_CUDA_FAST_MATH ON CACHE BOOL "Match AICore CUDA default")
set(WITH_OPENMP ON)
set(ExternalProject_CMAKE_ARGS_hidden
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    # Match find_dependencies.cmake: the nested build does not inherit these
    # target properties from the validation driver automatically.
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON)
include("${CMAKE_CURRENT_LIST_DIR}/../llama_cpp.cmake")

include(CTest)
if(BUILD_TESTING)
    add_executable(llama_runtime_smoke runtime_smoke.cpp)
    target_link_libraries(llama_runtime_smoke PRIVATE 3rdparty_llama)
    target_compile_definitions(llama_runtime_smoke PRIVATE
        ACV_GGML_MODULE_DIR="${GGML_MODULE_DIR}")
    add_test(NAME llama_runtime_smoke COMMAND llama_runtime_smoke)
    add_executable(llama_elementwise_epilogue_test elementwise_epilogue_test.cpp)
    target_link_libraries(llama_elementwise_epilogue_test PRIVATE 3rdparty_llama)
    add_test(NAME llama_cpu_elementwise_epilogue
        COMMAND llama_elementwise_epilogue_test "${GGML_MODULE_DIR}" CPU)
    add_executable(llama_depthwise_bias_test depthwise_bias_test.cpp)
    target_link_libraries(llama_depthwise_bias_test PRIVATE 3rdparty_llama)
    add_test(NAME llama_cpu_depthwise_bias
        COMMAND llama_depthwise_bias_test "${GGML_MODULE_DIR}" CPU)
    add_executable(llama_pool2d_test pool2d_test.cpp)
    target_link_libraries(llama_pool2d_test PRIVATE 3rdparty_llama)
    add_test(NAME llama_cpu_pool2d
        COMMAND llama_pool2d_test "${GGML_MODULE_DIR}" CPU)
    add_executable(llama_conv_transpose_regression conv_transpose_regression.cpp)
    target_link_libraries(llama_conv_transpose_regression PRIVATE 3rdparty_llama)
    add_test(NAME llama_cpu_conv_transpose_regression
        COMMAND llama_conv_transpose_regression "${GGML_MODULE_DIR}" cpu)
    if(GGML_USE_CUDA OR GGML_USE_VULKAN)
        add_executable(llama_im2col_test im2col_test.cpp)
        target_link_libraries(llama_im2col_test PRIVATE 3rdparty_llama)
        add_executable(llama_conv2d_accumulation_test conv2d_accumulation_test.cpp)
        target_link_libraries(llama_conv2d_accumulation_test PRIVATE 3rdparty_llama)
    endif()
    if(GGML_USE_CUDA)
        add_executable(llama_cuda_numerics_probe cuda_llama_numerics_probe.cpp)
        target_link_libraries(llama_cuda_numerics_probe PRIVATE 3rdparty_llama)
        set(LLAMA_NUMERICS_UPSTREAM_DIR "" CACHE PATH "Pristine pinned llama backend modules")
        set(LLAMA_NUMERICS_BASELINE_DIR "" CACHE PATH "Frozen pre-change backend modules")
        if(LLAMA_NUMERICS_UPSTREAM_DIR AND LLAMA_NUMERICS_BASELINE_DIR)
            add_test(NAME llama_cuda_numerics_scope
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test_cuda_llama_numerics.py
                    --probe $<TARGET_FILE:llama_cuda_numerics_probe>
                    --candidate ${GGML_MODULE_DIR}
                    --upstream ${LLAMA_NUMERICS_UPSTREAM_DIR}
                    --baseline ${LLAMA_NUMERICS_BASELINE_DIR}
                    --report ${CMAKE_BINARY_DIR}/Testing/llama-cuda-numerics-scope.json)
            set_tests_properties(llama_cuda_numerics_scope PROPERTIES RESOURCE_LOCK gpu)
        endif()
        add_test(NAME llama_cuda_pool2d
            COMMAND llama_pool2d_test "${GGML_MODULE_DIR}" CUDA0)
        set_tests_properties(llama_cuda_pool2d PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_cuda_conv2d_accumulation
            COMMAND llama_conv2d_accumulation_test "${GGML_MODULE_DIR}" CUDA0)
        set_tests_properties(llama_cuda_conv2d_accumulation PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_cuda_depthwise_bias
            COMMAND llama_depthwise_bias_test "${GGML_MODULE_DIR}" CUDA0)
        set_tests_properties(llama_cuda_depthwise_bias PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_cuda_elementwise_epilogue
            COMMAND llama_elementwise_epilogue_test "${GGML_MODULE_DIR}" CUDA0)
        set_tests_properties(llama_cuda_elementwise_epilogue PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_cuda_runtime_smoke COMMAND llama_runtime_smoke cuda)
        set_tests_properties(llama_cuda_runtime_smoke PROPERTIES RESOURCE_LOCK gpu)
        add_executable(llama_cuda_matmul_stride_test cuda_matmul_stride_test.cpp)
        target_link_libraries(llama_cuda_matmul_stride_test PRIVATE 3rdparty_llama)
        add_test(NAME llama_cuda_matmul_stride
            COMMAND llama_cuda_matmul_stride_test "${GGML_MODULE_DIR}")
        set_tests_properties(llama_cuda_matmul_stride PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_cuda_im2col
            COMMAND llama_im2col_test "${GGML_MODULE_DIR}" CUDA0)
        set_tests_properties(llama_cuda_im2col PROPERTIES RESOURCE_LOCK gpu)
    endif()
    if(GGML_USE_VULKAN)
        add_test(NAME llama_vulkan_conv2d_accumulation
            COMMAND llama_conv2d_accumulation_test "${GGML_MODULE_DIR}" Vulkan0)
        set_tests_properties(llama_vulkan_conv2d_accumulation PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_vulkan_pool2d
            COMMAND llama_pool2d_test "${GGML_MODULE_DIR}" Vulkan0)
        set_tests_properties(llama_vulkan_pool2d PROPERTIES RESOURCE_LOCK gpu)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
           CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
            # 真正的 checker clone 数值回归；仅覆盖 Linux x86 host/F32 路径。
            add_test(NAME llama_vulkan_dw_debug_clone
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_LIST_DIR}/test_vulkan_dw_debug_clone.py
                    --build ${CMAKE_BINARY_DIR}
                    --output ${CMAKE_BINARY_DIR}/Testing/dw-debug-clone)
            set_tests_properties(llama_vulkan_dw_debug_clone PROPERTIES
                LABELS "ggml;contract")
        endif()
        add_test(NAME llama_vulkan_depthwise_bias
            COMMAND llama_depthwise_bias_test "${GGML_MODULE_DIR}" Vulkan0)
        set_tests_properties(llama_vulkan_depthwise_bias PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_vulkan_elementwise_epilogue
            COMMAND llama_elementwise_epilogue_test "${GGML_MODULE_DIR}" Vulkan0)
        set_tests_properties(llama_vulkan_elementwise_epilogue PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_vulkan_runtime_smoke COMMAND llama_runtime_smoke vulkan)
        set_tests_properties(llama_vulkan_runtime_smoke PROPERTIES RESOURCE_LOCK gpu)
        add_test(NAME llama_vulkan_im2col
            COMMAND llama_im2col_test "${GGML_MODULE_DIR}" Vulkan0)
        set_tests_properties(llama_vulkan_im2col PROPERTIES RESOURCE_LOCK gpu)
    endif()
    if(GGML_USE_METAL)
        add_test(NAME llama_metal_runtime_smoke COMMAND llama_runtime_smoke metal)
        add_test(NAME llama_metal_conv_transpose_regression
            COMMAND llama_conv_transpose_regression "${GGML_MODULE_DIR}" metal)
        set_tests_properties(llama_metal_runtime_smoke
            llama_metal_conv_transpose_regression PROPERTIES RESOURCE_LOCK gpu)
    endif()
    add_test(NAME llama_source_owner_contract
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test_cmake_contract.py)
endif()
