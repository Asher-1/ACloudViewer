# AICore unit tests — mirrors src/ layout: common/, depth/, gaussian/
#
# Tiers:
#   fast   — no GGUF assets (default ctest -LE model)
#   model  — requires AICORE_TEST_DEPTH_GGUF / AICORE_TEST_GAUSSIAN_GGUF
#   capi   — links libAICore.so (exported C API contract)

set(AICORE_TEST_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin/aicore_tests)
set(AICORE_TEST_INC ${CMAKE_CURRENT_SOURCE_DIR})
file(MAKE_DIRECTORY ${AICORE_TEST_RUNTIME_DIR})
string(REPLACE ";" "|" AICORE_TEST_GGML_BACKEND_FILES_PIPE
       "${GGML_BACKEND_MODULE_FILES}")

function(_aicore_copy_ggml_for_test name)
    if(GGML_DYNAMIC_BACKENDS AND GGML_RUNTIME_LIB_DIR AND GGML_BACKEND_DIR)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DGGML_BACKEND_DIR="${GGML_RUNTIME_LIB_DIR}"
                -DDEST_DIR="$<TARGET_FILE_DIR:${name}>"
                -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
                -DLIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CopyGgmlBackends.cmake"
            COMMENT "Copying ggml shared libs for ${name}"
        )
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DGGML_BACKEND_DIR="${GGML_BACKEND_DIR}"
                -DDEST_DIR="$<TARGET_FILE_DIR:${name}>"
                -DLIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
                -DLIB_SUFFIX="${GGML_MODULE_SUFFIX}"
                -DREQUIRED_FILE="${CMAKE_SHARED_LIBRARY_PREFIX}ggml-cpu${GGML_MODULE_SUFFIX}"
                -DREQUESTED_FILES_PIPE="${AICORE_TEST_GGML_BACKEND_FILES_PIPE}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CopyGgmlBackends.cmake"
            COMMENT "Copying ggml backend modules for ${name}"
        )
    endif()
endfunction()

function(_aicore_test_common name)
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(${name} PRIVATE ${AICORE_TEST_INC})
    if(GGML_DYNAMIC_BACKENDS)
        target_compile_definitions(${name} PRIVATE GGML_BACKEND_DL)
    endif()
    set(_runtime_dir "${AICORE_TEST_RUNTIME_DIR}/${name}_runtime")
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_runtime_dir}"
        POSITION_INDEPENDENT_CODE ON
    )
    _aicore_copy_ggml_for_test(${name})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

function(aicore_add_common_test name)
    add_executable(${name} common/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_depth_test name)
    add_executable(${name} depth/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_depth_capi_test name)
    add_executable(${name} depth/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_gaussian_test name)
    add_executable(${name} gaussian/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_lightglue_capi_test name)
    add_executable(${name} lightglue/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_gaussian_capi_test name)
    add_executable(${name} gaussian/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

if(TARGET AICore_test)
    # White-box tests compile private AICore implementation sources.
    aicore_add_common_test(test_data_root)
    aicore_add_depth_test(test_gguf_keys)
    aicore_add_depth_test(test_path_util)
    aicore_add_depth_test(test_compute_mode)

    aicore_add_gaussian_test(test_loader)
    aicore_add_gaussian_test(test_graph_blocks)
    aicore_add_gaussian_test(test_image)
    aicore_add_gaussian_test(test_pose)
    aicore_add_gaussian_test(test_gaussian_path_util)
    aicore_add_gaussian_test(test_parity)
    set_tests_properties(test_parity PROPERTIES LABELS "model")
endif()

# --- depth (C API / libAICore) ---
aicore_add_depth_capi_test(test_depth_capi_contract)
aicore_add_depth_capi_test(test_image_depth)
aicore_add_depth_capi_test(test_depth_capi_load)
set_tests_properties(test_depth_capi_load PROPERTIES LABELS "capi;model")

# --- gaussian (C API / libAICore) ---
aicore_add_gaussian_capi_test(test_gaussian_capi_contract)
aicore_add_gaussian_capi_test(test_gaussian_capi_load)
set_tests_properties(test_gaussian_capi_load PROPERTIES LABELS "capi;model")

aicore_add_lightglue_capi_test(test_lightglue_capi_contract)

add_test(
    NAME test_aicore_public_headers
    COMMAND ${CMAKE_COMMAND}
            -DAICORE_PUBLIC_INCLUDE_DIR=${AICORE_INC_DIR}/aicore
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CheckPublicHeaders.cmake
)
set_tests_properties(test_aicore_public_headers PROPERTIES LABELS "capi")

# --- ABI hygiene: no da_capi_* / fs_capi_* in exported symbols ---
if(UNIX AND NOT APPLE)
    add_test(
        NAME test_no_legacy_symbols
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_no_legacy_symbols.sh
                $<TARGET_FILE:AICore> ${AICORE_INC_DIR}/aicore
    )
    set_tests_properties(test_no_legacy_symbols PROPERTIES LABELS "capi")
endif()

set(_aicore_contract_targets
    test_depth_capi_contract
    test_image_depth
    test_depth_capi_load
    test_gaussian_capi_contract
    test_gaussian_capi_load
    test_lightglue_capi_contract)
add_custom_target(aicore-contract-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L capi
            -C $<CONFIG>
    DEPENDS ${_aicore_contract_targets}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore public contract tests"
)
