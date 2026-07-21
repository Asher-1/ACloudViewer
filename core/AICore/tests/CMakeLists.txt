# AICore unit tests — mirrors src/ layout: common/, depth/, gaussian/
#
# Tiers:
#   fast   — no GGUF assets (default ctest -LE model)
#   model  — requires AICORE_TEST_DEPTH_GGUF / AICORE_TEST_GAUSSIAN_GGUF
#   capi   — links libAICore.so (exported C API contract)

set(AICORE_TEST_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin/aicore_tests)
set(AICORE_TEST_INC ${CMAKE_CURRENT_SOURCE_DIR})
file(MAKE_DIRECTORY ${AICORE_TEST_RUNTIME_DIR})

function(_aicore_test_common name)
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(${name} PRIVATE ${AICORE_TEST_INC})
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${AICORE_TEST_RUNTIME_DIR}
        POSITION_INDEPENDENT_CODE ON
    )
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

# --- common ---
aicore_add_common_test(test_data_root)

# --- depth (fast) ---
aicore_add_depth_test(test_gguf_keys)
aicore_add_depth_test(test_path_util)
aicore_add_depth_test(test_compute_mode)

# --- depth (C API / libAICore) ---
aicore_add_depth_capi_test(test_depth_capi_contract)
aicore_add_depth_capi_test(test_image_depth)
aicore_add_depth_capi_test(test_depth_capi_load)
set_tests_properties(test_depth_capi_load PROPERTIES LABELS "capi;model")

# --- gaussian (fast) ---
aicore_add_gaussian_test(test_loader)
aicore_add_gaussian_test(test_graph_blocks)
aicore_add_gaussian_test(test_image)
aicore_add_gaussian_test(test_pose)
aicore_add_gaussian_test(test_gaussian_path_util)

# --- gaussian (C API / libAICore) ---
aicore_add_gaussian_capi_test(test_gaussian_capi_contract)
aicore_add_gaussian_capi_test(test_gaussian_capi_load)
set_tests_properties(test_gaussian_capi_load PROPERTIES LABELS "capi;model")

# --- gaussian (model parity) ---
aicore_add_gaussian_test(test_parity)
set_tests_properties(test_parity PROPERTIES LABELS "model")

# --- ABI hygiene: no da_capi_* / fs_capi_* in exported symbols ---
add_test(
    NAME test_no_legacy_symbols
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_no_legacy_symbols.sh $<TARGET_FILE:AICore>
)
set_tests_properties(test_no_legacy_symbols PROPERTIES LABELS "capi")
