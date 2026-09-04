# AICore unit tests — mirrors src/ layout: common/, depth/, gaussian/
#
# Tiers:
#   fast   — no GGUF assets (default ctest -LE model)
#   model  — requires AICORE_TEST_DEPTH_GGUF / AICORE_TEST_GAUSSIAN_GGUF
#   capi   — links libAICore.so (exported C API contract)

set(AICORE_TEST_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin/aicore_tests)
set(AICORE_TEST_INC ${CMAKE_CURRENT_SOURCE_DIR})
file(MAKE_DIRECTORY ${AICORE_TEST_RUNTIME_DIR})

# The application and tests use the same platform-native data root. Explicit
# CMake input and CLOUDVIEWER_AICORE_TEST_ASSETS remain higher-priority escape
# hatches for self-hosted runners and readonly installations.
if(DEFINED ENV{CLOUDVIEWER_AICORE_TEST_ASSETS})
    set(_aicore_default_asset_root "$ENV{CLOUDVIEWER_AICORE_TEST_ASSETS}")
elseif(DEFINED ENV{CLOUDVIEWER_DATA_ROOT})
    set(_aicore_default_asset_root "$ENV{CLOUDVIEWER_DATA_ROOT}/extract")
elseif(WIN32 AND DEFINED ENV{USERPROFILE})
    set(_aicore_default_asset_root "$ENV{USERPROFILE}/cloudViewer_data/extract")
elseif(DEFINED ENV{HOME})
    set(_aicore_default_asset_root "$ENV{HOME}/cloudViewer_data/extract")
else()
    set(_aicore_default_asset_root "${CMAKE_BINARY_DIR}/aicore_test_assets")
endif()
set(AICORE_TEST_ASSET_ROOT "${_aicore_default_asset_root}" CACHE PATH
    "Root of the extracted AICore test-model bundle")
set(AICORE_TEST_DEVICE "cpu" CACHE STRING
    "Concrete device used by model CTests (cpu, cuda, vulkan, metal)")
set(AICORE_ALIKED_PARITY_DEVICE "all" CACHE STRING
    "GPU selector for ALIKED parity CTests (all, cuda, vulkan, metal)")
# Hosted CI has neither a durable model cache nor generally a GPU. Keep its
# normal contract job small and deterministic; self-hosted GPU jobs opt in.
if(DEFINED ENV{CI})
    set(_aicore_model_tests_default OFF)
else()
    set(_aicore_model_tests_default ON)
endif()
# Model assets are large. Only an explicit test invocation may opt into a
# network fetch; configuring or building ACloudViewer never does it.
set(_aicore_auto_download_default OFF)
option(AICORE_TEST_ENABLE_MODEL_TESTS
       "Include model/GPU/E2E tests in the AICore test target"
       ${_aicore_model_tests_default})
option(AICORE_TEST_AUTO_DOWNLOAD
       "Download missing assets when an asset-dependent CTest is executed"
       ${_aicore_auto_download_default})

function(_aicore_capi_link name)
    if(TARGET AICore_capi_link)
        target_link_libraries(${name} PRIVATE AICore_capi_link)
    endif()
endfunction()

function(_aicore_test_common name)
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(${name} PRIVATE ${AICORE_TEST_INC})
    target_include_directories(${name} PRIVATE ${AICORE_TEST_INC}/..)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AICoreCompileDefinitions.cmake)
    aicore_apply_compile_definitions(${name})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_test(NAME ${name} COMMAND ${name})
    cloudviewer_set_aicore_test_runtime_layout(${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

function(aicore_add_common_test name)
    add_executable(${name} common/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_common_capi_test name)
    add_executable(${name} common/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore Threads::Threads)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_depth_test name)
    add_executable(${name} depth/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_depth_capi_test name)
    add_executable(${name} depth/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_depth_whitebox_test name)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(${name} depth/whitebox/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/depth/whitebox)
    _aicore_test_common(${name})
    set(_labels "whitebox")
    if(ARGC GREATER 1 AND "${ARGV1}" STREQUAL "MODEL")
        list(APPEND _labels "model")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "${_labels}")
endfunction()

function(aicore_add_gaussian_test name)
    add_executable(${name} gaussian/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_lightglue_capi_test name)
    add_executable(${name} lightglue/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_loma_capi_test name)
    add_executable(${name} loma/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_loma_whitebox_test name)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(${name} loma/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
    set_tests_properties(${name} PROPERTIES LABELS "whitebox;model;reference;accuracy")
endfunction()

function(aicore_add_deeplsd_capi_test name)
    add_executable(${name} deeplsd/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_facedetect_whitebox_test name)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(${name} facedetect/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/facedetect)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_facedetect_capi_test name)
    add_executable(${name} facedetect/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_rfdetr_capi_test name)
    add_executable(${name} rfdetr/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_rfdetr_whitebox_test name)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(${name} rfdetr/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    # NOTE: must be the real src/tasks/rfdetr directory and must come before
    # the include dirs propagated by AICore_test (which list depth/, facedetect/,
    # gaussian/ first): image_io.hpp, model_loader.hpp and backend.hpp exist in
    # several task directories and would otherwise resolve to the wrong task.
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/rfdetr)
    _aicore_test_common(${name})
    set_tests_properties(${name} PROPERTIES LABELS "whitebox")
endfunction()

# Manual perf probe (NOT a ctest — needs a local GGUF, run by hand):
#   bench_rfdetr_perf <model.gguf> [device] [threads] [srcW] [srcH] [iters]
# Same whitebox linking rules as the tests above, minus add_test().
function(aicore_add_rfdetr_perf_bench)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(bench_rfdetr_perf rfdetr/bench_rfdetr_perf.cpp)
    target_link_libraries(bench_rfdetr_perf PRIVATE AICore_test)
    target_include_directories(bench_rfdetr_perf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/rfdetr)
    if(NOT MSVC)
        target_compile_options(bench_rfdetr_perf PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(bench_rfdetr_perf PRIVATE ${AICORE_TEST_INC})
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AICoreCompileDefinitions.cmake)
    aicore_apply_compile_definitions(bench_rfdetr_perf)
    set_target_properties(bench_rfdetr_perf PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cloudviewer_set_aicore_test_runtime_layout(bench_rfdetr_perf)
endfunction()

function(aicore_add_rmbg_capi_test name)
    add_executable(${name} rmbg/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_sam3_capi_test name)
    add_executable(${name} sam3/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_trellis_capi_test name)
    add_executable(${name} trellis/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

function(aicore_add_aliked_capi_test name)
    add_executable(${name} aliked/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    _aicore_capi_link(${name})
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
    _aicore_capi_link(${name})
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
    aicore_add_common_test(test_backend_registry)
    aicore_add_common_test(test_image_view_contract)
    aicore_add_depth_test(test_gguf_keys)
    aicore_add_depth_test(test_path_util)
    aicore_add_depth_test(test_compute_mode)

    # depth whitebox (migrated from qDA3/tests)
    set(_aicore_depth_whitebox_fast
        test_backend
        test_linalg
        test_ggml_extend
        test_winograd
        test_reconstruct
    )
    foreach(_t IN LISTS _aicore_depth_whitebox_fast)
        aicore_add_depth_whitebox_test(${_t})
    endforeach()

    set(_aicore_depth_whitebox_model
        test_dpt_blocks
        test_rope2d
        test_uv_posembed
        test_quantize
        test_quantize_accuracy
        test_model_loader
        test_cam_pose
        test_preprocess
        test_backbone
        test_backbone_prepare
        test_backbone_mv
        test_backbone_mv4
        test_backbone_metric
        test_backbone_giant
        test_backbone_da2
        test_dpt_head
        test_engine_depth
        test_engine_depth_native
        test_engine_pose
        test_engine_mono
        test_engine_mv
        test_engine_metric
        test_engine_da2
        test_fused_depth
        test_gs_head
        test_gs_adapter
        test_giant_depth_pose
        test_head_rays
        test_metric_head
        test_nested_align
        test_preprocess_real
        test_ray_pose
    )
    foreach(_t IN LISTS _aicore_depth_whitebox_model)
        aicore_add_depth_whitebox_test(${_t} MODEL)
    endforeach()

    aicore_add_gaussian_test(test_loader)
    aicore_add_gaussian_test(test_graph_blocks)
    aicore_add_gaussian_test(test_image)
    aicore_add_gaussian_test(test_pose)
    aicore_add_gaussian_test(test_gaussian_path_util)
    aicore_add_gaussian_test(test_parity)
    set_tests_properties(test_parity PROPERTIES LABELS "model")
endif()

# --- depth (C API / libAICore) ---
aicore_add_common_capi_test(test_runtime_capi_contract)
aicore_add_common_capi_test(test_pipeline_timing_contract)
aicore_add_depth_capi_test(test_depth_capi_contract)
aicore_add_depth_capi_test(test_image_depth)
aicore_add_depth_capi_test(test_depth_capi_load)
set_tests_properties(test_depth_capi_load PROPERTIES LABELS "capi;model")
aicore_add_depth_capi_test(test_depth_capi_backend_parity)
set_tests_properties(test_depth_capi_backend_parity
    PROPERTIES LABELS "capi;model;gpu;parity")

# --- gaussian (C API / libAICore) ---
aicore_add_gaussian_capi_test(test_gaussian_capi_contract)
aicore_add_gaussian_capi_test(test_gaussian_capi_load)
set_tests_properties(test_gaussian_capi_load PROPERTIES LABELS "capi;model")

aicore_add_lightglue_capi_test(test_lightglue_capi_contract)
aicore_add_lightglue_capi_test(test_lightglue_aliked_e2e)
aicore_add_lightglue_capi_test(test_lightglue_model_validation)
set_tests_properties(test_lightglue_aliked_e2e
    PROPERTIES LABELS "capi;model;gpu;e2e")
set_tests_properties(test_lightglue_model_validation
    PROPERTIES LABELS "capi;model;gpu;validation")

aicore_add_loma_capi_test(test_loma_capi_contract)
aicore_add_loma_capi_test(test_loma_default_triplet)
set_tests_properties(test_loma_default_triplet PROPERTIES LABELS "capi;model;e2e")
aicore_add_loma_capi_test(test_loma_matcher_e2e)
set_tests_properties(test_loma_matcher_e2e PROPERTIES LABELS "capi;model;e2e")
aicore_add_loma_capi_test(test_loma_published_model_gate)
set_tests_properties(test_loma_published_model_gate
    PROPERTIES LABELS "capi;model;validation")
aicore_add_loma_capi_test(test_loma_detector_reference_gate)
set_tests_properties(test_loma_detector_reference_gate
    PROPERTIES LABELS "capi;model;reference;accuracy")
aicore_add_loma_capi_test(test_loma_descriptor_reference_gate)
set_tests_properties(test_loma_descriptor_reference_gate
    PROPERTIES LABELS "capi;model;reference;accuracy")
aicore_add_loma_whitebox_test(test_loma_g_token_reference_gate)
aicore_add_loma_whitebox_test(test_loma_g_backend_parity)
aicore_add_loma_capi_test(test_loma_b128_e2e_reference_gate)
set_tests_properties(test_loma_b128_e2e_reference_gate
    PROPERTIES LABELS "capi;model;reference;accuracy;e2e")
aicore_add_loma_capi_test(test_loma_matcher_reference_gate)
set_tests_properties(test_loma_matcher_reference_gate
    PROPERTIES LABELS "capi;model;validation")

aicore_add_aliked_capi_test(test_aliked_capi_contract)
aicore_add_aliked_capi_test(test_aliked_capi_parity)
set_tests_properties(test_aliked_capi_parity PROPERTIES LABELS "capi;model;gpu")
aicore_add_aliked_capi_test(test_aliked_e2e_smoke)
set_tests_properties(test_aliked_e2e_smoke PROPERTIES LABELS "capi;model;gpu;smoke")

aicore_add_deeplsd_capi_test(test_deeplsd_capi_contract)
aicore_add_deeplsd_capi_test(test_deeplsd_capi_load)
set_tests_properties(test_deeplsd_capi_load PROPERTIES LABELS "capi;model")

if(TARGET AICore_test)
    aicore_add_facedetect_whitebox_test(test_facedetect_align)
endif()
aicore_add_facedetect_capi_test(test_facedetect_embed_capi)
set_tests_properties(test_facedetect_embed_capi PROPERTIES LABELS "capi;model")
aicore_add_facedetect_capi_test(test_facedetect_dense_validation)
set_tests_properties(test_facedetect_dense_validation
    PROPERTIES LABELS "capi;model;validation")
aicore_add_facedetect_capi_test(test_facedetect_match_capi)
aicore_add_facedetect_capi_test(test_facedetect_contexts_capi)
set_tests_properties(test_facedetect_contexts_capi PROPERTIES LABELS "capi;model")

# FaceDetect C API contract test (no GGUF assets: ABI, catalog, lifecycle,
# null guards — see tests/facedetect/test_facedetect_capi_contract.cpp).
aicore_add_facedetect_capi_test(test_facedetect_capi_contract)
set_tests_properties(test_facedetect_capi_contract PROPERTIES LABELS "capi")

aicore_add_rfdetr_whitebox_test(test_rfdetr_preprocess)
aicore_add_rfdetr_whitebox_test(test_rfdetr_two_stage_mask)
aicore_add_rfdetr_perf_bench()
aicore_add_rfdetr_capi_test(test_rfdetr_capi_contract)
aicore_add_rmbg_capi_test(test_rmbg_capi_contract)
aicore_add_rmbg_capi_test(test_rmbg_capi_performance)
aicore_add_sam3_capi_test(test_sam3_capi_contract)

# Cross-catalog default-row contract (see tests/catalog/ header): locks the
# aicore_<task>_model_default_index() declarations and the
# "first row of every family is the recommended pack" catalogs to the
# visible "(recommended)" marker. Fast, no GGUF assets.
add_executable(test_model_catalog_defaults catalog/test_model_catalog_defaults.cpp)
target_link_libraries(test_model_catalog_defaults PRIVATE AICore)
_aicore_capi_link(test_model_catalog_defaults)
_aicore_test_common(test_model_catalog_defaults)
if(NOT MSVC)
    target_compile_options(test_model_catalog_defaults PRIVATE -fPIC)
    target_link_options(test_model_catalog_defaults PRIVATE "-no-pie")
endif()
set_tests_properties(test_model_catalog_defaults PROPERTIES LABELS "capi")

# Manual real-model acceptance probe for a ggml-version worktree. It is not a
# CTest because it requires a local SAM3 GGUF, a Vulkan-capable GPU, and takes
# several minutes on CPU. See the source header for its exact parity gates.
add_executable(bench_sam3_backend_acceptance sam3/bench_sam3_backend_acceptance.cpp)
target_link_libraries(bench_sam3_backend_acceptance PRIVATE AICore)
_aicore_capi_link(bench_sam3_backend_acceptance)
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(bench_sam3_backend_acceptance PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(bench_sam3_backend_acceptance PRIVATE Qt5::Core Qt5::Gui)
endif()
if(NOT MSVC)
    target_compile_options(bench_sam3_backend_acceptance PRIVATE -Wall -Wextra -fPIC)
    target_link_options(bench_sam3_backend_acceptance PRIVATE "-no-pie")
endif()
target_include_directories(bench_sam3_backend_acceptance PRIVATE ${AICORE_TEST_INC})
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AICoreCompileDefinitions.cmake)
aicore_apply_compile_definitions(bench_sam3_backend_acceptance)
cloudviewer_set_aicore_test_runtime_layout(bench_sam3_backend_acceptance)

# Per-stage CPU/Vulkan divergence locator for the SAM3 ViT encoder. Manual
# diagnostic (needs a local SAM3 GGUF + Vulkan GPU), not a CTest. The internal
# implementation is deliberately compiled into this test-only static library;
# no SAM3 C++ symbol is part of libAICore's production ABI.
# Define SAM3_BUILD but never SAM3_SHARED here: SAM3_API must expand to
# nothing. libAICore intentionally does not export the SAM3 C++ symbols, and
# on MSVC a dllimport-consuming target cannot resolve __imp_?sam3_* against a
# static library (archive members are selected by exact symbol name, so the
# linker never pulls in the member defining the undecorated ?sam3_* symbol).
add_library(AICore_sam3_diagnostics STATIC ${AICORE_SAM3_SRC_DIR}/sam3.cpp)
target_include_directories(AICore_sam3_diagnostics PRIVATE
    ${AICORE_TEST_INC} ${AICORE_TEST_INC}/.. ${AICORE_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${AICORE_COMMON_SRC_DIR}
    ${AICORE_TASKS_SRC_DIR} ${AICORE_SAM3_SRC_DIR} ${GGML_INCLUDE_DIRS})
target_compile_definitions(AICore_sam3_diagnostics PRIVATE SAM3_BUILD)
if(NOT MSVC)
    target_compile_options(AICore_sam3_diagnostics PRIVATE -fPIC)
endif()
aicore_apply_compile_definitions(AICore_sam3_diagnostics)
target_link_libraries(AICore_sam3_diagnostics PRIVATE AICore)

add_executable(cmp_sam3_vit_stages sam3/cmp_sam3_vit_stages.cpp)
target_link_libraries(cmp_sam3_vit_stages PRIVATE AICore_sam3_diagnostics AICore)
# No SAM3_SHARED: the test TU must see SAM3_API as empty so its sam3.h calls
# resolve directly against the AICore_sam3_diagnostics static library instead
# of dllimport (__imp_) references that no import library provides.
add_executable(cmp_sam3_mask_triage sam3/cmp_sam3_mask_triage.cpp)
target_link_libraries(cmp_sam3_mask_triage PRIVATE AICore_sam3_diagnostics AICore)
set_target_properties(cmp_sam3_mask_triage PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(cmp_sam3_vit_stages PRIVATE -Wall -Wextra -fPIC)
    target_link_options(cmp_sam3_vit_stages PRIVATE "-no-pie")
    target_compile_options(cmp_sam3_mask_triage PRIVATE -Wall -Wextra -fPIC)
    target_link_options(cmp_sam3_mask_triage PRIVATE "-no-pie")
endif()
target_include_directories(cmp_sam3_vit_stages PRIVATE ${AICORE_TEST_INC})
target_include_directories(cmp_sam3_vit_stages PRIVATE ${AICORE_TEST_INC}/../src)
aicore_apply_compile_definitions(cmp_sam3_vit_stages)
cloudviewer_set_aicore_test_runtime_layout(cmp_sam3_vit_stages)
aicore_add_trellis_capi_test(test_trellis_capi_contract)
set_tests_properties(test_rmbg_capi_performance PROPERTIES
    LABELS "capi;model;gpu;performance")

# YOLO C API performance benchmark (upstream ultralytics-ggml bench matrix
# counterpart; see tests/yolo/test_yolo_capi_performance.cpp header for the
# timing contract). Skips (77) without AICORE_TEST_YOLO_* assets.
function(aicore_add_yolo_capi_test name)
    add_executable(${name} yolo/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    if(WIN32)
        # dirent.h is POSIX-only; 3rdparty/dirent provides the MSVC shim
        # (test_yolo_capi_performance.cpp includes <dirent/dirent.h> there).
        target_link_libraries(${name} PRIVATE 3rdparty_dirent)
    endif()
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "capi")
endfunction()

aicore_add_yolo_capi_test(test_yolo_capi_performance)
set_tests_properties(test_yolo_capi_performance PROPERTIES
    LABELS "capi;model;gpu;performance")
aicore_add_yolo_capi_test(test_yolo_text_model_validation)
set_tests_properties(test_yolo_text_model_validation PROPERTIES
    LABELS "capi;model;validation;performance")

# TEMPORARY probe for reproducing the yoloe/seg visualization issues
# (delete after diagnosis).
add_executable(yolo_prompt_probe yolo/yolo_prompt_probe.cpp)
target_link_libraries(yolo_prompt_probe PRIVATE AICore)
_aicore_capi_link(yolo_prompt_probe)
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(yolo_prompt_probe PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(yolo_prompt_probe PRIVATE Qt5::Core Qt5::Gui)
endif()
if(NOT MSVC)
    target_compile_options(yolo_prompt_probe PRIVATE -Wall -Wextra -fPIC)
    target_link_options(yolo_prompt_probe PRIVATE "-no-pie")
endif()
target_include_directories(yolo_prompt_probe PRIVATE ${AICORE_TEST_INC})
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AICoreCompileDefinitions.cmake)
aicore_apply_compile_definitions(yolo_prompt_probe)
cloudviewer_set_aicore_test_runtime_layout(yolo_prompt_probe)

# YOLO device-parity test (CPU vs GPU numeric gates on real GGUF assets;
# see tests/yolo/test_yolo_capi_parity.cpp for the gate table). Skips (77)
# without AICORE_TEST_YOLO_* assets or a GPU backend.
aicore_add_yolo_capi_test(test_yolo_capi_parity)
set_tests_properties(test_yolo_capi_parity PROPERTIES
    LABELS "capi;model;gpu")

# Full same-model/same-input matrix: CPU/CUDA/Vulkan latency plus CPU-vs-GPU
# numeric parity for every YOLO task and f32/f16/q8_0. Missing assets skip;
# partial assets or a missing requested backend fail with an explicit report.
if(Python3_EXECUTABLE)
    add_test(NAME test_yolo_model_matrix
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/yolo/run_yolo_model_matrix.py
            --benchmark $<TARGET_FILE:test_yolo_capi_performance>
            --parity $<TARGET_FILE:test_yolo_capi_parity>
            --report ${CMAKE_BINARY_DIR}/Testing/yolo_model_matrix.json)
    set_tests_properties(test_yolo_model_matrix PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "matrix;performance;parity")
    add_custom_target(aicore-yolo-model-matrix
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                -R "^test_yolo_model_matrix$" -C $<CONFIG>
        DEPENDS test_yolo_capi_performance test_yolo_capi_parity
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running YOLO task/quant/backend model matrix"
        VERBATIM)
endif()

# YOLO C API contract test (no GGUF assets: ABI, options, catalog, null
# guards, model-free error paths — see tests/yolo/test_yolo_capi_contract.cpp).
aicore_add_yolo_capi_test(test_yolo_capi_contract)
set_tests_properties(test_yolo_capi_contract PROPERTIES
    LABELS "capi")

# YOLOE visual-prompt (SAVPE) model-tier test (needs a savpe-converted
# yoloe GGUF + prompt image; skips with 77 without
# AICORE_TEST_YOLO_SAVPE_* assets — see test_yolo_capi_savpe.cpp).
aicore_add_yolo_capi_test(test_yolo_capi_savpe)
set_tests_properties(test_yolo_capi_savpe PROPERTIES
    LABELS "capi;model")

# YOLO geometry whitebox test (no GGUF assets: mask unscale / letterbox
# inverse mapping — see tests/yolo/test_yolo_unscale_masks.cpp).
function(aicore_add_yolo_whitebox_test name)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(${name} yolo/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
    set_tests_properties(${name} PROPERTIES LABELS "whitebox")
endfunction()

aicore_add_yolo_whitebox_test(test_yolo_unscale_masks)

# Whitebox op-level bisection tool for world-family device-parity failures
# (model/env-gated; see tests/yolo/test_yolo_world_optrace.cpp).
function(aicore_add_yolo_optrace_test name)
    if(NOT TARGET AICore_test)
        return()
    endif()
    add_executable(${name} yolo/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    # Qt was built with -reduce-relocations → compile with -fPIC (same
    # pattern as aicore_add_yolo_capi_test; -fPIE is not enough).
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    _aicore_test_common(${name})
    set_tests_properties(${name} PROPERTIES LABELS "whitebox;gpu")
endfunction()
aicore_add_yolo_optrace_test(test_yolo_world_optrace)

add_test(
    NAME test_aicore_public_headers
    COMMAND ${CMAKE_COMMAND}
            -DAICORE_PUBLIC_INCLUDE_DIR=${AICORE_INC_DIR}/aicore
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CheckPublicHeaders.cmake
)
set_tests_properties(test_aicore_public_headers PROPERTIES LABELS "capi")

function(aicore_configure_model_test_assets)
    if(NOT AICORE_TEST_ASSET_ROOT)
        return()
    endif()
    set(_root "${AICORE_TEST_ASSET_ROOT}")
    set(_env
        "AICORE_TEST_DEVICE=${AICORE_TEST_DEVICE}"
        "AICORE_TEST_DEPTH_GGUF=${_root}/da3_models/depth-anything-base-q4_k.gguf"
        "AICORE_TEST_DEPTH_IMAGE=${_root}/lightglue_test_images/sacre_coeur1.jpg"
        "AICORE_TEST_GAUSSIAN_GGUF=${_root}/freesplatter_models/freesplatter-scene-q8_0.gguf"
        "AICORE_TEST_GAUSSIAN_IMAGE_0=${_root}/lightglue_test_images/sacre_coeur1.jpg"
        "AICORE_TEST_GAUSSIAN_IMAGE_1=${_root}/lightglue_test_images/sacre_coeur2.jpg"
        # qLightGlue runs the ALIKED extractor and LightGlue matcher at the
        # SAME quantization the user selected, so the q8_0 matcher pairs with
        # the q8_0 extractor (the current GUI end-to-end baseline).
        "AICORE_TEST_ALIKED_GGUF=${_root}/lightglue_models/aliked-n16rot-q8_0.gguf"
        "AICORE_TEST_ALIKED_IMAGE=${_root}/lightglue_test_images/sacre_coeur1.jpg"
        "AICORE_TEST_LIGHTGLUE_GGUF=${_root}/lightglue_models/aliked-lightglue-q8_0.gguf"
        "AICORE_TEST_LIGHTGLUE_IMAGE0=${_root}/lightglue_test_images/sacre_coeur1.jpg"
        "AICORE_TEST_LIGHTGLUE_IMAGE1=${_root}/lightglue_test_images/sacre_coeur2.jpg"
        "AICORE_TEST_DEEPLSD_GGUF=${_root}/deeplsd_models/deeplsd_md-q8_0.gguf"
        "AICORE_TEST_DEEPLSD_IMAGE=${_root}/lightglue_test_images/sacre_coeur1.jpg"
        "AICORE_TEST_FACEDETECT_GGUF=${_root}/facedetect_models/buffalo_l.gguf"
        # friends1 is the fixture with stable 5-point landmarks. A cropped
        # single-face image is not suitable for this alignment contract.
        "AICORE_TEST_FACEDETECT_IMAGE=${_root}/friends_faces/query/friends1.jpg"
        "AICORE_TEST_FACEDETECT_DEVICE=${AICORE_TEST_DEVICE}"
        "AICORE_TEST_RFDETR_GGUF=${_root}/rfdetr_models/rfdetr-base-f16.gguf"
        "AICORE_TEST_RMBG_GGUF=${_root}/rmbg_models/rmbg_f16.gguf")
    if(AICORE_TEST_AUTO_DOWNLOAD)
        # This fixture is only selected by asset-dependent CTests. It is not a
        # build dependency, so compiling ACloudViewer cannot trigger I/O.
        add_test(
            NAME aicore_fetch_test_assets
            COMMAND ${CMAKE_COMMAND}
                    -DAICORE_TEST_ASSET_ROOT=${AICORE_TEST_ASSET_ROOT}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchAICoreTestAssets.cmake)
        set_tests_properties(aicore_fetch_test_assets PROPERTIES
            FIXTURES_SETUP aicore_model_assets
            LABELS "asset")
    endif()
    foreach(_test
            test_depth_capi_load
            test_depth_capi_backend_parity
            test_gaussian_capi_load
            test_aliked_capi_parity
            test_aliked_e2e_smoke
            test_lightglue_aliked_e2e
            test_deeplsd_capi_load
            test_facedetect_embed_capi
            test_facedetect_contexts_capi
            test_rmbg_capi_performance)
        if(TEST ${_test})
            set_tests_properties(${_test} PROPERTIES ENVIRONMENT "${_env}")
            if(AICORE_TEST_AUTO_DOWNLOAD)
                set_tests_properties(${_test} PROPERTIES
                    FIXTURES_REQUIRED aicore_model_assets)
            endif()
        endif()
    endforeach()

    # Other model C APIs accept one concrete device. ALIKED's parity runner
    # alone understands "all", which means one CPU reference against each
    # available GPU, so keep that selector isolated from the shared test env.
    foreach(_test test_aliked_capi_parity)
        if(TEST ${_test})
            set_tests_properties(${_test} PROPERTIES
                ENVIRONMENT "${_env};AICORE_TEST_DEVICE=${AICORE_ALIKED_PARITY_DEVICE}")
        endif()
    endforeach()

    # Do not let the small q8 extractor hide behind the GUI f16 baseline.
    # It is a separately supported C API model and must meet the same strict
    # CPU/GPU parity gate whenever model tests are explicitly enabled.
    if(TEST test_aliked_capi_parity)
        add_test(
            NAME test_aliked_capi_parity_512
            COMMAND $<TARGET_FILE:test_aliked_capi_parity>
                    ${_root}/lightglue_models/aliked-n16rot-f16.gguf
                    ${_root}/lightglue_test_images/sacre_coeur1.jpg
                    1024 512)
        set_tests_properties(test_aliked_capi_parity_512 PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS "capi;model;gpu;parity;f16"
            ENVIRONMENT "${_env};AICORE_TEST_DEVICE=${AICORE_ALIKED_PARITY_DEVICE}")
        if(AICORE_TEST_AUTO_DOWNLOAD)
            set_tests_properties(test_aliked_capi_parity_512 PROPERTIES
                FIXTURES_REQUIRED aicore_model_assets)
        endif()

        add_test(
            NAME test_aliked_capi_parity_q8
            COMMAND $<TARGET_FILE:test_aliked_capi_parity>
                    ${_root}/lightglue_models/aliked-n16rot-q8_0.gguf
                    ${_root}/lightglue_test_images/sacre_coeur1.jpg
                    1024 1024)
        set_tests_properties(test_aliked_capi_parity_q8 PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS "capi;model;gpu;parity;q8"
            ENVIRONMENT "${_env};AICORE_TEST_DEVICE=${AICORE_ALIKED_PARITY_DEVICE}")
        if(AICORE_TEST_AUTO_DOWNLOAD)
            set_tests_properties(test_aliked_capi_parity_q8 PROPERTIES
                FIXTURES_REQUIRED aicore_model_assets)
        endif()
    endif()
endfunction()

aicore_configure_model_test_assets()

# --- ABI hygiene: no da_capi_* / fs_capi_* in exported symbols ---
if(UNIX AND NOT APPLE)
    add_test(
        NAME test_no_legacy_symbols
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_no_legacy_symbols.sh
                $<TARGET_FILE:AICore> ${AICORE_INC_DIR}/aicore
    )
    set_tests_properties(test_no_legacy_symbols PROPERTIES LABELS "capi")
endif()

# --- Env hygiene: getenv/setenv only in the sanctioned bridge ---
# POSIX-only guard: this is a bash static scan (grep/find/sort). macOS and
# Linux run it fine, but Windows has no .sh launcher (CTest BAD_COMMAND),
# so register it only on non-Windows platforms.
if(NOT WIN32)
    add_test(
        NAME test_no_env_getenv
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_no_env_getenv.sh
                ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    set_tests_properties(test_no_env_getenv PROPERTIES LABELS "capi")
endif()

set(_aicore_contract_targets
    test_runtime_capi_contract
    test_pipeline_timing_contract
    test_depth_capi_contract
    test_image_depth
    test_depth_capi_load
    test_gaussian_capi_contract
    test_gaussian_capi_load
    test_lightglue_capi_contract
    test_lightglue_aliked_e2e
    test_aliked_capi_contract
    test_aliked_capi_parity
    test_aliked_e2e_smoke
    test_deeplsd_capi_contract
    test_facedetect_match_capi
    test_facedetect_capi_contract
    test_rfdetr_capi_contract
    test_yolo_capi_contract
    test_rmbg_capi_contract
    test_rmbg_capi_performance
    test_sam3_capi_contract
    test_trellis_capi_contract
    test_yolo_capi_performance
    test_model_catalog_defaults)
set(_aicore_contract_model_exclude)
if(NOT AICORE_TEST_ENABLE_MODEL_TESTS)
    list(APPEND _aicore_contract_model_exclude -LE "model|gpu|e2e")
endif()
add_custom_target(aicore-contract-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L capi
            ${_aicore_contract_model_exclude}
            -j1
            -C $<CONFIG>
    DEPENDS ${_aicore_contract_targets}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore public contract tests"
    VERBATIM
)

# --- Catalog vs release asset consistency (network; CI only) ---------------
# The catalog can be self-consistent while individual entries 404 on the
# release (historical rmbg_q4_K.gguf regression). Dump every catalog URL and
# HEAD-check it against GitHub so drift fails the build instead of the user's
# first download. Kept out of the "capi" label so offline/contract runs stay
# network-free.
add_executable(test_catalog_dump_urls catalog_dump_urls.cpp)
target_link_libraries(test_catalog_dump_urls PRIVATE AICore)
_aicore_capi_link(test_catalog_dump_urls)
_aicore_test_common(test_catalog_dump_urls)
if(NOT MSVC)
    target_compile_options(test_catalog_dump_urls PRIVATE -fPIC)
    target_link_options(test_catalog_dump_urls PRIVATE "-no-pie")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_EXECUTABLE AND NOT CMAKE_CROSSCOMPILING)
    add_test(
        NAME test_catalog_remote_assets
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_catalog_assets.py
                $<TARGET_FILE:test_catalog_dump_urls>)
    set_tests_properties(test_catalog_remote_assets PROPERTIES
        LABELS "catalog-remote" TIMEOUT 300)
endif()

# --- Fast tier: no GGUF assets (whitebox/common/gaussian units + capi
# contract subset + catalog URL dump). Mirror entry point of
# aicore-contract-tests so CI can address the tier by name. Whitebox targets
# exist only when AICore_BUILD_WHITEBOX_TESTS=ON; keep the DEPENDS list to
# targets present in this configuration, while ctest -LE runs every registered
# fast test. When adding a new fast-tier test target, append it here, otherwise
# ctest fails at runtime with a missing binary.
set(_aicore_fast_targets
    test_data_root
    test_backend_registry
    test_image_view_contract
    test_gguf_keys
    test_path_util
    test_compute_mode
    test_backend
    test_linalg
    test_ggml_extend
    test_winograd
    test_reconstruct
    test_loader
    test_graph_blocks
    test_image
    test_pose
    test_gaussian_path_util
    test_facedetect_align
    test_rfdetr_preprocess
    test_rfdetr_two_stage_mask
    test_yolo_unscale_masks
    test_catalog_dump_urls)
# The fast tier also runs the no-asset subset of the capi contract suite
# (ctest -LE "model|gpu|e2e" selects them), so those binaries must exist too.
list(APPEND _aicore_fast_targets ${_aicore_contract_targets})
set(_aicore_fast_targets_existing)
foreach(_t IN LISTS _aicore_fast_targets)
    if(TARGET ${_t})
        list(APPEND _aicore_fast_targets_existing ${_t})
    endif()
endforeach()
add_custom_target(aicore-fast-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE "model|gpu|e2e|cvpluginapi"
            -j2
            -C $<CONFIG>
    DEPENDS ${_aicore_fast_targets_existing}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore fast-tier tests (no GGUF assets)"
    VERBATIM
)

# --- Model tier: needs GGUF assets (AICORE_TEST_ENABLE_MODEL_TESTS=ON +
# asset root). Tests without assets skip (SKIP_RETURN_CODE 77), so this
# target never fails on missing assets — it only executes them. Mirror of
# the fast/contract entry points; the ctest -L "model" selection also covers
# the two parity variants that reuse the test_aliked_capi_parity binary.
set(_aicore_model_targets
    test_dpt_blocks
    test_rope2d
    test_uv_posembed
    test_quantize
    test_quantize_accuracy
    test_model_loader
    test_cam_pose
    test_preprocess
    test_backbone
    test_backbone_prepare
    test_backbone_mv
    test_backbone_mv4
    test_backbone_metric
    test_backbone_giant
    test_backbone_da2
    test_dpt_head
    test_engine_depth
    test_engine_depth_native
    test_engine_pose
    test_engine_mono
    test_engine_mv
    test_engine_metric
    test_engine_da2
    test_fused_depth
    test_gs_head
    test_gs_adapter
    test_giant_depth_pose
    test_head_rays
    test_metric_head
    test_nested_align
    test_preprocess_real
    test_ray_pose
    test_parity
    test_depth_capi_load
    test_depth_capi_backend_parity
    test_gaussian_capi_load
    test_lightglue_aliked_e2e
    test_lightglue_model_validation
    test_aliked_capi_parity
    test_aliked_e2e_smoke
    test_deeplsd_capi_load
    test_facedetect_embed_capi
    test_facedetect_dense_validation
    test_facedetect_contexts_capi
    test_rmbg_capi_performance
    test_yolo_capi_performance)
set(_aicore_model_targets_existing)
foreach(_t IN LISTS _aicore_model_targets)
    if(TARGET ${_t})
        list(APPEND _aicore_model_targets_existing ${_t})
    endif()
endforeach()
add_custom_target(aicore-model-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "model"
            -j1
            -C $<CONFIG>
    DEPENDS ${_aicore_model_targets_existing}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore model-tier tests (skips without GGUF assets)"
    VERBATIM
)

# Complete real-model/task regression gate. The runner discovers every model,
# executes task-specific probes serially, and rejects missing coverage.
if(Python3_EXECUTABLE AND NOT CMAKE_CROSSCOMPILING)
    set(_aicore_validation_targets
        test_depth_capi_load
        test_gaussian_capi_load
        test_aliked_capi_parity
        test_lightglue_model_validation
        test_deeplsd_capi_load
        test_facedetect_embed_capi
        test_facedetect_dense_validation
        bench_rfdetr_perf
        test_rmbg_capi_performance
        bench_sam3_backend_acceptance
        test_yolo_capi_performance
        test_yolo_text_model_validation
        test_yolo_capi_parity)
    # The validation runner launches C API executables directly.  Propagate the
    # Qt runtime selected at configure time so it cannot accidentally resolve a
    # compatible-but-older system Qt at test time.
    set(_aicore_validation_runtime_env)
    foreach(_aicore_qt_core_target Qt6::Core Qt5::Core)
        if(TARGET ${_aicore_qt_core_target})
            list(APPEND _aicore_validation_runtime_env
                 "AICORE_QT_LIBRARY_DIR=$<TARGET_FILE_DIR:${_aicore_qt_core_target}>")
            break()
        endif()
    endforeach()
    add_custom_target(aicore-validate-all
        COMMAND ${CMAKE_COMMAND} -E env ${_aicore_validation_runtime_env}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_all.py
                --build ${CMAKE_BINARY_DIR}
                --assets ${AICORE_TEST_ASSET_ROOT}
                --backend ${AICORE_TEST_DEVICE}
                --timeout 14400
                --output ${CMAKE_BINARY_DIR}/Testing/aicore_validation.json
        DEPENDS test_catalog_dump_urls ${_aicore_validation_targets}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running AICore model/task regression matrix (light tier: "
                "sam3/trellis lightweight subsets; use validate_all.py --full "
                "for the complete matrix)"
        USES_TERMINAL
        VERBATIM)

    add_test(NAME test_aicore_validation_runner
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_validate_all.py)
    set_tests_properties(test_aicore_validation_runner PROPERTIES
        LABELS "fast;validation")
endif()
