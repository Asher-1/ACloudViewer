# AICore unit tests — mirrors src/ layout: common/, depth/, gaussian/
#
# Tiers (selection is purely LABELS-driven):
#   fast   — no GGUF assets (default ctest -LE model)
#   model  — requires AICORE_TEST_DEPTH_GGUF / AICORE_TEST_GAUSSIAN_GGUF
#   capi   — links libAICore.so (exported C API contract)
# Every add_test below is captured by the registry override; the tier DEPENDS
# lists at the bottom of this file are DERIVED from the registry + LABELS, so
# adding a test never requires touching a tier list.

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
    # Optional extra args:
    #   NO_CTEST — set up the target (defines/includes/runtime layout) but do
    #   NOT register a CTest entry. For manual diagnostic tools that need
    #   side inputs (e.g. reid_diag's CHW binary); registering them would
    #   make the fast tier fail with a Not Run / usage-exit test.
    set(_aicore_no_ctest FALSE)
    foreach(_aicore_arg IN LISTS ARGN)
        if(_aicore_arg STREQUAL "NO_CTEST")
            set(_aicore_no_ctest TRUE)
        endif()
    endforeach()
    if(NOT MSVC)
        # Qt is built with -reduce-relocations: every AICore test binary
        # compiles with -fPIC and links -no-pie (same requirement as the
        # AICore libraries themselves).
        target_compile_options(${name} PRIVATE -Wall -Wextra -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    target_include_directories(${name} PRIVATE ${AICORE_TEST_INC})
    target_include_directories(${name} PRIVATE ${AICORE_TEST_INC}/..)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AICoreCompileDefinitions.cmake)
    aicore_apply_compile_definitions(${name})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cloudviewer_set_aicore_test_runtime_layout(${name})
    if(_aicore_no_ctest)
        return()
    endif()
    add_test(NAME ${name} COMMAND ${name})
    if(AICORE_TEST_ENV_EXTRA)
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77
            ENVIRONMENT "${AICORE_TEST_ENV_EXTRA}")
    else()
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endfunction()

# --- Tier filters (single source of truth) ---------------------------------
# Byte-identical to the ctest label arguments in the tier targets below; the
# derived DEPENDS lists and aicore_register_test route on these. CACHE INTERNAL
# so plugin directories (processed later) see the same values.
set(AICORE_TIER_FAST_EXCLUDE "model|gpu|e2e|cvpluginapi" CACHE INTERNAL
    "Labels that exclude a test from the fast tier")
set(AICORE_TIER_CONTRACT_LABEL "capi" CACHE INTERNAL
    "Label that selects a test into the contract tier")
set(AICORE_TIER_CONTRACT_MODEL_EXCLUDE "model|gpu|e2e" CACHE INTERNAL
    "Contract-tier exclusion applied when model tests are disabled")
set(AICORE_TIER_MODEL_LABEL "model" CACHE INTERNAL
    "Label that selects a test into the model tier")

# --- Test registration registry (self-adapting tier attachment) -------------
# Every add_test() call in this directory is intercepted below and recorded
# into a global registry: the test name plus — when the COMMAND invokes a
# buildable target directly — that target. The derivation block at the bottom
# of this file combines the registry with each test's LABELS property to build
# the tier DEPENDS lists, replicating the exact ctest filters, so a newly
# registered test is attached to the right tier automatically and the
# historical "Could not find executable / Not Run" (ctest exit 8) failure
# mode cannot recur. Outside this directory the override is a passthrough.
set(AICORE_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
function(add_test)
    cmake_parse_arguments(_aicore_at "" "NAME" "COMMAND" ${ARGN})
    if(_aicore_at_NAME
            AND "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${AICORE_TESTS_DIR}")
        set_property(GLOBAL APPEND PROPERTY AICORE_TEST_NAMES
            "${_aicore_at_NAME}")
        set(_aicore_at_tgt "")
        if(_aicore_at_COMMAND)
            list(GET _aicore_at_COMMAND 0 _aicore_at_cmd0)
            if(TARGET "${_aicore_at_cmd0}")
                set(_aicore_at_tgt "${_aicore_at_cmd0}")
            endif()
        endif()
        set_property(GLOBAL PROPERTY "AICORE_TEST_TGT_${_aicore_at_NAME}"
            "${_aicore_at_tgt}")
    endif()
    _add_test(${ARGN})
endfunction()

# External (plugin-directory) registration for tests that belong to an AICore
# tier: plugins are processed after this directory, so their add_test calls
# cannot flow through the registry override. One call registers the ctest
# entry and attaches the binary to the matching tier target — there is no
# cross-directory dependency list to maintain (this replaces the manual
# add_dependencies(aicore-fast-tests ...) contract).
#   aicore_register_test(NAME <test> TARGET <binary> [LABELS <l1;...>])
function(aicore_register_test)
    cmake_parse_arguments(AIRT "" "NAME;TARGET" "LABELS" ${ARGN})
    if(NOT AIRT_NAME OR NOT AIRT_TARGET)
        message(FATAL_ERROR "aicore_register_test requires NAME and TARGET")
    endif()
    if(NOT TARGET ${AIRT_TARGET})
        message(FATAL_ERROR "aicore_register_test: unknown target ${AIRT_TARGET}")
    endif()
    add_test(NAME ${AIRT_NAME} COMMAND ${AIRT_TARGET})
    if(AIRT_LABELS)
        set_tests_properties(${AIRT_NAME} PROPERTIES LABELS "${AIRT_LABELS}")
    endif()
    set(_airt_model FALSE)
    set(_airt_fast FALSE)
    set(_airt_contract FALSE)
    if(AIRT_LABELS)
        if(AIRT_LABELS MATCHES "${AICORE_TIER_MODEL_LABEL}")
            set(_airt_model TRUE)
        endif()
        if(AIRT_LABELS MATCHES "${AICORE_TIER_CONTRACT_LABEL}"
                AND (AICORE_TEST_ENABLE_MODEL_TESTS
                    OR NOT AIRT_LABELS MATCHES
                        "${AICORE_TIER_CONTRACT_MODEL_EXCLUDE}"))
            set(_airt_contract TRUE)
        endif()
        if(NOT AIRT_LABELS MATCHES "${AICORE_TIER_FAST_EXCLUDE}")
            set(_airt_fast TRUE)
        endif()
    else()
        set(_airt_fast TRUE)
    endif()
    if(_airt_fast AND TARGET aicore-fast-tests)
        add_dependencies(aicore-fast-tests ${AIRT_TARGET})
    endif()
    if(_airt_contract AND TARGET aicore-contract-tests)
        add_dependencies(aicore-contract-tests ${AIRT_TARGET})
    endif()
    if(_airt_model AND TARGET aicore-model-tests)
        add_dependencies(aicore-model-tests ${AIRT_TARGET})
    endif()
endfunction()

# Propagate the configure-time Qt runtime into every test environment so the
# dynamic loader cannot mix a compatible-but-older system Qt with the
# selected one (versioned operator-delete interposition fails on a mixed
# stack). Same rationale as the AICORE_QT_LIBRARY_DIR plumbing on the
# aicore-validate-all target. Empty on Windows (no LD_LIBRARY_PATH).
set(AICORE_TEST_ENV_EXTRA "")
# CUDA runtime directory: a dlopen'ed libggml-cuda.so needs the toolkit-
# matching libcudart FIRST on the loader path — ldconfig may expose older
# CUDA 11.x runtimes that lack versioned symbols the backend needs (e.g.
# cudaLaunchKernelExC), and ggml_backend_load_all_from_path is silent under
# NDEBUG, so the CUDA backend would disappear without any error.
# Version-agnostic by design: locate the toolkit lib DIRECTORY, never a
# specific soname, so CUDA 12/13/14+ toolkits all work without edits here.
if(NOT WIN32)
    # Prefer the toolkit the build itself linked against: find_package(
    # CUDAToolkit) in the parent project sets CUDAToolkit_LIBRARY_DIR.
    if(CUDAToolkit_LIBRARY_DIR AND EXISTS "${CUDAToolkit_LIBRARY_DIR}")
        set(_aicore_cuda_lib_dir "${CUDAToolkit_LIBRARY_DIR}")
    else()
        set(_aicore_cuda_hint "$ENV{CUDA_HOME}")
        if(NOT _aicore_cuda_hint)
            set(_aicore_cuda_hint "$ENV{CUDA_PATH}")
        endif()
        if(NOT _aicore_cuda_hint)
            set(_aicore_cuda_hint "/usr/local/cuda")
        endif()
        foreach(_aicore_cuda_dir
                "${_aicore_cuda_hint}/lib64" "${_aicore_cuda_hint}/lib"
                "${_aicore_cuda_hint}")
            # Any libcudart soname proves this is the toolkit runtime dir.
            file(GLOB _aicore_cudart_libs
                "${_aicore_cuda_dir}/libcudart.so*")
            if(_aicore_cudart_libs)
                set(_aicore_cuda_lib_dir "${_aicore_cuda_dir}")
                break()
            endif()
        endforeach()
        unset(_aicore_cuda_hint)
    endif()
endif()
if(NOT MSVC)
    set(_aicore_ld_library_path "")
    if(DEFINED _aicore_cuda_lib_dir)
        set(_aicore_ld_library_path "${_aicore_cuda_lib_dir}:")
    endif()
    foreach(_aicore_qt_core_target Qt6::Core Qt5::Core)
        if(TARGET ${_aicore_qt_core_target})
            string(APPEND _aicore_ld_library_path
                "$<TARGET_FILE_DIR:${_aicore_qt_core_target}>")
            set(AICORE_TEST_ENV_EXTRA
                "LD_LIBRARY_PATH=${_aicore_ld_library_path}")
            break()
        endif()
    endforeach()
    unset(_aicore_ld_library_path)
endif()

# Unit tests against the whitebox library: aicore_add_whitebox_test with the
# task directory. (The former aicore_add_common_test / aicore_add_depth_test /
# aicore_add_gaussian_test helpers were the same shape.)

# Generic C API test binary: <dir>/<name>.cpp linked against libAICore and
# registered as a ctest. One call per test — the label set lives here, not in
# a follow-up set_tests_properties. Keywords:
#   LIBS <l...>   — extra link libraries (e.g. Threads::Threads)
#   LABELS <l...> — replaces the default "capi" label set
function(aicore_add_capi_test name dir)
    cmake_parse_arguments(ACT "" "" "LIBS;LABELS" ${ARGN})
    add_executable(${name} ${dir}/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore)
    foreach(_aicore_lib IN LISTS ACT_LIBS)
        target_link_libraries(${name} PRIVATE ${_aicore_lib})
    endforeach()
    _aicore_capi_link(${name})
    _aicore_test_common(${name})
    if(NOT ACT_LABELS)
        set(ACT_LABELS capi)
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "${ACT_LABELS}")
endfunction()

# C API diagnostic/bench binary (NOT a ctest — manual tools with side-input
# requirements; registering them would put Not Run / usage-exit entries into
# the tiers): same compile setup as aicore_add_capi_test, no registration.
function(aicore_add_capi_tool name source)
    cmake_parse_arguments(AICT "" "" "LIBS;INCLUDE_DIRS" ${ARGN})
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE AICore)
    foreach(_aicore_lib IN LISTS AICT_LIBS)
        target_link_libraries(${name} PRIVATE ${_aicore_lib})
    endforeach()
    _aicore_capi_link(${name})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -Wall -Wextra -fPIC)
        target_link_options(${name} PRIVATE "-no-pie")
    endif()
    if(AICT_INCLUDE_DIRS)
        foreach(_aicore_inc IN LISTS AICT_INCLUDE_DIRS)
            target_include_directories(${name} PRIVATE ${_aicore_inc})
        endforeach()
    else()
        target_include_directories(${name} PRIVATE ${AICORE_TEST_INC})
        target_include_directories(${name} PRIVATE ${AICORE_TEST_INC}/..)
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AICoreCompileDefinitions.cmake)
    aicore_apply_compile_definitions(${name})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    cloudviewer_set_aicore_test_runtime_layout(${name})
endfunction()

# White-box unit tests compile private AICore implementation sources and link
# the static AICore_test library; they are only built with
# AICORE_BUILD_WHITEBOX_TESTS=ON and default to LABELS "whitebox". Keywords:
#   SOURCE <path>       — overrides the default <dir>/<name>.cpp
#   INCLUDE_DIRS <d...> — prepended before AICore_test-propagated dirs
#   LABELS <l...>       — replaces the default "whitebox" label set
function(aicore_add_whitebox_test name dir)
    cmake_parse_arguments(AWT "" "SOURCE" "INCLUDE_DIRS;LABELS" ${ARGN})
    if(NOT TARGET AICore_test)
        return()
    endif()
    if(AWT_SOURCE)
        set(_aicore_src "${AWT_SOURCE}")
    else()
        set(_aicore_src "${dir}/${name}.cpp")
    endif()
    add_executable(${name} ${_aicore_src})
    target_link_libraries(${name} PRIVATE AICore_test)
    foreach(_aicore_inc IN LISTS AWT_INCLUDE_DIRS)
        target_include_directories(${name} PRIVATE ${_aicore_inc})
    endforeach()
    _aicore_test_common(${name})
    if(NOT DEFINED AWT_LABELS)
        set(AWT_LABELS whitebox)
    endif()
    if(AWT_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${AWT_LABELS}")
    endif()
endfunction()

function(aicore_add_depth_whitebox_test name)
    cmake_parse_arguments(DWT "" "" "LABELS" ${ARGN})
    set(_aicore_dwt_args
        SOURCE depth/whitebox/${name}.cpp
        INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/depth/whitebox)
    if(DWT_LABELS)
        list(APPEND _aicore_dwt_args LABELS ${DWT_LABELS})
    endif()
    aicore_add_whitebox_test(${name} depth ${_aicore_dwt_args})
endfunction()

function(aicore_add_gaussian_test name)
    add_executable(${name} gaussian/${name}.cpp)
    target_link_libraries(${name} PRIVATE AICore_test)
    _aicore_test_common(${name})
endfunction()

function(aicore_add_loma_whitebox_test name)
    aicore_add_whitebox_test(${name} loma
        LABELS "whitebox;model;reference;accuracy")
endfunction()

function(aicore_add_facedetect_whitebox_test name)
    aicore_add_whitebox_test(${name} facedetect
        INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src/facedetect
        LABELS "")
endfunction()

function(aicore_add_rfdetr_whitebox_test name)
    aicore_add_whitebox_test(${name} rfdetr
        # NOTE: must be the real src/tasks/rfdetr directory and must come
        # before the include dirs propagated by AICore_test (which list
        # depth/, facedetect/, gaussian/ first): image_io.hpp, model_loader.hpp
        # and backend.hpp exist in several task directories and would
        # otherwise resolve to the wrong task.
        INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/rfdetr
        LABELS whitebox)
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
    _aicore_test_common(bench_rfdetr_perf NO_CTEST)
endfunction()

if(TARGET AICore_test)
    # White-box tests compile private AICore implementation sources.
    aicore_add_whitebox_test(test_data_root common)
    aicore_add_whitebox_test(test_backend_registry common)
    aicore_add_whitebox_test(test_image_view_contract common)
    aicore_add_whitebox_test(test_gguf_keys depth)
    aicore_add_whitebox_test(test_path_util depth)
    aicore_add_whitebox_test(test_compute_mode depth)

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
        aicore_add_depth_whitebox_test(${_t} LABELS whitebox;model)
    endforeach()

    aicore_add_whitebox_test(test_loader gaussian)
    aicore_add_whitebox_test(test_graph_blocks gaussian)
    aicore_add_whitebox_test(test_image gaussian)
    aicore_add_whitebox_test(test_pose gaussian)
    aicore_add_whitebox_test(test_gaussian_path_util gaussian)
    aicore_add_whitebox_test(test_parity gaussian LABELS model)
endif()

# --- depth (C API / libAICore) ---
aicore_add_capi_test(test_runtime_capi_contract common LIBS Threads::Threads)
aicore_add_capi_test(test_pipeline_timing_contract common LIBS Threads::Threads)
# Shared GGUF open helper (F-06): synthetic-file unit coverage for both
# no_alloc modes, ownership round-trip, and the failure-message shape.
# Links AICore_test (not the AICore shared lib) because the helper header
# includes ggml's gguf.h, which only the static-test target propagates.
if(TARGET AICore_test)
    add_executable(test_gguf_file_io common/test_gguf_file_io.cpp)
    target_link_libraries(test_gguf_file_io PRIVATE AICore_test)
    _aicore_test_common(test_gguf_file_io)
    if(NOT MSVC)
        target_compile_options(test_gguf_file_io PRIVATE -fPIC)
        target_link_options(test_gguf_file_io PRIVATE "-no-pie")
    endif()
    set_tests_properties(test_gguf_file_io PROPERTIES LABELS "capi")
endif()
aicore_add_capi_test(test_depth_capi_contract depth)
aicore_add_capi_test(test_image_depth depth)
aicore_add_capi_test(test_depth_capi_load depth LABELS capi;model)
aicore_add_capi_test(test_depth_capi_backend_parity depth LABELS capi;model;gpu;parity)

# --- gaussian (C API / libAICore) ---
aicore_add_capi_test(test_gaussian_capi_contract gaussian)
aicore_add_capi_test(test_gaussian_capi_load gaussian LABELS capi;model)

aicore_add_capi_test(test_lightglue_capi_contract lightglue)
aicore_add_capi_test(test_lightglue_aliked_e2e lightglue LABELS capi;model;gpu;e2e)
aicore_add_capi_test(test_lightglue_model_validation lightglue LABELS capi;model;gpu;validation)

aicore_add_capi_test(test_loma_capi_contract loma)
aicore_add_capi_test(test_loma_default_triplet loma LABELS capi;model;e2e)
aicore_add_capi_test(test_loma_matcher_e2e loma LABELS capi;model;e2e)
aicore_add_capi_test(test_loma_published_model_gate loma LABELS capi;model;validation)
aicore_add_capi_test(test_loma_detector_reference_gate loma LABELS capi;model;reference;accuracy)
aicore_add_capi_test(test_loma_descriptor_reference_gate loma LABELS capi;model;reference;accuracy)
aicore_add_loma_whitebox_test(test_loma_g_token_reference_gate)
aicore_add_loma_whitebox_test(test_loma_g_backend_parity)
aicore_add_capi_test(test_loma_b128_e2e_reference_gate loma LABELS capi;model;reference;accuracy;e2e)
aicore_add_capi_test(test_loma_matcher_reference_gate loma LABELS capi;model;validation)

aicore_add_capi_test(test_aliked_capi_contract aliked)
aicore_add_capi_test(test_aliked_capi_parity aliked LABELS capi;model;gpu)
aicore_add_capi_test(test_aliked_e2e_smoke aliked LABELS capi;model;gpu;smoke)

aicore_add_capi_test(test_deeplsd_capi_contract deeplsd)
aicore_add_capi_test(test_deeplsd_capi_load deeplsd LABELS capi;model)

aicore_add_capi_test(test_gkd_capi_contract gkd)
aicore_add_capi_test(test_gkd_capi_load gkd LABELS capi;model)
aicore_add_capi_test(test_anycalib_capi_contract anycalib)
aicore_add_capi_test(test_map_anything_capi_contract map_anything)
aicore_add_capi_test(test_map_anything_capi_load map_anything LABELS capi;model)
# Candidate half of the upstream cpp_ggml frames parity gate (NOT a ctest: it
# needs the external vggt-cli reference; driven by the manifest row
# map_anything-upstream-frames via run_map_anything_upstream_parity.py).
aicore_add_capi_tool(test_map_anything_upstream_frames
    map_anything/test_map_anything_upstream_frames.cpp)

aicore_add_capi_test(test_lingbot_capi_contract lingbot)
aicore_add_capi_test(test_lingbot_capi_load lingbot LABELS capi;model)
# Upstream numeric-parity probe (pinned courthouse fixtures + captured
# upstream goldens; see tests/lingbot/gen_stream_parity_golden.py). Needs
# AICORE_TEST_LINGBOT_GGUF/AICORE_TEST_LINGBOT_INPUT_DIR — bare ctest runs
# skip it (exit 77); the validation runner wires the assets.
aicore_add_capi_test(test_lingbot_stream_parity lingbot LABELS capi;model)
# The probe decodes the pinned PNG fixtures through QImage.
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(test_lingbot_stream_parity PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(test_lingbot_stream_parity PRIVATE Qt5::Core Qt5::Gui)
endif()

aicore_add_capi_test(test_reid_capi_contract reid)
aicore_add_capi_test(test_reid_capi_load reid LABELS capi;model)

# --- OCR (C API / libAICore) ---
aicore_add_capi_test(test_ocr_capi_contract ocr LABELS capi;ocr)
aicore_add_capi_tool(test_ocr_orientation ocr/test_ocr_orientation.cpp
    LIBS 3rdparty_jsoncpp)
# Real-model recognize-only ABI against the det path (feedback, order,
# degenerate quads, BGR); env wired in aicore_configure_model_test_assets,
# exits 77 without the assets.
aicore_add_capi_test(test_ocr_capi_e2e ocr LABELS capi;ocr;model;e2e)
# Model-free DB box-score rasterization parity: ocr_box_score.cpp is pure C++
# and compiled directly; the golden mask/score rows come from PaddleX
# box_score_fast + cv2.fillPoly (scripts/gen_db_box_score_golden.py).
add_executable(test_ocr_box_score ocr/test_ocr_box_score.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/ocr/ocr_box_score.cpp)
target_include_directories(test_ocr_box_score PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/ocr ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(test_ocr_box_score PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(test_ocr_box_score PRIVATE
    AICORE_OCR_BOX_SCORE_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/ocr/db_box_score_golden.txt")
_aicore_test_common(test_ocr_box_score)
set_tests_properties(test_ocr_box_score PROPERTIES LABELS "capi;ocr")
# Model-free rec crop parity (minAreaRect re-fit + cubic warpPerspective)
# against PaddleX CropByPolys + cv2 (scripts/gen_ocr_rec_crop_golden.py),
# and DB post-process parity (findContours + get_mini_boxes + pyclipper
# unclip + SortQuadBoxes) against PaddleX DBPostProcess
# (scripts/gen_db_postprocess_golden.py). Same contraction rule as the
# libAICore TUs (core/AICore/CMakeLists.txt).
set(_ocr_geom_dir ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/ocr)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        ${_ocr_geom_dir}/ocr_rec_crop.cpp ${_ocr_geom_dir}/ocr_db_postprocess.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
add_executable(test_ocr_rec_crop ocr/test_ocr_rec_crop.cpp
    ${_ocr_geom_dir}/ocr_rec_crop.cpp)
add_executable(test_ocr_db_postprocess ocr/test_ocr_db_postprocess.cpp
    ${_ocr_geom_dir}/ocr_db_postprocess.cpp ${_ocr_geom_dir}/ocr_rec_crop.cpp
    ${_ocr_geom_dir}/ocr_box_score.cpp)
foreach(_t test_ocr_rec_crop test_ocr_db_postprocess)
    target_include_directories(${_t} PRIVATE ${_ocr_geom_dir}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set_target_properties(${_t} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    _aicore_test_common(${_t})
    set_tests_properties(${_t} PROPERTIES LABELS "capi;ocr")
endforeach()
target_compile_definitions(test_ocr_rec_crop PRIVATE
    AICORE_OCR_REC_CROP_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/ocr/rec_crop_golden.txt")
target_compile_definitions(test_ocr_db_postprocess PRIVATE
    AICORE_OCR_DB_POSTPROCESS_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/ocr/db_postprocess_golden.txt"
    AICORE_OCR_SOURCE_DET_FILTER_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/ocr/source_det_filter_golden.txt")

# --- table (C API / libAICore) ---
aicore_add_capi_test(test_table_capi_contract table)
# Real-model OCR + SLANet_plus (+ layout page) end-to-end; env wired in
# aicore_configure_model_test_assets, exits 77 without the assets.
aicore_add_capi_test(test_table_capi_e2e table LABELS capi;model;e2e)
# Model-free post-process unit test: structure decode, span grid, OCR cell
# assignment, HTML/CSV. table_postprocess.cpp is pure C++ over the task
# headers (ggml headers only, no ggml call), so it is compiled directly.
add_executable(test_table_postprocess table/test_table_postprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/table/table_postprocess.cpp)
target_include_directories(test_table_postprocess PRIVATE
    ${AICORE_INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/ocr
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/table
    ${JSONCPP_INCLUDE_DIRS})
target_link_libraries(test_table_postprocess PRIVATE ${GGML_DEPENDENCY_TARGET} 3rdparty_jsoncpp)
set_target_properties(test_table_postprocess PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
_aicore_test_common(test_table_postprocess)
set_tests_properties(test_table_postprocess PROPERTIES LABELS "capi;table")

# RAM / RAM++ / Tag2Text public ABI and real-asset matrix probe.
aicore_add_capi_test(test_relation_capi_contract relation)
add_executable(test_relation_probe relation/test_relation_probe.cpp)
target_link_libraries(test_relation_probe PRIVATE AICore Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui)
_aicore_capi_link(test_relation_probe)
_aicore_test_common(test_relation_probe NO_CTEST)
if(UNIX AND NOT APPLE)
    target_compile_options(test_relation_probe PRIVATE -fPIC)
    target_link_options(test_relation_probe PRIVATE -no-pie)
endif()

aicore_add_capi_test(test_tagging_capi_contract tagging)
add_executable(test_tagging_probe tagging/test_tagging_probe.cpp)
target_link_libraries(test_tagging_probe PRIVATE AICore)
_aicore_capi_link(test_tagging_probe)
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(test_tagging_probe PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(test_tagging_probe PRIVATE Qt5::Core Qt5::Gui)
endif()
if(UNIX AND NOT APPLE)
    target_compile_options(test_tagging_probe PRIVATE -fPIC)
    target_link_options(test_tagging_probe PRIVATE -no-pie)
endif()
_aicore_test_common(test_tagging_probe NO_CTEST)

aicore_add_capi_test(test_llm_capi_contract llm)
aicore_add_capi_test(test_attribute_capi_contract attribute)
aicore_add_capi_test(test_attribute_capi_e2e attribute LIBS 3rdparty_jsoncpp LABELS capi;model;e2e)
target_include_directories(test_attribute_capi_e2e PRIVATE ${JSONCPP_INCLUDE_DIRS})
add_executable(test_llm_probe llm/test_llm_probe.cpp)
target_link_libraries(test_llm_probe PRIVATE AICore)
_aicore_capi_link(test_llm_probe)
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(test_llm_probe PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(test_llm_probe PRIVATE Qt5::Core Qt5::Gui)
endif()
if(UNIX AND NOT APPLE)
    target_compile_options(test_llm_probe PRIVATE -fPIC)
    target_link_options(test_llm_probe PRIVATE -no-pie)
endif()
_aicore_test_common(test_llm_probe NO_CTEST)


# Manual parity dump tool (NOT a ctest): writes upstream-compatible LBF3
# per-frame binaries for the compare_lingbot_parity.py A/B against
# lingbot-map-cli --stream-dir. Same compile setup as the capi tests, minus
# add_test() so the ctest tiers never select a missing binary.
aicore_add_capi_tool(dump_lingbot_frames lingbot/dump_lingbot_frames.cpp)

aicore_add_capi_tool(test_lingbot_reset_freshness
    lingbot/test_lingbot_reset_freshness.cpp)

if(TARGET AICore_test)
    aicore_add_facedetect_whitebox_test(test_facedetect_align)
endif()
aicore_add_capi_test(test_facedetect_embed_capi facedetect LABELS capi;model)
aicore_add_capi_test(test_facedetect_dense_validation facedetect LABELS capi;model;validation)
aicore_add_capi_test(test_facedetect_match_capi facedetect)
aicore_add_capi_test(test_facedetect_contexts_capi facedetect LABELS capi;model)

# Detector-only validation for the pure-detector SCRFD packs (no recognizer
# head, e.g. scrfd_10g_bnkps): real-image detection + run-to-run stability.
# (Upstream registered this through the removed facedetect wrapper; the
# generic aicore_add_capi_test(name dir) call is its post-refactor equivalent —
# explicit set_tests_properties below still carries the LABELS/SKIP/TIMEOUT.)
aicore_add_capi_test(test_facedetect_detect_only facedetect)
set_tests_properties(test_facedetect_detect_only
    PROPERTIES LABELS "capi;model;validation"
               SKIP_RETURN_CODE 77
               TIMEOUT 1800)

# FaceDetect C API contract test (no GGUF assets: ABI, catalog, lifecycle,
# null guards — see tests/facedetect/test_facedetect_capi_contract.cpp).
aicore_add_capi_test(test_facedetect_capi_contract facedetect LABELS capi)

aicore_add_rfdetr_whitebox_test(test_rfdetr_preprocess)
aicore_add_rfdetr_whitebox_test(test_rfdetr_two_stage_mask)
aicore_add_rfdetr_perf_bench()
aicore_add_capi_test(test_rfdetr_capi_contract rfdetr)
aicore_add_capi_test(test_rmbg_capi_contract rmbg)
aicore_add_capi_test(test_rmbg_capi_performance rmbg LABELS capi;model;gpu;performance)
aicore_add_capi_test(test_sam3_capi_contract sam3)

# Cross-catalog default-row contract (see tests/catalog/ header): locks the
# aicore_<task>_model_default_index() declarations and the
# "first row of every family is the recommended pack" catalogs to the
# visible "(recommended)" marker. Fast, no GGUF assets.
aicore_add_capi_test(test_model_catalog_defaults catalog)
aicore_add_capi_test(test_shared_model_catalog catalog)

# Manual real-model acceptance probe for a ggml-version worktree. It is not a
# CTest because it requires a local SAM3 GGUF, a Vulkan-capable GPU, and takes
# several minutes on CPU. See the source header for its exact parity gates.
aicore_add_capi_tool(bench_sam3_backend_acceptance
    sam3/bench_sam3_backend_acceptance.cpp
    INCLUDE_DIRS ${AICORE_TEST_INC})
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(bench_sam3_backend_acceptance PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(bench_sam3_backend_acceptance PRIVATE Qt5::Core Qt5::Gui)
endif()

# Per-stage CPU/Vulkan divergence locator for the SAM3 ViT encoder. Manual
# diagnostic (needs a local SAM3 GGUF + Vulkan GPU), not a CTest. The internal
# implementation is deliberately compiled into this test-only static library;
# no SAM3 C++ symbol is part of libAICore's production ABI.
# Define SAM3_BUILD but never SAM3_SHARED here: SAM3_API must expand to
# nothing. libAICore intentionally does not export the SAM3 C++ symbols, and
# on MSVC a dllimport-consuming target cannot resolve __imp_?sam3_* against a
# static library (archive members are selected by exact symbol name, so the
# linker never pulls in the member defining the undecorated ?sam3_* symbol).
add_library(AICore_sam3_diagnostics STATIC
    ${AICORE_SAM3_SRC_DIR}/sam3.cpp
    # sam3.cpp discovers backends via ggml_common::load_backends_once() (same
    # mechanism as every other task), whose implementation calls into the
    # env-bridge TU — it must be part of this archive's closure.
    ${AICORE_COMMON_SRC_DIR}/ggml_env_bridge.cpp
    # sam3.cpp holds its backend through aicore::runtime::BackendLease
    # (acquire_backend_lease). libAICore hides all C++ symbols, so the
    # registry TU defining those symbols must be part of this archive's
    # closure too; its ggml_common helpers are header-only and the ggml C
    # API resolves through libAICore's DT_NEEDED ggml library.
    ${AICORE_COMMON_SRC_DIR}/ggml_backend_registry.cpp)
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
# sam3.cpp calls dladdr() through ggml_backend_utils.hpp.  Because this
# implementation is archived into a static library, the final executable
# must carry the platform's dynamic-loader library explicitly.
target_link_libraries(AICore_sam3_diagnostics PRIVATE ${CMAKE_DL_LIBS})

add_executable(cmp_sam3_vit_stages sam3/cmp_sam3_vit_stages.cpp)
target_link_libraries(cmp_sam3_vit_stages PRIVATE AICore_sam3_diagnostics AICore)
# No SAM3_SHARED: the test TU must see SAM3_API as empty so its sam3.h calls
# resolve directly against the AICore_sam3_diagnostics static library instead
# of dllimport (__imp_) references that no import library provides.
add_executable(cmp_sam3_mask_triage sam3/cmp_sam3_mask_triage.cpp)
target_link_libraries(cmp_sam3_mask_triage PRIVATE AICore_sam3_diagnostics AICore)
set_target_properties(cmp_sam3_mask_triage PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Same grouped runtime dir as cmp_sam3_vit_stages — without this the triage
# probe would land in bin/ root and miss the aicore_tests ggml backend copies.
cloudviewer_set_aicore_test_runtime_layout(cmp_sam3_mask_triage)
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
aicore_add_capi_test(test_trellis_capi_contract trellis)

# SAM 3D C API contract test (model-free) plus the real-asset acceptance
# probe consumed by the aicore-validate-all runner.
aicore_add_capi_test(test_sam3d_capi_contract sam3d)
aicore_add_capi_test(test_sam3d_scene_assemble sam3d)

aicore_add_capi_tool(bench_sam3d_backend_acceptance
    sam3d/bench_sam3d_backend_acceptance.cpp
    INCLUDE_DIRS ${AICORE_TEST_INC})

# White-box stage-bisect probe: runs cmd_e2e from a fixed condition/noise
# directory (upstream sam3d-cli --conditions-out layout) so output
# divergence versus the upstream reference is attributable to kernel
# execution. Static white-box link keeps cmd_e2e private.
if(TARGET AICore_test)
    add_executable(sam3d_stage_bisect sam3d/sam3d_stage_bisect.cpp)
    target_link_libraries(sam3d_stage_bisect PRIVATE AICore_test)
    target_include_directories(sam3d_stage_bisect PRIVATE
        ${AICORE_TEST_INC}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tasks/sam3d)
    aicore_apply_compile_definitions(sam3d_stage_bisect)
    cloudviewer_set_aicore_test_runtime_layout(sam3d_stage_bisect)
endif()


# YOLO C API performance benchmark (upstream ultralytics-ggml bench matrix
# counterpart; see tests/yolo/test_yolo_capi_performance.cpp header for the
# timing contract). Skips (77) without AICORE_TEST_YOLO_* assets.
function(aicore_add_yolo_capi_test name)
    cmake_parse_arguments(YCT "" "" "LIBS;LABELS" ${ARGN})
    set(_aicore_yolo_args)
    if(WIN32)
        # dirent.h is POSIX-only; 3rdparty/dirent provides the MSVC shim
        # (test_yolo_capi_performance.cpp includes <dirent/dirent.h> there).
        list(APPEND _aicore_yolo_args LIBS 3rdparty_dirent)
    endif()
    if(YCT_LIBS)
        list(APPEND _aicore_yolo_args LIBS ${YCT_LIBS})
    endif()
    if(YCT_LABELS)
        list(APPEND _aicore_yolo_args LABELS ${YCT_LABELS})
    endif()
    aicore_add_capi_test(${name} yolo ${_aicore_yolo_args})
endfunction()

aicore_add_yolo_capi_test(test_yolo_capi_performance LABELS capi;model;gpu;performance)
aicore_add_yolo_capi_test(test_yolo_text_model_validation LABELS capi;model;validation;performance)

# TEMPORARY probe for reproducing the yoloe/seg visualization issues
# (delete after diagnosis).
aicore_add_capi_tool(yolo_prompt_probe yolo/yolo_prompt_probe.cpp
    INCLUDE_DIRS ${AICORE_TEST_INC})
if(TARGET Qt::Core AND TARGET Qt::Gui)
    target_link_libraries(yolo_prompt_probe PRIVATE Qt::Core Qt::Gui)
elseif(Qt5_FOUND)
    target_link_libraries(yolo_prompt_probe PRIVATE Qt5::Core Qt5::Gui)
endif()

# YOLO device-parity test (CPU vs GPU numeric gates on real GGUF assets;
# see tests/yolo/test_yolo_capi_parity.cpp for the gate table). Skips (77)
# without AICORE_TEST_YOLO_* assets or a GPU backend.
aicore_add_yolo_capi_test(test_yolo_capi_parity LABELS capi;model;gpu)

# Per-call vocabulary switching on a resident world detector (the reload-
# avoidance contract). Skips (77) without AICORE_TEST_YOLO_* assets.
aicore_add_yolo_capi_test(test_yolo_with_classes LABELS capi;model)

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
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
                -R "^test_yolo_model_matrix$" -C $<CONFIG>
        DEPENDS test_yolo_capi_performance test_yolo_capi_parity
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running YOLO task/quant/backend model matrix"
        VERBATIM)
endif()

# YOLO C API contract test (no GGUF assets: ABI, options, catalog, null
# guards, model-free error paths — see tests/yolo/test_yolo_capi_contract.cpp).
aicore_add_yolo_capi_test(test_yolo_capi_contract LABELS capi)

# YOLOE visual-prompt (SAVPE) model-tier test (needs a savpe-converted
# yoloe GGUF + prompt image; skips with 77 without
# AICORE_TEST_YOLO_SAVPE_* assets — see test_yolo_capi_savpe.cpp).
aicore_add_yolo_capi_test(test_yolo_capi_savpe LABELS capi;model)

# YOLO geometry whitebox test (no GGUF assets: mask unscale / letterbox
# inverse mapping — see tests/yolo/test_yolo_unscale_masks.cpp).
function(aicore_add_yolo_whitebox_test name)
    set(_labels whitebox)
    if(ARGC GREATER 1)
        set(_labels "whitebox;${ARGV1}")
    endif()
    aicore_add_whitebox_test(${name} yolo LABELS ${_labels})
endfunction()

aicore_add_yolo_whitebox_test(test_yolo_unscale_masks)
aicore_add_yolo_whitebox_test(test_yolo_f32_convolution)
aicore_add_yolo_whitebox_test(test_yolo_mask_upsample)
aicore_add_yolo_capi_test(test_yolo_mask_upsample_capi)

# Whitebox op-level bisection tool for world-family device-parity failures
# (model/env-gated; see tests/yolo/test_yolo_world_optrace.cpp).
function(aicore_add_yolo_optrace_test name)
    aicore_add_yolo_whitebox_test(${name} gpu)
endfunction()
aicore_add_yolo_optrace_test(test_yolo_world_optrace)

# ReID algebraic diagnostic (white-box bisection): feed an exact CHW tensor
# (saved from the PyTorch baseline) and dump the SAME graph run's logits and
# embedding, so the numpy side can check logits == W @ embed + b and pin the
# first divergent node. Manual tool, NOT a ctest: it needs a CHW binary on
# the side, and it links the whitebox library, which only exists with
# AICore_BUILD_WHITEBOX_TESTS=ON (without the guard the unconditional target
# would compile without Qt include dirs / link a nonexistent -lAICore_test
# on GUI builds like macOS CI).
#   reid_diag --gguf <file> --chw <f32.bin> [--device cuda] [--dump <prefix>]
if(TARGET AICore_test)
    add_executable(reid_diag ${CMAKE_CURRENT_SOURCE_DIR}/../tools/reid_diag.cpp)
    target_link_libraries(reid_diag PRIVATE AICore_test)
    if(NOT MSVC)
        target_compile_options(reid_diag PRIVATE -fPIC)
        target_link_options(reid_diag PRIVATE "-no-pie")
    endif()
    _aicore_test_common(reid_diag NO_CTEST)
endif()

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
        "${AICORE_TEST_ENV_EXTRA}"
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
        "AICORE_TEST_RMBG_GGUF=${_root}/rmbg_models/rmbg_f16.gguf"
        "AICORE_TEST_GKD_GGUF=${_root}/gkd_models/gkd_fullset-q4_K.gguf"
        "AICORE_TEST_GKD_IMAGE=${_root}/objects_detection_data/images/bus.jpg"
        "AICORE_TEST_TABLE_GGUF=${_root}/ocr_models/slanet-plus-table-f32.gguf"
        "AICORE_TEST_OCR_DET_GGUF=${_root}/ocr_models/ppocr-det-ch-f32.gguf"
        "AICORE_TEST_OCR_REC_GGUF=${_root}/ocr_models/ppocr-rec-ch-f32.gguf"
        "AICORE_TEST_OCR_LAYOUT_GGUF=${_root}/ocr_models/pplcnet-layout-f32.gguf"
        "AICORE_TEST_TABLE_IMAGE=${_root}/ppocr_testdata/test_files/table.jpg"
        "AICORE_TEST_TABLE_PAGE=${_root}/ppocr_testdata/layout/table_page_1.png")
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
            test_rmbg_capi_performance
            test_gkd_capi_load
            test_map_anything_capi_load
            test_table_capi_e2e
            test_ocr_capi_e2e)
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

    # OCR interpreter parity probes: full rec/det graphs against the numpy
    # PaddlePaddle-semantics fixtures (regenerated by
    # scripts/gen_ocr_fixtures.py; see the manifest accuracy_gate prose for
    # the measured tolerance rationale). The probe exits 77 when the
    # fixtures or the f32 models have not been fetched into the asset root.
    add_test(
        NAME probe_ocr_rec
        COMMAND $<TARGET_FILE:test_ocr_runtime>
                ${_root}/ocr_fixtures
                ${_root}/ocr_models/ppocr-rec-ch-f32.gguf
                rec
                ${AICORE_TEST_DEVICE}
                -1 -
                0.01)
    set_tests_properties(probe_ocr_rec PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
        LABELS "capi;model;ocr"
        ENVIRONMENT "${_env}")
    add_test(
        NAME probe_ocr_det
        COMMAND $<TARGET_FILE:test_ocr_runtime>
                ${_root}/ocr_fixtures
                ${_root}/ocr_models/ppocr-det-ch-f32.gguf
                det
                ${AICORE_TEST_DEVICE}
                -1 -
                0.01)
    set_tests_properties(probe_ocr_det PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
        LABELS "capi;model;ocr"
        ENVIRONMENT "${_env}")
    add_test(
        NAME probe_ocr_rec_en
        COMMAND $<TARGET_FILE:test_ocr_runtime>
                ${_root}/ocr_fixtures
                ${_root}/ocr_models/ppocr-rec-en-f32.gguf
                rec
                ${AICORE_TEST_DEVICE}
                -1 -
                0.01)
    set_tests_properties(probe_ocr_rec_en PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
        LABELS "capi;model;ocr"
        ENVIRONMENT "${_env}")
    # Layout heads (PicoDet CDLA) against the official Paddle inference
    # fixtures; fetches are contiguous and the fixtures may be Fortran order.
    add_test(
        NAME probe_ocr_layout
        COMMAND $<TARGET_FILE:test_ocr_runtime>
                ${_root}/ocr_fixtures
                ${_root}/ocr_models/pplcnet-layout-f32.gguf
                layout
                ${AICORE_TEST_DEVICE}
                -1 -
                0.02)
    # SLANet_plus structure probs/loc on three official table crops
    # (table_fixtures.json); token argmax must match the fixture exactly.
    add_test(
        NAME probe_table
        COMMAND $<TARGET_FILE:test_table_runtime>
                ${_root}/ocr_fixtures
                ${_root}/ocr_models/slanet-plus-table-f32.gguf
                ${AICORE_TEST_DEVICE}
                0.01 0.005)
    set_tests_properties(probe_ocr_layout probe_table PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
        LABELS "capi;model;ocr"
        ENVIRONMENT "${_env}")
    if(AICORE_TEST_AUTO_DOWNLOAD)
        set_tests_properties(probe_ocr_rec probe_ocr_det probe_ocr_rec_en
            probe_ocr_layout probe_table
            PROPERTIES FIXTURES_REQUIRED aicore_model_assets)
    endif()

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

# Contract tier: ctest -L capi, with -LE model|gpu|e2e when model tests are
# disabled. The DEPENDS list is derived at the bottom of this file from the
# registration registry + LABELS (the exact predicate ctest applies), so a
# newly registered capi test needs no list edit here. Every tier passes
# --no-tests=error: ctest exits 0 on an empty selection, which would turn a
# relabeled or unregistered tier into a green no-op.
set(_aicore_contract_model_exclude)
if(NOT AICORE_TEST_ENABLE_MODEL_TESTS)
    list(APPEND _aicore_contract_model_exclude -LE "model|gpu|e2e")
endif()
add_custom_target(aicore-contract-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error -L capi
            ${_aicore_contract_model_exclude}
            -j1
            -C $<CONFIG>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore public contract tests"
    VERBATIM
)

# --- Catalog vs release asset consistency (network; CI only) ---------------
# The catalog can be self-consistent while individual entries 404 on the
# release (historical rmbg_q4_K.gguf regression). Dump every catalog URL and
# HEAD-check it against GitHub/Hugging Face so drift fails the build instead
# of the user's first download. Host throttling (HTTP 429/5xx, e.g. Hugging
# Face anonymous rate limits on shared CI runners) is retried by the checker
# and reported as SKIPPED when persistent — it is transient, not drift. Kept
# out of the "capi" label so offline/contract runs stay network-free.
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
    # The checker spawns test_catalog_dump_urls directly, so propagate the
    # configure-time Qt runtime (same rationale as AICORE_TEST_ENV_EXTRA and
    # the validation runner below): on a mixed Qt stack the spawned binary
    # would otherwise resolve a system Qt and fail on versioned operator
    # delete (exit 127).
    add_test(
        NAME test_catalog_remote_assets
        COMMAND ${CMAKE_COMMAND} -E env ${AICORE_TEST_ENV_EXTRA}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_catalog_assets.py
                $<TARGET_FILE:test_catalog_dump_urls>)
    set_tests_properties(test_catalog_remote_assets PROPERTIES
        LABELS "catalog-remote" TIMEOUT 600)
endif()

# --- Fast tier: no GGUF assets (whitebox/common/gaussian units + no-asset
# capi subset + catalog URL dump). Mirror entry point of aicore-contract-tests
# so CI can address the tier by name. DEPENDS derived at the bottom of this
# file: fast-classified binaries plus the contract-tier set (the fast ctest
# -LE also runs the no-asset capi subset).
add_custom_target(aicore-fast-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
            -LE "model|gpu|e2e|cvpluginapi"
            -j2
            -C $<CONFIG>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore fast-tier tests (no GGUF assets)"
    VERBATIM
)

# --- Model tier: needs GGUF assets (AICORE_TEST_ENABLE_MODEL_TESTS=ON +
# asset root). Tests without assets skip (SKIP_RETURN_CODE 77), so this
# target never fails on missing assets — it only executes them. Mirror of
# the fast/contract entry points; the ctest -L "model" selection also covers
# the two parity variants that reuse the test_aliked_capi_parity binary.
add_custom_target(aicore-model-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error -L "model"
            -j1
            -C $<CONFIG>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AICore model-tier tests (skips without GGUF assets)"
    VERBATIM
)

# Complete real-model/task regression gate. The runner discovers every model,
# executes task-specific probes serially, and rejects missing coverage.
if(Python3_EXECUTABLE AND NOT CMAKE_CROSSCOMPILING)
    set(_aicore_validation_targets
        test_tagging_probe
        test_llm_probe
        test_relation_probe
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
        test_gkd_capi_load
        test_map_anything_capi_load
        test_map_anything_upstream_frames
        test_lingbot_capi_load
        test_reid_capi_load
        test_yolo_capi_performance
        test_yolo_text_model_validation
        test_yolo_capi_parity)
    list(APPEND _aicore_validation_targets bench_sam3d_backend_acceptance)
    # bench_rfdetr_perf is only generated when the whitebox library exists
    # (AICore_BUILD_WHITEBOX_TESTS); drop it from the dependency list
    # otherwise so the gate is still runnable from a plain build.
    if(NOT TARGET AICore_test)
        list(REMOVE_ITEM _aicore_validation_targets bench_rfdetr_perf)
    endif()
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
    # Same rationale as AICORE_TEST_ENV_EXTRA: the runner spawns probe
    # processes directly, and a dlopen'ed libggml-cuda.so needs the toolkit
    # runtime directory or CUDA silently disappears in those probes too.
    if(DEFINED _aicore_cuda_lib_dir)
        list(APPEND _aicore_validation_runtime_env
             "AICORE_CUDA_LIBRARY_DIR=${_aicore_cuda_lib_dir}")
    endif()
    # External truth programs for {reference:NAME} manifest rows, e.g.
    # "map_anything_ggml=/path/to/map-anything-ggml". Rows whose reference is
    # not configured are reported skipped, which fails the gate.
    set(AICORE_VALIDATION_REFERENCES "" CACHE STRING
        "Semicolon list of NAME=PATH upstream references for aicore-validate-all")
    set(_aicore_validation_reference_args)
    foreach(_aicore_reference IN LISTS AICORE_VALIDATION_REFERENCES)
        list(APPEND _aicore_validation_reference_args --reference "${_aicore_reference}")
    endforeach()
    add_custom_target(aicore-validate-all
        COMMAND ${CMAKE_COMMAND} -E env ${_aicore_validation_runtime_env}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_all.py
                --build ${CMAKE_BINARY_DIR}
                --assets ${AICORE_TEST_ASSET_ROOT}
                --backend ${AICORE_TEST_DEVICE}
                --tasks ${_aicore_enabled_tasks_csv}
                --timeout 14400
                ${_aicore_validation_reference_args}
                --output ${CMAKE_BINARY_DIR}/Testing/aicore_validation.json
        DEPENDS test_catalog_dump_urls ${_aicore_validation_targets}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running AICore model/task regression matrix (manifest-declared "
                "light tier; use validate_all.py --full for the complete matrix)"
        USES_TERMINAL
        VERBATIM)

    add_test(NAME test_capi_coverage_audit
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/common/test_capi_coverage_audit.py)
    set_tests_properties(test_capi_coverage_audit PROPERTIES LABELS "fast;validation")

    add_test(NAME test_aicore_validation_runner
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_validate_all.py)
    set_tests_properties(test_aicore_validation_runner PROPERTIES
        LABELS "fast;validation")

    add_test(NAME test_attribute_probe_contract
        COMMAND ${CMAKE_COMMAND} -E env ${_aicore_validation_runtime_env}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/attribute/test_attribute_probe_contract.py
                --library $<TARGET_FILE:AICore>)
    set_tests_properties(test_attribute_probe_contract PROPERTIES LABELS "model;attribute;validation" TIMEOUT 120)

    add_test(NAME test_tagging_timing_audit
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tagging/test_upstream_metrics_probe.py)
    set_tests_properties(test_tagging_timing_audit PROPERTIES
        LABELS "fast;validation")

    add_test(NAME test_relation_upstream_audit
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/relation/test_upstream_probe.py)
    set_tests_properties(test_relation_upstream_audit PROPERTIES
        LABELS "fast;validation")

    add_test(NAME test_relation_accuracy_gate
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/relation/test_accuracy.py)
    set_tests_properties(test_relation_accuracy_gate PROPERTIES
        LABELS "fast;validation")

    add_test(NAME test_relation_dependency_closure
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/relation/test_dependency_closure.py
                --root ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_relation_dependency_closure PROPERTIES
        LABELS "fast;validation")
    add_test(NAME test_map_anything_upstream_parity_runner
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_run_map_anything_upstream_parity.py)
    set_tests_properties(test_map_anything_upstream_parity_runner PROPERTIES
        LABELS "fast;validation")

    add_test(NAME test_ocr_gt_eval
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_ocr_gt_eval.py)
    set_tests_properties(test_ocr_gt_eval PROPERTIES
        LABELS "fast;validation;ocr")

    add_test(NAME test_ocr_layout_gt_eval
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_layout_gt_eval.py)
    set_tests_properties(test_ocr_layout_gt_eval PROPERTIES
        LABELS "fast;validation;ocr" SKIP_RETURN_CODE 77)

    add_test(NAME test_ocr_paddle_stored_weights
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/test_export_paddle_ocr_stored_weights.py)
    set_tests_properties(test_ocr_paddle_stored_weights PROPERTIES
        LABELS "fast;validation;ocr")

    add_test(NAME test_ocr_probability_gate
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_ocr_onnx_probe.py)
    add_test(NAME test_ocr_gpu_measurement_audit
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_gpu_audit.py)
    add_test(NAME test_ocr_paddle_page_reference
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_ocr_paddle_pipeline.py)
    add_test(NAME test_ocr_paddle_graph_reference
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_ocr_paddle_probe.py)
    add_test(NAME test_ocr_paddle_layout_reference
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_ocr_paddle_layout_probe.py)
    add_test(NAME test_ocr_paddle_reference_device
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_paddle_reference_device.py)
    add_test(NAME test_table_paddle_graph_reference
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_table_paddle_probe.py)
    add_test(NAME test_table_capi_measurement
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_table_capi_probe.py)
    set_tests_properties(test_ocr_probability_gate test_ocr_gpu_measurement_audit
        test_ocr_paddle_page_reference test_ocr_paddle_graph_reference
        test_ocr_paddle_layout_reference test_table_paddle_graph_reference
        test_ocr_paddle_reference_device test_table_capi_measurement
        PROPERTIES LABELS "fast;validation;ocr")

    add_test(NAME test_aicore_plugin_boundaries
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_plugin_aicore_boundaries.py
                --root ${CMAKE_SOURCE_DIR}
                --strict)
    set_tests_properties(test_aicore_plugin_boundaries PROPERTIES
        LABELS "fast;validation")

    add_test(NAME test_aicore_plugin_boundary_scanner
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_check_plugin_aicore_boundaries.py)
    set_tests_properties(test_aicore_plugin_boundary_scanner PROPERTIES
        LABELS "fast;validation")
endif()

# --- Tier DEPENDS derivation (KEEP LAST) ------------------------------------
# Runs after every add_test and LABELS assignment above. Combines the registry
# captured by the add_test override with each test's final LABELS property,
# using the same unanchored label regexes ctest applies, so every tier builds
# exactly the binaries its ctest selection will run. Registering a test here
# can never again miss its tier.
get_property(_aicore_registry_tests GLOBAL PROPERTY AICORE_TEST_NAMES)
set(_aicore_fast_bins "")
set(_aicore_contract_bins "")
set(_aicore_model_bins "")
foreach(_aicore_t IN LISTS _aicore_registry_tests)
    get_property(_aicore_tgt GLOBAL PROPERTY "AICORE_TEST_TGT_${_aicore_t}")
    if(NOT _aicore_tgt)
        continue() # script / cmake -P / multi-binary commands: nothing to build
    endif()
    get_test_property(${_aicore_t} LABELS _aicore_labels)
    if(NOT _aicore_labels OR _aicore_labels MATCHES "NOTFOUND")
        set(_aicore_labels "")
    endif()
    if(_aicore_labels MATCHES "${AICORE_TIER_MODEL_LABEL}")
        list(APPEND _aicore_model_bins ${_aicore_tgt})
    endif()
    if(_aicore_labels MATCHES "${AICORE_TIER_CONTRACT_LABEL}"
            AND (AICORE_TEST_ENABLE_MODEL_TESTS
                OR NOT _aicore_labels MATCHES
                    "${AICORE_TIER_CONTRACT_MODEL_EXCLUDE}"))
        list(APPEND _aicore_contract_bins ${_aicore_tgt})
    endif()
    if(NOT _aicore_labels MATCHES "${AICORE_TIER_FAST_EXCLUDE}")
        list(APPEND _aicore_fast_bins ${_aicore_tgt})
    endif()
endforeach()
foreach(_aicore_list _aicore_fast_bins _aicore_contract_bins _aicore_model_bins)
    if(${_aicore_list})
        list(REMOVE_DUPLICATES ${_aicore_list})
    endif()
endforeach()
# The fast tier also runs the no-asset subset of the capi contract suite, so
# its graph carries the contract binaries too (same union as before).
add_dependencies(aicore-contract-tests ${_aicore_contract_bins})
add_dependencies(aicore-fast-tests ${_aicore_fast_bins} ${_aicore_contract_bins})
add_dependencies(aicore-model-tests ${_aicore_model_bins})
