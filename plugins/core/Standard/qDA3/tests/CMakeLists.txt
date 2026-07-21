set(DA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

function(da_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE AICore)
  target_include_directories(${name} PRIVATE
      ${DA_ROOT}/core/AICore/src/depth ${DA_ROOT}/core/AICore/include)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      WORKING_DIRECTORY ${DA_ROOT}
      ENVIRONMENT "DA_TEST_GGUF=${DA_ROOT}/models/depth-anything-base-f32.gguf;DA_TEST_BASELINE=${DA_ROOT}/dumps/reference.gguf;DA_TEST_BASELINE_MV=${DA_ROOT}/dumps/reference_mv.gguf;DA_TEST_BASELINE_MV4=${DA_ROOT}/dumps/reference_mv4.gguf;DA_TEST_GGUF_GIANT=${DA_ROOT}/models/depth-anything-giant-f32.gguf;DA_TEST_BASELINE_GIANT=${DA_ROOT}/dumps/reference_giant.gguf;DA_TEST_GGUF_METRIC=${DA_ROOT}/models/depth-anything-nested-metric.gguf;DA_TEST_GGUF_ANYVIEW=${DA_ROOT}/models/depth-anything-nested-anyview.gguf;DA_TEST_BASELINE_NESTED=${DA_ROOT}/dumps/reference_nested.gguf;DA_TEST_BASELINE_PREPROC=${DA_ROOT}/dumps/reference_preproc_real.gguf;DA_TEST_PREPROC_PNG=${DA_ROOT}/dumps/preproc_real_input.png;DA_TEST_BASELINE_NATIVE=${DA_ROOT}/dumps/reference_native.gguf;DA_TEST_NATIVE_PNG=${DA_ROOT}/dumps/native_input.png;DA_TEST_GGUF_MONO=${DA_ROOT}/models/depth-anything-mono-large-f32.gguf;DA_TEST_BASELINE_MONO=${DA_ROOT}/dumps/reference_mono.gguf;DA_TEST_MONO_PNG=${DA_ROOT}/dumps/mono_input.png;DA_TEST_GGUF_AUX=${DA_ROOT}/models/depth-anything-base-aux-f32.gguf;DA_TEST_BASELINE_RAYS=${DA_ROOT}/dumps/reference_rays.gguf;DA_TEST_BASELINE_RAY_POSE=${DA_ROOT}/dumps/reference_ray_pose.gguf;DA_TEST_GGUF_DA2=${DA_ROOT}/models/depth-anything2-large-f32.gguf;DA_TEST_BASELINE_DA2=${DA_ROOT}/dumps/reference_da2.gguf")
endfunction()

# tests/test_image.cpp — image loading and preprocessing
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_image.cpp)
    da_add_test(test_image)
endif()

# tests/test_loader.cpp — GGUF model loading
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_loader.cpp)
    da_add_test(test_loader)
endif()

# tests/test_pose.cpp — camera pose recovery
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_pose.cpp)
    da_add_test(test_pose)
endif()

# tests/test_parity.cpp — numerical parity vs reference
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_parity.cpp)
    da_add_test(test_parity)
endif()

# tests/test_graph_blocks.cpp — graph building
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_blocks.cpp)
    da_add_test(test_graph_blocks)
endif()

# Standalone harness for scripts/parity_glb.py (takes CLI args; not a ctest).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/glb_parity_dump.cpp)
    add_executable(glb_parity_dump glb_parity_dump.cpp)
    target_link_libraries(glb_parity_dump PRIVATE AICore)
    target_include_directories(glb_parity_dump PRIVATE
        ${DA_ROOT}/core/AICore/src/depth ${DA_ROOT}/core/AICore/include)
endif()

# Standalone harness for scripts/parity_colmap.py (takes CLI args; not a ctest).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/colmap_parity_dump.cpp)
    add_executable(colmap_parity_dump colmap_parity_dump.cpp)
    target_link_libraries(colmap_parity_dump PRIVATE AICore)
    target_include_directories(colmap_parity_dump PRIVATE
        ${DA_ROOT}/core/AICore/src/depth ${DA_ROOT}/core/AICore/include)
endif()
