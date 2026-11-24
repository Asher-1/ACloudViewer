
add_executable(test_relate_workflow test_relate_workflow.cpp
    ../src/RelateAnythingDialog.cpp ../src/RelateAnythingWorker.cpp
    ../include/RelateAnythingDialog.h ../include/RelateAnythingWorker.h ../include/RelateAnythingVideoWidget.h)
target_include_directories(test_relate_workflow PRIVATE ../include
    # QtCompat.h (Qt5.12/Qt6 split-behavior constants) in plugin sources.
    PRIVATE ${CMAKE_SOURCE_DIR}/core/include)
target_link_libraries(test_relate_workflow PRIVATE CVPluginAPI AICore video_base gtest)
set_target_properties(test_relate_workflow PROPERTIES AUTOMOC ON)
if(UNIX AND NOT APPLE)
    target_compile_options(test_relate_workflow PRIVATE -fPIC)
    target_link_options(test_relate_workflow PRIVATE -no-pie -Wl,--allow-shlib-undefined)
endif()
cloudviewer_set_bin_runtime_output_directory(test_relate_workflow)
add_dependencies(test_relate_workflow QRELATEANYTHING_PLUGIN)
cloudviewer_add_test(NAME test_relate_workflow LABELS suite-aicore COMMAND test_relate_workflow --plugin-module $<TARGET_FILE:QRELATEANYTHING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_relate_workflow)

# Qt-free unit test of the ported upstream tracker / belief filter.
add_executable(test_relate_temporal test_relate_temporal.cpp ../include/RelateAnythingTemporal.h)
target_include_directories(test_relate_temporal PRIVATE ../include)
target_link_libraries(test_relate_temporal PRIVATE gtest_main)
cloudviewer_set_bin_runtime_output_directory(test_relate_temporal)
cloudviewer_add_test(NAME test_relate_temporal LABELS suite-aicore COMMAND test_relate_temporal)
cloudviewer_set_plugin_test_env(test_relate_temporal)

# Numeric parity of the tracker / belief port against upstream render_video.py.
# Skipped (exit 77) unless AICORE_TEST_RELATE_UPSTREAM points at a checkout.
add_executable(test_relate_temporal_replay test_relate_temporal_replay.cpp ../include/RelateAnythingTemporal.h)
target_include_directories(test_relate_temporal_replay PRIVATE ../include)
target_link_libraries(test_relate_temporal_replay PRIVATE gtest)
cloudviewer_set_bin_runtime_output_directory(test_relate_temporal_replay)
find_program(RELATE_PYTHON3 python3)
if(RELATE_PYTHON3)
    cloudviewer_add_test(NAME test_relate_temporal_upstream
        COMMAND ${RELATE_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/compare_temporal_upstream.py
                --replay $<TARGET_FILE:test_relate_temporal_replay>)
    set_property(TEST test_relate_temporal_upstream APPEND PROPERTY LABELS "e2e")
    cloudviewer_set_plugin_test_env(test_relate_temporal_upstream)
    set_tests_properties(test_relate_temporal_upstream PROPERTIES SKIP_RETURN_CODE 77)
endif()
