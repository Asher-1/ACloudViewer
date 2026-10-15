add_executable(test_relate_workflow test_relate_workflow.cpp
    ../src/RelateAnythingDialog.cpp ../src/RelateAnythingWorker.cpp
    ../include/RelateAnythingDialog.h ../include/RelateAnythingWorker.h ../include/RelateAnythingVideoWidget.h)
target_include_directories(test_relate_workflow PRIVATE ../include)
target_link_libraries(test_relate_workflow PRIVATE CVPluginAPI AICore video_base)
set_target_properties(test_relate_workflow PROPERTIES AUTOMOC ON)
if(UNIX AND NOT APPLE)
    target_compile_options(test_relate_workflow PRIVATE -fPIC)
    target_link_options(test_relate_workflow PRIVATE -no-pie -Wl,--allow-shlib-undefined)
endif()
cloudviewer_set_bin_runtime_output_directory(test_relate_workflow)
add_dependencies(test_relate_workflow QRELATEANYTHING_PLUGIN)
add_test(NAME test_relate_workflow COMMAND test_relate_workflow --plugin-module $<TARGET_FILE:QRELATEANYTHING_PLUGIN>)
set_tests_properties(test_relate_workflow PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
if(TARGET aicore-fast-tests)
    add_dependencies(aicore-fast-tests test_relate_workflow)
endif()

# Qt-free unit test of the ported upstream tracker / belief filter.
add_executable(test_relate_temporal test_relate_temporal.cpp ../include/RelateAnythingTemporal.h)
target_include_directories(test_relate_temporal PRIVATE ../include)
cloudviewer_set_bin_runtime_output_directory(test_relate_temporal)
add_test(NAME test_relate_temporal COMMAND test_relate_temporal)
if(TARGET aicore-fast-tests)
    add_dependencies(aicore-fast-tests test_relate_temporal)
endif()

# Numeric parity of the tracker / belief port against upstream render_video.py.
# Skipped (exit 77) unless AICORE_TEST_RELATE_UPSTREAM points at a checkout.
add_executable(test_relate_temporal_replay test_relate_temporal_replay.cpp ../include/RelateAnythingTemporal.h)
target_include_directories(test_relate_temporal_replay PRIVATE ../include)
cloudviewer_set_bin_runtime_output_directory(test_relate_temporal_replay)
find_program(RELATE_PYTHON3 python3)
if(RELATE_PYTHON3)
    add_test(NAME test_relate_temporal_upstream
        COMMAND ${RELATE_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/compare_temporal_upstream.py
                --replay $<TARGET_FILE:test_relate_temporal_replay>)
    set_tests_properties(test_relate_temporal_upstream PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(TARGET aicore-fast-tests)
    add_dependencies(aicore-fast-tests test_relate_temporal_replay)
endif()
