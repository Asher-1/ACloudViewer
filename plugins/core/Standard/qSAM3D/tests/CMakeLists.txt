add_executable(test_qsam3d_model_selection test_model_selection.cpp
    ../src/Sam3dDialog.cpp ../include/Sam3dDialog.h)
set_target_properties(test_qsam3d_model_selection PROPERTIES AUTOMOC ON
    CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_include_directories(test_qsam3d_model_selection PRIVATE ../include)
target_compile_definitions(test_qsam3d_model_selection PRIVATE AICore_ENABLED)
target_link_libraries(test_qsam3d_model_selection PRIVATE
    AICore CVPluginAPI Qt::Core Qt::Gui Qt::Widgets Qt::Network)
if(NOT MSVC)
    target_compile_options(test_qsam3d_model_selection PRIVATE -fPIC)
    if(NOT APPLE)
        target_link_options(test_qsam3d_model_selection PRIVATE -no-pie)
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(test_qsam3d_model_selection)
if(COMMAND aicore_register_test)
    aicore_register_test(NAME test_qsam3d_model_selection
        TARGET test_qsam3d_model_selection LABELS "cvpluginapi;sam3d;capi")
else()
    add_test(NAME test_qsam3d_model_selection COMMAND test_qsam3d_model_selection)
    set_tests_properties(test_qsam3d_model_selection PROPERTIES LABELS "cvpluginapi;sam3d")
endif()
cloudviewer_set_plugin_test_env(test_qsam3d_model_selection)
