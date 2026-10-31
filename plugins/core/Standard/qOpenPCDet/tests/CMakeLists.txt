find_package(Threads REQUIRED)
add_executable(test_qopenpcdet_dialog test_openpcdet_dialog.cpp
    ../src/OpenPCDetDialog.cpp ../src/OpenPCDetModelCatalog.cpp
    ../src/OpenPCDetWorker.cpp ../src/OpenPCDetDbExport.cpp
    ../include/OpenPCDetDialog.h ../include/OpenPCDetWorker.h)
set_target_properties(test_qopenpcdet_dialog PROPERTIES AUTOMOC ON
    CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_include_directories(test_qopenpcdet_dialog PRIVATE ../include)
target_compile_definitions(test_qopenpcdet_dialog PRIVATE AICore_ENABLED)
target_link_libraries(test_qopenpcdet_dialog PRIVATE AICore CVPluginAPI
    CVCoreLib CV_DB_LIB CV_IO_LIB Qt::Core Qt::Gui Qt::Widgets
    Qt::Network Threads::Threads)
if(NOT MSVC)
    target_compile_options(test_qopenpcdet_dialog PRIVATE -fPIC)
    target_link_options(test_qopenpcdet_dialog PRIVATE -no-pie)
endif()
cloudviewer_set_bin_runtime_output_directory(test_qopenpcdet_dialog)
add_test(NAME test_qopenpcdet_dialog COMMAND test_qopenpcdet_dialog)
cloudviewer_set_plugin_test_env_offscreen(test_qopenpcdet_dialog)
set_tests_properties(test_qopenpcdet_dialog PROPERTIES LABELS "capi;openpcdet;cvpluginapi")

if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qOpenPCDet tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qopenpcdet_helpers test_openpcdet_dual_input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/OpenPCDetModelCatalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/OpenPCDetDbExport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/OpenPCDetWorker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/OpenPCDetWorker.h
)

target_compile_definitions(test_qopenpcdet_helpers PRIVATE AICORE_ENABLED)

set_target_properties(test_qopenpcdet_helpers PROPERTIES AUTOMOC ON)

target_include_directories(test_qopenpcdet_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qopenpcdet_helpers PRIVATE
    CVCoreLib
    CVPluginAPI
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Network
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qopenpcdet_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qopenpcdet_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qopenpcdet_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qopenpcdet_helpers)
add_test(NAME test_qopenpcdet_helpers COMMAND test_qopenpcdet_helpers)
cloudviewer_set_plugin_test_env(test_qopenpcdet_helpers)
