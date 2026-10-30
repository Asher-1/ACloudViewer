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
