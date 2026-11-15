add_executable(test_qtrellis_model_selection test_model_selection.cpp
    ../src/TrellisDialog.cpp ../src/TrellisModelCatalog.cpp
    ../src/TrellisWorker.cpp ../include/TrellisDialog.h
    ../include/TrellisWorker.h)
set_target_properties(test_qtrellis_model_selection PROPERTIES AUTOMOC ON
    CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_include_directories(test_qtrellis_model_selection PRIVATE ../include)
target_compile_definitions(test_qtrellis_model_selection PRIVATE AICore_ENABLED)
target_link_libraries(test_qtrellis_model_selection PRIVATE
    AICore CVPluginAPI Qt::Core Qt::Gui Qt::Widgets Qt::Network)
if(NOT MSVC)
    target_compile_options(test_qtrellis_model_selection PRIVATE -fPIC)
    if(NOT APPLE)
        target_link_options(test_qtrellis_model_selection PRIVATE -no-pie)
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(test_qtrellis_model_selection)
if(COMMAND aicore_register_test)
    aicore_register_test(NAME test_qtrellis_model_selection
        TARGET test_qtrellis_model_selection LABELS "cvpluginapi;trellis;capi")
else()
    add_test(NAME test_qtrellis_model_selection COMMAND test_qtrellis_model_selection)
    set_tests_properties(test_qtrellis_model_selection PROPERTIES LABELS "cvpluginapi;trellis")
endif()
cloudviewer_set_plugin_test_env(test_qtrellis_model_selection)

if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qTrellis tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qtrellis_catalog test_trellis_catalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/TrellisModelCatalog.cpp
)

target_include_directories(test_qtrellis_catalog PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qtrellis_catalog PRIVATE
    CVCoreLib
    CVPluginAPI
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Sql
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qtrellis_catalog PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qtrellis_catalog PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    if(NOT APPLE)
        target_link_options(test_qtrellis_catalog PRIVATE "-no-pie")
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory(test_qtrellis_catalog)
add_test(NAME test_qtrellis_catalog COMMAND test_qtrellis_catalog)
cloudviewer_set_plugin_test_env(test_qtrellis_catalog)
