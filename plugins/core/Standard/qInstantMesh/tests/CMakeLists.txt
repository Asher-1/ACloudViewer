find_package(Threads REQUIRED)
add_executable(test_qinstantmesh_worker test_instantmesh_worker.cpp
    ../src/InstantMeshWorker.cpp ../src/InstantMeshModelCatalog.cpp
    ../include/InstantMeshWorker.h)
set_target_properties(test_qinstantmesh_worker PROPERTIES
    AUTOMOC ON CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_include_directories(test_qinstantmesh_worker PRIVATE ../include)
target_compile_definitions(test_qinstantmesh_worker PRIVATE AICore_ENABLED)
target_link_libraries(test_qinstantmesh_worker PRIVATE
    AICore CVPluginAPI CVCoreLib Qt::Core Qt::Gui Threads::Threads)
if(NOT MSVC)
    target_compile_options(test_qinstantmesh_worker PRIVATE -fPIC)
    target_link_options(test_qinstantmesh_worker PRIVATE -no-pie)
endif()
cloudviewer_set_bin_runtime_output_directory(test_qinstantmesh_worker)
if(COMMAND aicore_register_test)
    aicore_register_test(NAME test_qinstantmesh_worker TARGET test_qinstantmesh_worker
        LABELS "capi;instantmesh;cvpluginapi")
else()
    add_test(NAME test_qinstantmesh_worker COMMAND test_qinstantmesh_worker)
endif()
cloudviewer_set_plugin_test_env(test_qinstantmesh_worker)

add_executable(test_qinstantmesh_dialog test_instantmesh_dialog.cpp
    ../src/InstantMeshDialog.cpp ../src/InstantMeshModelCatalog.cpp
    ../include/InstantMeshDialog.h)
add_executable(test_qinstantmesh_plugin test_instantmesh_plugin.cpp
    ../src/qInstantMesh.cpp ../src/InstantMeshDialog.cpp
    ../src/InstantMeshWorker.cpp ../src/InstantMeshModelCatalog.cpp
    ../include/qInstantMesh.h ../include/InstantMeshDialog.h
    ../include/InstantMeshWorker.h ../qInstantMesh.qrc)
foreach(test IN ITEMS test_qinstantmesh_dialog test_qinstantmesh_plugin)
    set_target_properties(${test} PROPERTIES AUTOMOC ON AUTORCC ON
        CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${test} PRIVATE ../include)
    target_compile_definitions(${test} PRIVATE AICore_ENABLED)
    target_link_libraries(${test} PRIVATE AICore CVPluginAPI CVPluginStub
        CVCoreLib CV_DB_LIB CV_IO_LIB Qt::Core Qt::Gui Qt::Widgets
        Qt::Network Threads::Threads)
    if(NOT MSVC)
        target_compile_options(${test} PRIVATE -fPIC)
        target_link_options(${test} PRIVATE -no-pie)
    endif()
    cloudviewer_set_bin_runtime_output_directory(${test})
    if(COMMAND aicore_register_test)
        aicore_register_test(NAME ${test} TARGET ${test}
            LABELS "capi;instantmesh;cvpluginapi")
    else()
        add_test(NAME ${test} COMMAND ${test})
    endif()
    cloudviewer_set_plugin_test_env(${test})
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endforeach()

# Real-model batch host gate; packaged CLI/RPC children require separate L3/L4 validation.
if(AICORE_TEST_ASSET_ROOT)
    get_filename_component(_instantmesh_data_root "${AICORE_TEST_ASSET_ROOT}" DIRECTORY)
    add_test(NAME test_qinstantmesh_batch COMMAND test_qinstantmesh_plugin
        "${_instantmesh_data_root}" "${AICORE_TEST_DEVICE}")
    cloudviewer_set_plugin_test_env_offscreen(test_qinstantmesh_batch)
    set_tests_properties(test_qinstantmesh_batch PROPERTIES
        LABELS "capi;instantmesh;model;e2e;cvpluginapi" RESOURCE_LOCK gpu)
    if(TARGET aicore-model-tests)
        add_dependencies(aicore-model-tests test_qinstantmesh_plugin)
    endif()
endif()

if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qInstantMesh tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qinstantmesh_catalog test_instantmesh_catalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/InstantMeshModelCatalog.cpp
)

target_include_directories(test_qinstantmesh_catalog PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qinstantmesh_catalog PRIVATE
    CVCoreLib
    CVPluginAPI
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qinstantmesh_catalog PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qinstantmesh_catalog PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qinstantmesh_catalog PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qinstantmesh_catalog)
add_test(NAME test_qinstantmesh_catalog COMMAND test_qinstantmesh_catalog)
cloudviewer_set_plugin_test_env(test_qinstantmesh_catalog)
