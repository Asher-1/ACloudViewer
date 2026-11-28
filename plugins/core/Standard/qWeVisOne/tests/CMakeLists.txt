add_executable(test_qwevisone_dialog test_wevisone_dialog.cpp
    ../src/WeVisOneDialog.cpp ../src/WeVisOneModelCatalog.cpp
    ../src/WeVisOneWorker.cpp ../include/WeVisOneDialog.h
    ../include/WeVisOneWorker.h)
set_target_properties(test_qwevisone_dialog PROPERTIES AUTOMOC ON
    CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_include_directories(test_qwevisone_dialog PRIVATE ../include)
target_compile_definitions(test_qwevisone_dialog PRIVATE AICore_ENABLED)
target_link_libraries(test_qwevisone_dialog PRIVATE
    AICore CVPluginAPI Qt::Core Qt::Gui Qt::Widgets Qt::Network)
if(NOT MSVC)
    target_compile_options(test_qwevisone_dialog PRIVATE -fPIC)
    if(NOT APPLE)
        target_link_options(test_qwevisone_dialog PRIVATE -no-pie)
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(test_qwevisone_dialog)
target_link_libraries(test_qwevisone_dialog PRIVATE 3rdparty_googletest)
cloudviewer_add_test(NAME test_qwevisone_dialog COMMAND test_qwevisone_dialog
    LABELS cvpluginapi wevisone capi)
cloudviewer_set_plugin_test_env_offscreen(test_qwevisone_dialog)

add_executable(test_qwevisone_helpers test_wevisone_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/WeVisOneModelCatalog.cpp
)

target_include_directories(test_qwevisone_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
    ${CMAKE_SOURCE_DIR}/core/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qwevisone_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qwevisone_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qwevisone_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qwevisone_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qwevisone_helpers)
cloudviewer_add_test(NAME test_qwevisone_helpers COMMAND test_qwevisone_helpers
    LABELS wevisone)
cloudviewer_set_plugin_test_env(test_qwevisone_helpers)

# Cold-start evidence for the shared WeVisOne test-data chain (download ->
# digest verification -> extraction -> demo fill). Needs network access to
# huggingface.co for the first download; skips honestly without a proxy on
# proxied sites.
add_executable(test_qwevisone_testdata test_wevisone_testdata.cpp)

target_include_directories(test_qwevisone_testdata PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qwevisone_testdata PRIVATE
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
    target_compile_options(test_qwevisone_testdata PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qwevisone_testdata PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qwevisone_testdata PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qwevisone_testdata)
cloudviewer_add_test(NAME test_qwevisone_testdata COMMAND test_qwevisone_testdata
    LABELS wevisone network)
cloudviewer_set_plugin_test_env(test_qwevisone_testdata)
