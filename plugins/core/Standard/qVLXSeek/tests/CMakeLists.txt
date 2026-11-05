if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qVLXSeek tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qvlxseek_helpers test_vlxseek_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/VLXSeekModelCatalog.cpp
)

target_include_directories(test_qvlxseek_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
    ${CMAKE_SOURCE_DIR}/core/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qvlxseek_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qvlxseek_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qvlxseek_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qvlxseek_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qvlxseek_helpers)
add_test(NAME test_qvlxseek_helpers COMMAND test_qvlxseek_helpers)
cloudviewer_set_plugin_test_env(test_qvlxseek_helpers)

# Cold-start evidence for the shared VlxSeek test-data chain (download ->
# digest verification -> extraction -> demo fill). Needs the site proxy for
# the first download; skips honestly without one.
add_executable(test_qvlxseek_testdata test_vlxseek_testdata.cpp)

target_include_directories(test_qvlxseek_testdata PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qvlxseek_testdata PRIVATE
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
    target_compile_options(test_qvlxseek_testdata PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qvlxseek_testdata PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qvlxseek_testdata PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qvlxseek_testdata)
add_test(NAME test_qvlxseek_testdata COMMAND test_qvlxseek_testdata)
cloudviewer_set_plugin_test_env(test_qvlxseek_testdata)
