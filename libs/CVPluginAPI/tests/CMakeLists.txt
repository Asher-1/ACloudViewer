# Focused unit tests for the shared AICore UI helper header
# (include/ecvAICoreUiHelper.h). Header-only: links Qt Widgets only, runs
# headless (QT_QPA_PLATFORM=offscreen is forced inside the test binary).

if(NOT TARGET Qt${QT_VERSION_MAJOR}::Widgets)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets)
endif()

add_executable(test_cvpluginapi_ui_helpers test_ui_helpers.cpp)

target_include_directories(test_cvpluginapi_ui_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(test_cvpluginapi_ui_helpers PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
)

if(NOT MSVC)
    target_compile_options(test_cvpluginapi_ui_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_cvpluginapi_ui_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_cvpluginapi_ui_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_ui_helpers)
add_test(NAME test_cvpluginapi_ui_helpers COMMAND test_cvpluginapi_ui_helpers)
set_tests_properties(test_cvpluginapi_ui_helpers PROPERTIES LABELS "cvpluginapi")

# Focused unit tests for the shared AICore runtime helper header
# (include/ecvAICoreRuntimeHelpers.h). Header-only under test, but the lock
# helper exercises the real exported C API, so link libAICore.
if(TARGET AICore)
add_executable(test_cvpluginapi_runtime_helpers test_runtime_helpers.cpp)

target_include_directories(test_cvpluginapi_runtime_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(test_cvpluginapi_runtime_helpers PRIVATE
    AICore
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
)

if(NOT MSVC)
    target_compile_options(test_cvpluginapi_runtime_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_cvpluginapi_runtime_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_cvpluginapi_runtime_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_runtime_helpers)
add_test(NAME test_cvpluginapi_runtime_helpers COMMAND test_cvpluginapi_runtime_helpers)
set_tests_properties(test_cvpluginapi_runtime_helpers PROPERTIES LABELS "cvpluginapi")
endif()
