# Focused unit tests for the shared AICore UI helper header
# (include/ecvAICoreUiHelper.h). Header-only: links Qt Widgets only, runs
# headless (QT_QPA_PLATFORM=offscreen is forced inside the test binary).

if(NOT TARGET Qt${QT_VERSION_MAJOR}::Widgets)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets)
endif()

function(_cvpluginapi_register_asset_test name)
    if(COMMAND aicore_register_test)
        # Include cache contracts in the existing native CI contract gate.
        aicore_register_test(NAME ${name} TARGET ${name}
            LABELS "cvpluginapi;capi")
    else()
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES LABELS "cvpluginapi")
    endif()
    cloudviewer_set_plugin_test_env(${name})
endfunction()

add_executable(test_cvpluginapi_ui_helpers test_ui_helpers.cpp)

target_include_directories(test_cvpluginapi_ui_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/core/include
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
    if(NOT APPLE)
        target_link_options(test_cvpluginapi_ui_helpers PRIVATE "-no-pie")
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_ui_helpers)
add_test(NAME test_cvpluginapi_ui_helpers COMMAND test_cvpluginapi_ui_helpers)
cloudviewer_set_plugin_test_env(test_cvpluginapi_ui_helpers)
set_tests_properties(test_cvpluginapi_ui_helpers PROPERTIES LABELS "cvpluginapi")

# Exercise the real shared cache ledger used by model-downloading plugins.
add_executable(test_cvpluginapi_asset_integrity test_asset_integrity.cpp)
target_include_directories(test_cvpluginapi_asset_integrity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/core/AICore/include)
target_link_libraries(test_cvpluginapi_asset_integrity PRIVATE
    CVPluginAPI Qt${QT_VERSION_MAJOR}::Core)
if(NOT MSVC)
    target_compile_options(test_cvpluginapi_asset_integrity PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_cvpluginapi_asset_integrity PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    if(NOT APPLE)
        target_link_options(test_cvpluginapi_asset_integrity PRIVATE "-no-pie")
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_asset_integrity)
_cvpluginapi_register_asset_test(test_cvpluginapi_asset_integrity)

# Real HTTP transport with a local server; no external assets or network.
add_executable(test_cvpluginapi_model_downloader test_model_downloader.cpp)
target_include_directories(test_cvpluginapi_model_downloader PRIVATE
    ${CMAKE_SOURCE_DIR}/core/AICore/include)
target_link_libraries(test_cvpluginapi_model_downloader PRIVATE
    CVPluginAPI Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Network)
if(NOT MSVC)
    target_compile_options(test_cvpluginapi_model_downloader PRIVATE -Wall -Wextra -fPIC)
    if(NOT APPLE)
        target_link_options(test_cvpluginapi_model_downloader PRIVATE "-no-pie")
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_model_downloader)
_cvpluginapi_register_asset_test(test_cvpluginapi_model_downloader)

# Cached archive preparation and completion state, without external assets.
add_executable(test_cvpluginapi_dataset_repository test_dataset_repository.cpp)
target_include_directories(test_cvpluginapi_dataset_repository PRIVATE
    ${CMAKE_SOURCE_DIR}/core/AICore/include)
target_link_libraries(test_cvpluginapi_dataset_repository PRIVATE
    CVPluginAPI Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Network)
if(NOT MSVC)
    target_compile_options(test_cvpluginapi_dataset_repository PRIVATE -Wall -Wextra -fPIC)
    if(NOT APPLE)
        target_link_options(test_cvpluginapi_dataset_repository PRIVATE "-no-pie")
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_dataset_repository)
_cvpluginapi_register_asset_test(test_cvpluginapi_dataset_repository)

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
    if(NOT APPLE)
        target_link_options(test_cvpluginapi_runtime_helpers PRIVATE "-no-pie")
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory(test_cvpluginapi_runtime_helpers)
add_test(NAME test_cvpluginapi_runtime_helpers COMMAND test_cvpluginapi_runtime_helpers)
cloudviewer_set_plugin_test_env(test_cvpluginapi_runtime_helpers)
set_tests_properties(test_cvpluginapi_runtime_helpers PROPERTIES LABELS "cvpluginapi")
endif()
