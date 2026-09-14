if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qGKD tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qgkd_helpers test_qgkd_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/GKDModelCatalog.cpp
)

target_include_directories(test_qgkd_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qgkd_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Sql
    Threads::Threads
)
# The test defines its own main() (QGuiApplication for QPainter).

if(NOT MSVC)
    target_compile_options(test_qgkd_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qgkd_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qgkd_helpers PRIVATE "-no-pie")
    # Dynamic ggml backends: libAICore.so keeps backend symbols unresolved
    # until runtime dlopen (same allowance as the AICore C API tests).
    target_link_options(test_qgkd_helpers PRIVATE "LINKER:--allow-shlib-undefined")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qgkd_helpers)
add_test(NAME test_qgkd_helpers COMMAND test_qgkd_helpers)
