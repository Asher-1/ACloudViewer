if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qRMBG tests: googletest not available — skipping")
    return()
endif()

add_executable(test_qrmbg_helpers test_rmbg_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/RMBGModelCatalog.cpp
)

target_include_directories(test_qrmbg_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

target_link_libraries(test_qrmbg_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Sql
    Threads::Threads
)

if(NOT MSVC)
    target_compile_options(test_qrmbg_helpers PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_qrmbg_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_options(test_qrmbg_helpers PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qrmbg_helpers)
add_test(NAME test_qrmbg_helpers COMMAND test_qrmbg_helpers)
