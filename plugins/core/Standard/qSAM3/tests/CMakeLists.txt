if(NOT TARGET 3rdparty_googletest)
    message(STATUS "qSAM3 tests: googletest not available — skipping")
    return()
endif()

add_executable(test_sam3_contract test_sam3_contract.cpp)

target_include_directories(test_sam3_contract PRIVATE
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package(Threads REQUIRED)

# AICore loads ggml backends dynamically; at test link time we need ggml
# explicitly so the test binary resolves the backend init symbols.
if(TARGET ggml::all)
    set(_ggml_target ggml::all)
else()
    set(_ggml_target "")
endif()

target_link_libraries(test_sam3_contract PRIVATE
    AICore
    3rdparty_googletest
    gtest_main
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Threads::Threads
    ${_ggml_target}
)

if(NOT MSVC)
    target_compile_options(test_sam3_contract PRIVATE -Wall -Wextra -fPIC)
    set_target_properties(test_sam3_contract PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    # GNU ld only; Apple ld64 does not know --allow-shlib-undefined.
    if(UNIX AND NOT APPLE)
        target_link_options(test_sam3_contract PRIVATE "-Wl,--allow-shlib-undefined")
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory(test_sam3_contract)
add_test(NAME test_sam3_contract COMMAND test_sam3_contract)