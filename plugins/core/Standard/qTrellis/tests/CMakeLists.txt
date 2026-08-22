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
    target_link_options(test_qtrellis_catalog PRIVATE "-no-pie")
endif()

cloudviewer_set_bin_runtime_output_directory(test_qtrellis_catalog)
add_test(NAME test_qtrellis_catalog COMMAND test_qtrellis_catalog)
