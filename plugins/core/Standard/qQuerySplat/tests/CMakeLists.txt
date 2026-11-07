# qQuerySplat plugin tests (googletest, sources compiled in like the
# qMapAnything helpers target: the plugin target carries moc wiring that the
# test binary does not need).

add_executable( test_qquerysplat_helpers test_querysplat_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/QuerySplatModelCatalog.cpp
)

target_include_directories( test_qquerysplat_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package( Threads REQUIRED )

target_link_libraries( test_qquerysplat_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Threads::Threads
)

if( NOT MSVC )
    target_compile_options( test_qquerysplat_helpers PRIVATE -Wall -Wextra -fPIC )
    set_target_properties( test_qquerysplat_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON )
    if( UNIX AND NOT APPLE )
        target_link_options( test_qquerysplat_helpers PRIVATE "-no-pie" )
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory( test_qquerysplat_helpers )
add_test( NAME test_qquerysplat_helpers COMMAND test_qquerysplat_helpers )
cloudviewer_set_plugin_test_env(test_qquerysplat_helpers)
