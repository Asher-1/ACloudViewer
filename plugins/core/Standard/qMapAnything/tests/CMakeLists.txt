add_executable( test_qmapanything_helpers test_map_anything_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingModelCatalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingGeometry.cpp
)

target_include_directories( test_qmapanything_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core/AICore/include
)

find_package( Threads REQUIRED )

target_link_libraries( test_qmapanything_helpers PRIVATE
    CVCoreLib
    AICore
    3rdparty_googletest
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Threads::Threads
)

if( NOT MSVC )
    target_compile_options( test_qmapanything_helpers PRIVATE -Wall -Wextra -fPIC )
    set_target_properties( test_qmapanything_helpers PROPERTIES
        POSITION_INDEPENDENT_CODE ON )
    if( UNIX AND NOT APPLE )
        target_link_options( test_qmapanything_helpers PRIVATE "-no-pie" )
    endif()
endif()

cloudviewer_set_bin_runtime_output_directory( test_qmapanything_helpers )
add_test( NAME test_qmapanything_helpers COMMAND test_qmapanything_helpers )
cloudviewer_set_plugin_test_env(test_qmapanything_helpers)
