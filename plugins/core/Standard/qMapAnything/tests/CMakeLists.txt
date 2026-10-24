add_executable( test_qmapanything_helpers test_map_anything_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingModelCatalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingReconstruction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingSceneMeta.cpp
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

# GUI/headless closure harness: the REAL MapAnythingWorker (dialog thread)
# writing the merged cloud in the headless CLOUD_PLY byte layout. Driven by
# check_gui_headless_parity.py against -MAP_ANYTHING_RECONSTRUCT (needs a
# GGUF + images, so not a ctest).
add_executable( map_anything_worker_parity_harness worker_parity_harness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/MapAnythingWorker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingWorker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingModelCatalog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingReconstruction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/MapAnythingSceneMeta.cpp
)
set_target_properties( map_anything_worker_parity_harness PROPERTIES AUTOMOC ON )
target_include_directories( map_anything_worker_parity_harness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/core/AICore/include
    ${CMAKE_SOURCE_DIR}/libs/CVPluginAPI/include
)
target_compile_definitions( map_anything_worker_parity_harness PRIVATE AICore_ENABLED )
target_link_libraries( map_anything_worker_parity_harness PRIVATE
    AICore
    # MapAnythingGeometry.cpp includes <Eigen/Dense>; on Linux the docker
    # image resolves it from the default /usr/include/eigen3 path, but MSVC
    # has no default Eigen location (measured: C1083 on windows-2022).
    Eigen3::Eigen
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Concurrent
    Threads::Threads
)
if( NOT MSVC )
    target_compile_options( map_anything_worker_parity_harness PRIVATE -Wall -Wextra -fPIC )
    set_target_properties( map_anything_worker_parity_harness PROPERTIES
        POSITION_INDEPENDENT_CODE ON )
    if( UNIX AND NOT APPLE )
        target_link_options( map_anything_worker_parity_harness PRIVATE "-no-pie" )
    endif()
endif()
cloudviewer_set_bin_runtime_output_directory( map_anything_worker_parity_harness )
