cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

project(QMPLANE_PLUGIN_TESTS)

add_executable(
    QMPLANE_PLUGIN_TESTS 
    
    # Sources to be tested
    ../src/ccItemTree.h
    ../src/ccItemTree.cpp
    ../src/ccMPlaneErrors.h
    
    # Testing sources
    main.cpp
    item_tree_test.cpp
    mocks/ecvMainAppInterfaceMock.h
    mocks/ccGenericGLDisplayMock.h
)

target_include_directories(QMPLANE_PLUGIN_TESTS PRIVATE ${EXTERNAL_TOOLS})
target_include_directories(QMPLANE_PLUGIN_TESTS PRIVATE ../src/)

target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${EXTERNAL_LIBS_INCLUDE_DIR} )

target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${ACloudViewerPlugins_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${ACloudViewer_SOURCE_DIR}/../common )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CVCoreLib_SOURCE_DIR}/include )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${ECV_IO_LIB_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${ECV_DB_LIB_SOURCE_DIR} )
if( MSVC )
    target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()

target_link_libraries( QMPLANE_PLUGIN_TESTS CVCoreLib )
target_link_libraries( QMPLANE_PLUGIN_TESTS ECV_DB_LIB )
target_link_libraries( QMPLANE_PLUGIN_TESTS ECV_IO_LIB )

install( TARGETS QMPLANE_PLUGIN_TESTS DESTINATION ${ACloudViewer_DEST_FOLDER})
