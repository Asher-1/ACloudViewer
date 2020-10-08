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
    mocks/ccMainAppInterfaceMock.h
    mocks/ccGenericGLDisplayMock.h
)

target_include_directories(QMPLANE_PLUGIN_TESTS PRIVATE ${EXTERNAL_TOOLS})
target_include_directories(QMPLANE_PLUGIN_TESTS PRIVATE ../src/)

target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${QCC_GL_LIB_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${EXTERNAL_LIBS_INCLUDE_DIR} )

target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CloudComparePlugins_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CloudCompare_SOURCE_DIR}/../common )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CC_CORE_LIB_SOURCE_DIR}/include )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${CC_FBO_LIB_SOURCE_DIR}/include )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${QCC_IO_LIB_SOURCE_DIR} )
target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${QCC_DB_LIB_SOURCE_DIR} )
if( MSVC )
    target_include_directories( QMPLANE_PLUGIN_TESTS PRIVATE ${QCC_DB_LIB_SOURCE_DIR}/msvc )
endif()

target_link_libraries( QMPLANE_PLUGIN_TESTS CC_FBO_LIB )
target_link_libraries( QMPLANE_PLUGIN_TESTS CC_CORE_LIB )
target_link_libraries( QMPLANE_PLUGIN_TESTS QCC_DB_LIB )
target_link_libraries( QMPLANE_PLUGIN_TESTS QCC_IO_LIB )
target_link_libraries( QMPLANE_PLUGIN_TESTS QCC_GL_LIB )


install( TARGETS QMPLANE_PLUGIN_TESTS DESTINATION ${CLOUDCOMPARE_DEST_FOLDER})