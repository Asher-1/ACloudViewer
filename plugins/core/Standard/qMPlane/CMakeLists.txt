cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

# Add an option to CMake to control whether we build this plugin or not
option( PLUGIN_STANDARD_QMPLANE "Check to install QMPlane_PLUGIN plugin" OFF )

if ( PLUGIN_STANDARD_QMPLANE )

    project( QMPLANE_PLUGIN )
    
	AddPlugin( NAME ${PROJECT_NAME} )

    # Compatibility switch; BUILD_UNIT_TESTS also enables this suite.
    option( COMPILE_QMPLANE_PLUGIN_TESTS "Build qMPlane unit tests" OFF )
    
	include_directories( ${ACloudViewer_SOURCE_DIR} )
    
	target_sources( ${PROJECT_NAME}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/qMPlane.h
			${CMAKE_CURRENT_SOURCE_DIR}/qMPlane.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/MPlaneCommands.h
			${CMAKE_CURRENT_SOURCE_DIR}/MPlaneCommands.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/qMPlane.qrc
	)

    # Add plugin ui
    add_subdirectory( ui )
    
    # Add plugin sources
	add_subdirectory( src )
    
    if ( BUILD_UNIT_TESTS OR COMPILE_QMPLANE_PLUGIN_TESTS )
        # Add plugin tests through the shared root testing configuration.
        add_subdirectory(tests)
    endif()
endif()
