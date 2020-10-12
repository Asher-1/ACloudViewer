# Add an option to CMake to control whether we build this plugin or not
option( PLUGIN_STANDARD_QMPLANE "Check to install QMPlane_PLUGIN plugin" OFF )

if ( PLUGIN_STANDARD_QMPLANE )

    project( QMPLANE_PLUGIN )
    
	AddPlugin( NAME ${PROJECT_NAME} )

    # Option for building TESTS
    option( COMPILE_QMPLANE_PLUGIN_TESTS "Check to enable REST support" OFF ) 
    
	include_directories( ${ErowCloudViewer_SOURCE_DIR} )
    
    # Set path to external dependencies
    SET( EXTERNAL_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/external/ )
    
	target_sources( ${PROJECT_NAME}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/qMPlane.h
			${CMAKE_CURRENT_SOURCE_DIR}/qMPlane.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/qMPlane.qrc
	)

    # Add plugin ui
    add_subdirectory( ui )
    
    # Add plugin sources
	add_subdirectory( src )
    
    if ( COMPILE_QMPLANE_PLUGIN_TESTS )
        # Add plugin tests
        add_subdirectory(Tests)
    endif()
endif()
