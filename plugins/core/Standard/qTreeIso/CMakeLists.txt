cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QTREEISO "Install qTreeIso plugin" OFF )

if( PLUGIN_STANDARD_QTREEISO )
	
	project( QTREEISO_PLUGIN )
	
	AddPlugin( NAME ${PROJECT_NAME} )	

	if (APPLE OR WIN32 OR ${GLIBCXX_USE_CXX11_ABI})
    find_package(Boost REQUIRED COMPONENTS
                graph)
	endif ()

	if(NOT Boost_FOUND)
		set( BOOST_ROOT_DIR "" CACHE PATH "Boost root (contains the Boost directory)" )
		if ( NOT BOOST_ROOT_DIR )
			message( SEND_ERROR "No Boost directory specified (BOOST_ROOT_DIR)" )
		endif()
		include_directories(${BOOST_ROOT_DIR})
	else()
		target_link_libraries(${PROJECT_NAME} Boost::graph)
	endif()

	target_compile_definitions( ${PROJECT_NAME} PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS )
	target_link_libraries(${PROJECT_NAME} 3rdparty_eigen3)
	
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )

endif()
