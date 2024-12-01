cmake_minimum_required( VERSION 3.10 )

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
	
	# find_package(Eigen3 QUIET)
	# if(NOT Eigen3_FOUND)
	# 	set( EIGEN_ROOT_DIR "" CACHE PATH "Eigen root (contains the Eigen directory)" )
	# 	if ( NOT EIGEN_ROOT_DIR )
	# 		message( SEND_ERROR "No Eigen root directory specified (EIGEN_ROOT_DIR)" )
	# 	endif()
	# 	include_directories(${EIGEN_ROOT_DIR})		
	# else()
	# 	target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
	# endif()

	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )

endif()
