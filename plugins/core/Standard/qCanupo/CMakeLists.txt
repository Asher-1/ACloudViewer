option( PLUGIN_STANDARD_QCANUPO "Check to install qCanupo plugin" OFF )

if( PLUGIN_STANDARD_QCANUPO )
	project( QCANUPO_PLUGIN )
	
	AddPlugin( NAME ${PROJECT_NAME} )

	include_directories( DLIB_ROOT )

	# set( DLIB_ROOT "" CACHE PATH "DLib library root directory" )
	# if ( NOT DLIB_ROOT )
		# message( SEND_ERROR "No DLib library root specified (DLIB_ROOT)" )
	# else()
		# include_directories( ${DLIB_ROOT} )
	# endif()
	
	option( QCANUPO_COMPILE_PRIVATE_METHODS "Compile qCanupo plugin with private methods (CNRS/UEB only)" OFF )
	if( QCANUPO_COMPILE_PRIVATE_METHODS )
		target_compile_definitions( ${PROJECT_NAME} PRIVATE COMPILE_PRIVATE_CANUPO )
	endif()
	
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )

endif()
