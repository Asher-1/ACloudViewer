project( CVAppCommon )

add_library( ${PROJECT_NAME} SHARED )

#add_subdirectory( devices )
add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( ui )

target_link_libraries( ${PROJECT_NAME}
	CVPluginAPI
)

if (USE_PCL_BACKEND)
	if( CMAKE_VERSION VERSION_LESS 3.13 )
		link_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR} )
	else()
		target_link_directories( ${PROJECT_NAME}
			PUBLIC ${QPCL_ENGINE_LIB_SOURCE_DIR}
		)
	endif()
	include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR})
	target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
endif()

target_compile_definitions( ${PROJECT_NAME} PRIVATE CVAPPCOMMON_LIBRARY_BUILD )
# if ( USE_PCL_BACKEND )
	# target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_PCL_BACKEND )	
# endif()

set_target_properties( ${PROJECT_NAME} PROPERTIES
	AUTOUIC ON # FIXME Remove after everything has moved to targets and we can set it globally
	CXX_VISIBILITY_PRESET hidden
)

# if( OPTION_SUPPORT_GAMEPADS )
	# find_package( Qt5
		# COMPONENTS
			# Gamepad
		# REQUIRED
	# )

	# target_link_libraries( ${PROJECT_NAME} Qt5::Gamepad )
# endif()

# if( OPTION_SUPPORT_3DCONNEXION_DEVICES )
	# target_link_3DXWARE( ${PROJECT_NAME} )
# endif()

InstallSharedLibrary( TARGET ${PROJECT_NAME} )
