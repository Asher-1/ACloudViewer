project( CVAppCommon )

#include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
#include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
#if( MSVC )
#   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
#endif()
#include_directories( ${ECV_IO_LIB_SOURCE_DIR} )

add_library( ${PROJECT_NAME} SHARED )

#add_subdirectory( devices )
add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( ui )

target_link_libraries( ${PROJECT_NAME}
	CVPluginAPI
)

target_compile_definitions( ${PROJECT_NAME} PRIVATE CVAPPCOMMON_LIBRARY_BUILD )
# if ( USE_PCL_BACKEND )
	# target_compile_definitions( ${PROJECT_NAME} PUBLIC ECV_PCL_ENGINE_LIBRARY_BUILD )	
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
