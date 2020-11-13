project( CVPluginAPI )

add_library( ${PROJECT_NAME} SHARED )

find_package( Qt5
	COMPONENTS
		Gui
	REQUIRED
)

add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( ui )

# We only need the includes from the stub
get_target_property( STUB_INCLUDES CVPluginStub INCLUDE_DIRECTORIES )

target_include_directories( ${PROJECT_NAME} PUBLIC "${STUB_INCLUDES}" )

target_link_libraries( ${PROJECT_NAME}
	ECV_DB_LIB
	ECV_IO_LIB
)

if (USE_PCL_BACKEND)
	target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
endif()

target_compile_definitions( ${PROJECT_NAME} PRIVATE CVPLUGIN_API_LIBRARY_BUILD )

set_target_properties( ${PROJECT_NAME} PROPERTIES
	AUTOUIC ON # FIXME Remove after everything has moved to targets and we can set it globally
	CXX_VISIBILITY_PRESET hidden
)

InstallSharedLibrary( TARGET ${PROJECT_NAME} )
