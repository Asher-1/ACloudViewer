cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project( CVPluginAPI )

add_library( ${PROJECT_NAME} SHARED )

# Qt5/Qt6 compatibility
# Note: Gui is already found in CMakeExternalLibs.cmake

add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( ui )

# We only need the includes from the stub
get_target_property( STUB_INCLUDES CVPluginStub INCLUDE_DIRECTORIES )

target_include_directories( ${PROJECT_NAME} PUBLIC "${STUB_INCLUDES}" )

target_link_libraries( ${PROJECT_NAME}
	CV_DB_LIB
	CV_IO_LIB
)

target_compile_definitions( ${PROJECT_NAME} PRIVATE CVPLUGIN_API_LIBRARY_BUILD )

set_target_properties( ${PROJECT_NAME} PROPERTIES
	AUTOUIC ON # FIXME Remove after everything has moved to targets and we can set it globally
	CXX_VISIBILITY_PRESET hidden
)

if (NOT APPLE) # no need to copy targets on macos due to rpath
	# install (shared) lib to specified destinations
	InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()