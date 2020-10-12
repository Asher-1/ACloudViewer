project( ECV_DB_LIB )

file(GLOB         RPLY_SOURCE_FILES "../../3rdparty/rply/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../3rdparty/liblzf/liblzf/*.c")

add_library(${PROJECT_NAME} SHARED 
			${RPLY_SOURCE_FILES}
			${LIBLZF_SOURCE_FILES})

add_subdirectory( include )
add_subdirectory( src )

if (CVCORELIB_USE_TBB)
	include_directories( ${TBB_INCLUDE_DIRS})
endif()

# INTERNAL LIB
target_link_libraries( ${PROJECT_NAME} CVCoreLib )
# Qt
target_link_libraries(${PROJECT_NAME} Qt5::OpenGL)

# Enforce 3rd party dependencies
add_definitions( -DPOINTSWITHINDEX )
add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
target_link_libraries( ${PROJECT_NAME} ${3RDPARTY_LIBRARIES} )

# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PUBLIC ECV_DB_LIBRARY_BUILD )

set_target_properties( ${PROJECT_NAME} PROPERTIES
	CXX_VISIBILITY_PRESET hidden
)

# install (shared) lib to specified destinations
InstallSharedLibrary( TARGET ${PROJECT_NAME} )

include( cmake/InstallCGALDependencies.cmake )