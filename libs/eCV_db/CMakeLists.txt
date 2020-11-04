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
target_link_libraries( ${PROJECT_NAME} PUBLIC CVCoreLib )
# Qt
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::OpenGL)

# Enforce 3rd party dependencies
cloudViewer_link_3rdparty_libraries(${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PRIVATE ${QRANSAC_INCLUDE_DIRS})

# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PUBLIC ECV_DB_LIBRARY_BUILD )

set_target_properties( ${PROJECT_NAME} PROPERTIES
	CXX_VISIBILITY_PRESET hidden
)

add_custom_command(TARGET ${PROJECT_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${PROJECT_NAME}>" ${CLOUDVIEWER_OUTPUT_DIRECTORY}
	VERBATIM
	)

# install (shared) lib to specified destinations
InstallSharedLibrary( TARGET ${PROJECT_NAME} )

include( cmake/InstallCGALDependencies.cmake )