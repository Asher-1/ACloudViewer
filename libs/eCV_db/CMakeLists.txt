cmake_minimum_required(VERSION 3.0)

include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )

project(ECV_DB_LIB )

file( GLOB header_list *.h)
file( GLOB source_list *.cpp)
file(GLOB         RPLY_SOURCE_FILES "../../3rdparty/rply/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../3rdparty/liblzf/liblzf/*.c")

add_library(${PROJECT_NAME} SHARED 
			${header_list} 
			${source_list} 
			${RPLY_SOURCE_FILES}
			${LIBLZF_SOURCE_FILES})

# Enforce 3rd party dependencies
add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
target_link_libraries( ${PROJECT_NAME} ${3RDPARTY_LIBRARIES} )

# INTERNAL LIB
target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Widgets Qt5::OpenGL)

# Add custom preprocessor definitions
if (WIN32)
	set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS CV_USE_AS_DLL ECV_DB_USE_AS_DLL )
endif()

# install (shared) lib to specified destinations
if( WIN32 )
	foreach( dest ${INSTALL_DESTINATIONS} )
		install_shared( ${PROJECT_NAME} ${dest} 1 )
	endforeach()
else()
	install_shared( ${PROJECT_NAME} ${CMAKE_INSTALL_LIBDIR}/ErowCloudViewer 0 ) #default destination: /usr/lib
endif()
