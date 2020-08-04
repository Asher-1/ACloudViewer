cmake_minimum_required(VERSION 3.0)

include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
include_directories(${ECV_DB_LIB_SOURCE_DIR})
if( MSVC )
   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

project( ECV_IO_LIB )

file( GLOB header_list *.h)
file( GLOB source_list *.cpp *.c)
file( GLOB ui_list ui_templates/*.ui )

set( CMAKE_CXX_VISIBILITY_PRESET hidden )

# PDMS support
if( ${OPTION_SUPPORT_MAC_PDMS_FORMAT} )
	file( GLOB PDMS_header_list PDMS/*.h )
	file( GLOB PDMS_source_list PDMS/*.cpp )
	list( APPEND header_list ${PDMS_header_list} )
	list( APPEND source_list ${PDMS_source_list} )
endif()

qt5_wrap_ui( generated_ui_list ${ui_list} )
add_library( ${PROJECT_NAME} SHARED ${header_list} ${source_list} ${generated_ui_list} )

# Enforce 3rd party dependencies
file(GLOB         RPLY_SOURCE_FILES "../../../3rdparty/rply/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../../3rdparty/liblzf/liblzf/*.c")
add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
target_link_libraries( ${PROJECT_NAME} ${3RDPARTY_LIBRARIES} )

# INTERNAL LIB
target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )
target_link_libraries( ${PROJECT_NAME} ECV_DB_LIB )

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core)

# contrib. libraries support
target_link_contrib( ${PROJECT_NAME} )

# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_IO_LIBRARY_BUILD )

if( APPLE )
	set_target_properties( ${PROJECT_NAME} PROPERTIES
		INSTALL_RPATH "." )
endif()

# install (shared) lib to specified destinations
if( WIN32 OR APPLE )
	foreach( dest ${INSTALL_DESTINATIONS} )
		install_shared( ${PROJECT_NAME} ${dest} 1 )
	endforeach()
else()
	install_shared( ${PROJECT_NAME} ${CMAKE_INSTALL_LIBDIR}/ErowCloudViewer 0 ) #default destination: /usr/lib
endif()

if(BUILD_TESTING)
	add_subdirectory(Tests)
endif()