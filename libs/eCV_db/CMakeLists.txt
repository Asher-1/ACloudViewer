cmake_minimum_required( VERSION 3.10 )

include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )

if (CVCORELIB_USE_TBB)
	include_directories( ${TBB_INCLUDE_DIRS})
endif()

project( ECV_DB_LIB )

file( GLOB header_list *.h)
file( GLOB source_list *.cpp)
file(GLOB         RPLY_SOURCE_FILES "../../3rdparty/rply/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../3rdparty/liblzf/liblzf/*.c")

set( CMAKE_CXX_VISIBILITY_PRESET hidden )

add_library(${PROJECT_NAME} SHARED 
			${header_list} 
			${source_list} 
			${RPLY_SOURCE_FILES}
			${LIBLZF_SOURCE_FILES})

# Enforce 3rd party dependencies
add_definitions( -DPOINTSWITHINDEX )
add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
target_link_libraries( ${PROJECT_NAME} ${3RDPARTY_LIBRARIES} )

# INTERNAL LIB
target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Widgets Qt5::OpenGL)

# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PUBLIC CV_USE_AS_DLL )
target_compile_definitions( ${PROJECT_NAME} PUBLIC ECV_DB_LIBRARY_BUILD )

# install (shared) lib to specified destinations
InstallSharedLibrary( TARGET ${PROJECT_NAME} )