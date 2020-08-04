cmake_minimum_required(VERSION 3.0)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )

project( PCV_LIB )

# no Qt stuff here
set( CMAKE_AUTOMOC OFF )
set( CMAKE_AUTOUIC OFF )
set( CMAKE_AUTORCC OFF )

file( GLOB header_list *.h )
file( GLOB source_list *.cpp )

add_library( ${PROJECT_NAME} STATIC ${header_list} ${source_list} )

target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )

# Add preprocessor definitions
set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS NOMINMAX _CRT_SECURE_NO_WARNINGS )
if (WIN32)
	set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS CV_USE_AS_DLL )
endif()
