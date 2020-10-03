# SPDX-License-Identifier: MIT
# Copyright © Andy Maloney <asmaloney@gmail.com>

cmake_minimum_required( VERSION 3.10 )

# Options
option( CVCORELIB_USE_CGAL "Check to compile CV_CORE_LIB with CGAL lib. (to enable Delaunay 2.5D triangulation with a GPL compliant licence)" OFF )
option( CVCORELIB_USE_TBB " Check to compile CV_CORE_LIB with Intel Threading Building Blocks lib (enables some parallel processing )" OFF )
option( CVCORELIB_SHARED "Check to compile CV_CORE_LIB as a shared library (DLL/so)" ON )
option( CVCORELIB_USE_QT_CONCURRENT "Compile CCCoreLib with QtConcurrent (to enable parallel processing)" ON)
# ScalarType as float (default) or double
option( CVCORELIB_SCALAR_DOUBLE "Check to define ScalarType as double" OFF )

if ( CVCORELIB_SCALAR_DOUBLE )
	add_definitions( -DSCALAR_TYPE_DOUBLE )
	set (USE_SCALAR_TYPE_DOUBLE ON PARENT_SCOPE)
else()
	add_definitions( -DSCALAR_TYPE_FLOAT )
endif()

# to compile CVLib only! (CMake implicitly imposes to declare a project before anything...)
project( CV_CORE_LIB 	
	DESCRIPTION
		"Core CloudCompare data structures & algorithms for handling point clouds"
	LANGUAGES
		CXX
	VERSION
		1.0
)

# CGAL takes precedence over triangle
if (CVCORELIB_USE_CGAL)
	include( cmake/CGALSupport.cmake )
endif()

if (CVCORELIB_USE_TBB)
	include( cmake/FindTBB.cmake )
else()
	if (NOT WIN32)
		message( WARNING "CVLib configured without parallel algorithm capabilities - see CVCORELIB_USE_TBB" )
	endif()
endif()

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

file( GLOB header_list include/*.h)
file( GLOB source_list src/*.cpp src/*.h)

if (CVCORELIB_SHARED)
	set( CMAKE_CXX_VISIBILITY_PRESET hidden )
	set( CMAKE_VISIBILITY_INLINES_HIDDEN 1 )
	add_library( ${PROJECT_NAME} SHARED ${header_list} ${source_list} )
else()
	add_library( ${PROJECT_NAME} STATIC ${header_list} ${source_list} )
	target_compile_definitions( ${PROJECT_NAME} PRIVATE CV_CORE_LIB_STATIC_DEFINE )	
endif()

# Compiler & definitions
target_compile_features( ${PROJECT_NAME}
	PRIVATE
		cxx_std_14
)

set_target_properties( ${PROJECT_NAME}
	PROPERTIES
		CXX_EXTENSIONS OFF
)

# Enforce 3rd party dependencies
add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
target_link_libraries( ${PROJECT_NAME} PUBLIC ${3RDPARTY_LIBRARIES} )


if (CVCORELIB_USE_CGAL)
	target_link_libraries( ${PROJECT_NAME} PUBLIC ${CGAL_LIBRARIES} )
	target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_CGAL_LIB )
endif()

if (CVCORELIB_USE_TBB)
	include_directories( ${TBB_INCLUDE_DIRS} )
	target_link_libraries( ${PROJECT_NAME} PUBLIC ${TBB_LIBRARIES} )
	target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_TBB )
else()
	if (NOT WIN32)
		message( WARNING "CVLib configured without parallel algorithm capabilities - see CVCORELIB_USE_TBB" )
	endif()
endif()

if (BUILD_EIGEN3)
	target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_EIGEN )
endif()

# QT (optional)
if ( CVCORELIB_USE_QT_CONCURRENT )
	find_package( Qt5
		COMPONENTS
			Concurrent
		REQUIRED
	)

    set_target_properties( ${PROJECT_NAME} PROPERTIES
        AUTOMOC OFF
        AUTORCC OFF
        AUTOUIC OFF
    )

	target_link_libraries( ${PROJECT_NAME}
		PUBLIC
			Qt5::Concurrent
	)

	target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_QT )
	target_compile_definitions( ${PROJECT_NAME} PUBLIC CV_CORE_LIB_USES_QT_CONCURRENT )

endif()

# Load advanced scripts
include( ../cmake/CMakeInclude.cmake )

if ( CVCORELIB_SHARED )
	InstallSharedLibrary( TARGET ${PROJECT_NAME} )
endif()

if (WIN32)
	target_compile_definitions( ${PROJECT_NAME} PUBLIC _CRT_SECURE_NO_WARNINGS )
endif()

set( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DCV_DEBUG" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DCV_DEBUG" )
