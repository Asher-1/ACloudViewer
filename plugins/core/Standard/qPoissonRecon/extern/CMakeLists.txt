cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

project( QPOISSON_RECON_LIB )

add_library( ${PROJECT_NAME} STATIC )

# To remove very annoying "NO ARGUMENTS THAT DEPEND ON A TEMPLATE PARAMETER" errors with G++!
if( UNIX OR MINGW )
	target_compile_options( ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fpermissive -w> )
elseif( MSVC )
	target_compile_options( ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj> )
endif()

target_sources( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap/PoissonReconLib.h
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap/PointData.h
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap/PoissonReconLib.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap/PointData.cpp
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap
	PRIVATE
		${POISSON_INCLUDE_DIRS}/Src
)

add_dependencies( ${PROJECT_NAME} ext_poisson )

target_compile_definitions( ${PROJECT_NAME}
	PRIVATE
		_CRT_SECURE_NO_DEPRECATE
		_CRT_SECURE_NO_WARNINGS
		NOMINMAX
)

if (MSVC)
    # fix compiling error on windows platform
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
endif ()

if( WITH_OPENMP AND TARGET 3rdparty_openmp )
	message(STATUS "Enabling OpenMP support for PoissonRecon plugin")
	target_link_libraries( ${PROJECT_NAME} PUBLIC 3rdparty_openmp )
	target_compile_definitions( ${PROJECT_NAME} PRIVATE WITH_OPENMP )
endif()
