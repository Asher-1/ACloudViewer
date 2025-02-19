# ------------------------------------------------------------------------------
# FFmpeg+CMake support for ACloudViewer
# ------------------------------------------------------------------------------

find_package(PkgConfig QUIET)
pkg_check_modules(FFMPEG QUIET
		libavcodec
		libavformat
		libavutil
		libswscale
)
if( NOT EXISTS "${FFMPEG_INCLUDE_DIRS}" )
	if (NOT EXISTS "${PKG_FFMPEG_INCLUDE_DIRS}")
		if (BUILD_WITH_CONDA)
			if(WIN32)
				set(FFMPEG_ROOT "$ENV{CONDA_PREFIX}/Library")
				set(FFMPEG_INCLUDE_DIRS "${FFMPEG_ROOT}/include")
			else()
				set(FFMPEG_ROOT "$ENV{CONDA_PREFIX}")
				set(FFMPEG_INCLUDE_DIRS "${FFMPEG_ROOT}/include")
			endif()
		endif()
	else ()
		set(FFMPEG_INCLUDE_DIRS ${PKG_FFMPEG_INCLUDE_DIRS})
		message(STATUS "PKG_FFMPEG_INCLUDE_DIRS: ${PKG_FFMPEG_INCLUDE_DIRS}")
	endif ()
endif()
message(STATUS "Found FFmpeg ${FFMPEG_INCLUDE_DIRS}: ${FFMPEG_LIBRARIES}")

# Find FFmpeg includes
if( NOT EXISTS "${FFMPEG_INCLUDE_DIRS}" )
	find_path( FFMPEG_AVCODEC_INCLUDE_DIR 
		libavcodec/avcodec.h
		HINTS
		$ENV{CONDA_PREFIX}/include
		$ENV{CONDA_PREFIX}/Library/include # windows
	)
	set( FFMPEG_INCLUDE_DIRS "${FFMPEG_AVCODEC_INCLUDE_DIR}" CACHE PATH "FFmpeg include directory" )
	
	message( STATUS "Setting FFmpeg include dir: ${FFMPEG_INCLUDE_DIRS}" )
	unset( FFMPEG_AVCODEC_INCLUDE_DIR CACHE )
endif()

# Find FFmpeg libraries
if( NOT EXISTS "${FFMPEG_LIBRARY_DIR}" )
	find_library( FFMPEG_AVCODEC_LIBRARY_DIR 
		avcodec DOC "FFmpeg library directory" 
		HINTS 
					$ENV{CONDA_PREFIX}/lib
					$ENV{CONDA_PREFIX}/Library/lib # windows
	)
	get_filename_component( FFMPEG_AVCODEC_LIBRARY_DIR ${FFMPEG_AVCODEC_LIBRARY_DIR} DIRECTORY )
	set( FFMPEG_LIBRARY_DIR "${FFMPEG_AVCODEC_LIBRARY_DIR}" CACHE PATH "FFmpeg library directory" )
	
	message( STATUS "Setting FFmpeg library dir: ${FFMPEG_LIBRARY_DIR}" )
	unset( FFMPEG_AVCODEC_LIBRARY_DIR CACHE )
endif()

if( NOT EXISTS "${FFMPEG_INCLUDE_DIRS}" )
	message( FATAL_ERROR "FFmpeg include dir does not exist (FFMPEG_INCLUDE_DIRS): ${FFMPEG_INCLUDE_DIRS}" )
endif()

if( NOT EXISTS "${FFMPEG_LIBRARY_DIR}" )
	message( FATAL_ERROR "FFmpeg library dir does not exist (FFMPEG_LIBRARY_DIR): ${FFMPEG_LIBRARY_DIR}" )
endif()

if( WIN32 )
	if( NOT EXISTS "${FFMPEG_BINARY_DIR}" )
		get_filename_component( FFMPEG_ROOT_DIR "${FFMPEG_LIBRARY_DIR}/.." ABSOLUTE )
		set( FFMPEG_BINARY_DIR "${FFMPEG_ROOT_DIR}/bin" CACHE PATH "FFmpeg binary directory (where the DLLs are ;-)" )
		
		message( STATUS "Setting FFmpeg binary dir: ${FFMPEG_BINARY_DIR}" )
		unset( FFMPEG_ROOT_DIR CACHE )
	endif()
elseif( APPLE )
	if( NOT EXISTS "${FFMPEG_X264_LIBRARY_DIR}" )
		find_library( X264_LIBRARY_DIR x264 )
		get_filename_component( X264_LIBRARY_DIR ${X264_LIBRARY_DIR} DIRECTORY )
		set( FFMPEG_X264_LIBRARY_DIR "${X264_LIBRARY_DIR}" CACHE PATH "The directory containing the x264 library." )
		
		message( STATUS "Setting x264 library dir: ${FFMPEG_X264_LIBRARY_DIR}" )
		unset( X264_LIBRARY_DIR CACHE )
	endif()
	
	if( NOT EXISTS "${FFMPEG_X264_LIBRARY_DIR}" )
		message( FATAL_ERROR "x264 library dir does not exist (FFMPEG_X264_LIBRARY_DIR): ${FFMPEG_X264_LIBRARY_DIR}" )
	endif()
endif()

# link project with ffmpeg libraries
function( target_link_ffmpeg ) # 1 argument: ARGV0 = project name
	set( FFMPEG_LIBRARIES "" )
	set( FFMPEG_LIBRARIES_ROOT_NAME avutil avcodec avformat swscale ) #unused: avdevice avfilter postproc swresample
	
	foreach( libfile ${FFMPEG_LIBRARIES_ROOT_NAME} )
		if(WIN32)
			list( APPEND FFMPEG_LIBRARIES ${FFMPEG_LIBRARY_DIR}/${libfile}.lib )
		elseif( APPLE )
			list( APPEND FFMPEG_LIBRARIES ${FFMPEG_LIBRARY_DIR}/lib${libfile}.dylib )
		else()
			list( APPEND FFMPEG_LIBRARIES ${FFMPEG_LIBRARY_DIR}/lib${libfile}.so )
		endif()
	endforeach()

	target_link_libraries( ${ARGV0} ${FFMPEG_LIBRARIES} )

	if( APPLE )
		target_link_libraries( ${ARGV0}
			"-liconv"
			"-L${FFMPEG_X264_LIBRARY_DIR} -lx264"
			"-lz"
			"-framework CoreVideo"
			)
	endif()

	# Required for some C99 defines
	target_compile_definitions( ${ARGV0} PRIVATE __STDC_CONSTANT_MACROS )

	if (MSVC)
        # fix compiling error on windows platform
        target_compile_options(${ARGV0} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    endif()
	
	unset( FFMPEG_LIBRARIES )
	unset( FFMPEG_LIBRARIES_ROOT_NAME )
endfunction()

function( export_ffmpeg_dlls ) # 1 argument: ARGV0 = destination directory
	if( WIN32 )
		if( EXISTS "${FFMPEG_BINARY_DIR}" )
			set( FFMPEG_DLL "")
			
			file( GLOB CODEC_DLL ${FFMPEG_BINARY_DIR}/avcodec*.dll )
			list( APPEND FFMPEG_DLL ${CODEC_DLL} )

			file( GLOB FORMAT_DLL ${FFMPEG_BINARY_DIR}/avformat*.dll )
			list( APPEND FFMPEG_DLL ${FORMAT_DLL} )

			file( GLOB UTIL_DLL ${FFMPEG_BINARY_DIR}/avutil*.dll )
			list( APPEND FFMPEG_DLL ${UTIL_DLL} )

			file( GLOB SW_DLLS ${FFMPEG_BINARY_DIR}/sw*.dll )
			list( APPEND FFMPEG_DLL ${SW_DLLS} )

			cloudViewer_install_files( "${FFMPEG_DLL}" "${ARGV0}" ) #mind the quotes
		else()
			message( FATAL_ERROR "FFmpeg binary dir does not exist (FFMPEG_BINARY_DIR)" )
		endif()
	endif()
endfunction()
