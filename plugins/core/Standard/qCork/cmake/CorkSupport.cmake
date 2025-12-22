# ------------------------------------------------------------------------------
# Cork+MPIR/GMP+CMake cross-platform support for ACloudViewer
# ------------------------------------------------------------------------------
# NOTE: This file depends on variables set by cork_download.cmake
#       Make sure to include cork_download.cmake before this file!
# ------------------------------------------------------------------------------

# Verify that CORK_DIR has been set by cork_download.cmake
if(NOT CORK_DIR)
	message(FATAL_ERROR "CORK_DIR is not set! Make sure cork_download.cmake is included before CorkSupport.cmake")
endif()

message(STATUS "CorkSupport: CORK_DIR = ${CORK_DIR}")

# Cork library (CC fork) https://github.com/cloudcompare/cork
if(WIN32)
	# Windows: pre-built binaries have headers in cork/
	set( CORK_INCLUDE_DIR "${CORK_DIR}/cork" CACHE PATH "Cork include directory" )
else()
	# Linux/macOS: built from source, headers in src/
	set( CORK_INCLUDE_DIR "${CORK_DIR}/src" CACHE PATH "Cork include directory" )
endif()

if ( NOT CORK_INCLUDE_DIR )
	message( SEND_ERROR "No Cork include dir specified (CORK_INCLUDE_DIR)" )
else()
	message(STATUS "CorkSupport: Cork include directory: ${CORK_INCLUDE_DIR}")
	include_directories( ${CORK_INCLUDE_DIR} )
endif()

if(WIN32)
	# Windows: Use MPIR (Windows-compatible GMP alternative)
	set( CORK_RELEASE_LIBRARY_FILE "${CORK_DIR}/cork/lib/x64/Release/wincork2013.lib" CACHE FILEPATH "Cork library file (release mode)" )
	set( CORK_DEBUG_LIBRARY_FILE "${CORK_DIR}/cork/lib/x64/Debug/wincork2013.lib" CACHE FILEPATH "Cork library file (debug mode)" )
	
	set( MPIR_INCLUDE_DIR "${CORK_DIR}/mpir" CACHE PATH "MPIR include directory" )
	set( MPIR_RELEASE_LIBRARY_FILE "${CORK_DIR}/mpir/lib/x64/Release/mpir.lib" CACHE FILEPATH "MPIR library file (release mode)" )
	set( MPIR_DEBUG_LIBRARY_FILE "${CORK_DIR}/mpir/lib/x64/Debug/mpir.lib" CACHE FILEPATH "MPIR library file (debug mode)" )
	
	if ( NOT MPIR_INCLUDE_DIR )
		message( SEND_ERROR "No MPIR include dir specified (MPIR_INCLUDE_DIR)" )
	else()
		include_directories( ${MPIR_INCLUDE_DIR} )
	endif()
else()
	# Linux/macOS: Use GMP (already found by cork_download.cmake)
	if(NOT GMP_FOUND)
		message(FATAL_ERROR "GMP was not found by cork_download.cmake. This should not happen!")
	endif()
	
	message(STATUS "CorkSupport: Using GMP libraries: ${GMP_LIBRARIES}")
	message(STATUS "CorkSupport: Using GMP library dirs: ${GMP_LIBRARY_DIRS}")
	message(STATUS "CorkSupport: Using GMP include dirs: ${GMP_INCLUDE_DIRS}")
	include_directories(${GMP_INCLUDE_DIRS})
endif()

# Link project with Cork + MPIR/GMP library
function( target_link_cork ) # 1 argument: ARGV0 = project name

	if(WIN32)
		# Windows: Link with MPIR and pre-built Cork
		if( CORK_RELEASE_LIBRARY_FILE AND MPIR_RELEASE_LIBRARY_FILE )
		
			#Release mode only by default
			target_link_libraries( ${ARGV0} optimized ${CORK_RELEASE_LIBRARY_FILE} ${MPIR_RELEASE_LIBRARY_FILE} )
			
			#optional: debug mode
			if ( CORK_DEBUG_LIBRARY_FILE AND MPIR_DEBUG_LIBRARY_FILE )
				target_link_libraries( ${ARGV0} debug ${CORK_DEBUG_LIBRARY_FILE} ${MPIR_DEBUG_LIBRARY_FILE} )
			endif()
		
		else()
			message( SEND_ERROR "No Cork or MPIR release library files specified (CORK_RELEASE_LIBRARY_FILE / MPIR_RELEASE_LIBRARY_FILE)" )
		endif()
	else()
		# Linux/macOS: Link with GMP and built Cork
		# CORK_DIR already points to src/ext_cork, so just append /lib/libcork.a
		set(CORK_LIBRARY "${CORK_DIR}/lib/libcork.a")
		
		message(STATUS "Looking for Cork library at: ${CORK_LIBRARY}")
		
		# Build full GMP library paths for proper linking
		set(GMP_FULL_LIBRARIES)
		foreach(gmp_lib ${GMP_LIBRARIES})
			# Check if it's already a full path
			if(IS_ABSOLUTE ${gmp_lib})
				list(APPEND GMP_FULL_LIBRARIES ${gmp_lib})
			else()
				# Find the full path using the library dirs
				find_library(GMP_${gmp_lib}_PATH 
					NAMES ${gmp_lib}
					HINTS ${GMP_LIBRARY_DIRS}
					NO_DEFAULT_PATH
				)
				if(GMP_${gmp_lib}_PATH)
					list(APPEND GMP_FULL_LIBRARIES ${GMP_${gmp_lib}_PATH})
					message(STATUS "Found GMP library: ${GMP_${gmp_lib}_PATH}")
				else()
					# Fallback to library name (will use system search)
					list(APPEND GMP_FULL_LIBRARIES ${gmp_lib})
				endif()
			endif()
		endforeach()
		
		if(EXISTS ${CORK_LIBRARY})
			message(STATUS "Linking Cork library: ${CORK_LIBRARY}")
			message(STATUS "Linking GMP libraries: ${GMP_FULL_LIBRARIES}")
			target_link_libraries(${ARGV0} ${CORK_LIBRARY} ${GMP_FULL_LIBRARIES})
		else()
			message(STATUS "Cork library not found at ${CORK_LIBRARY}. It will be built by ext_cork target.")
			# Link anyway - the library will be built before linking due to dependencies
			target_link_libraries(${ARGV0} ${CORK_LIBRARY} ${GMP_FULL_LIBRARIES})
		endif()
	endif()

endfunction()
