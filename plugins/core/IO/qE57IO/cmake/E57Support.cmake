
# libE57Format
set( LIBE57FORMAT_INSTALL_DIR "${E57FORMAT_ROOT}" CACHE PATH "libE57Format install directory (CMAKE_INSTALL_PREFIX from libE57Format)" )

if( NOT LIBE57FORMAT_INSTALL_DIR )
	message( SEND_ERROR "No libE57Format install dir specified (LIBE57FORMAT_INSTALL_DIR)" )
endif()

# ensure the include directory exists
set( LIBE57FORMAT_INCLUDE_DIR "${LIBE57FORMAT_INSTALL_DIR}/include/E57Format" CACHE INTERNAL "" )

if( NOT LIBE57FORMAT_INCLUDE_DIR )
	message( SEND_ERROR "Cannot find include directory '${LIBE57FORMAT_INCLUDE_DIR}'" )
else()
	include_directories( ${LIBE57FORMAT_INSTALL_DIR}/include/E57Format )
endif()

# ensure we can find at least one of the libs
set( LIBE57FORMAT_LIB_DIR "${LIBE57FORMAT_INSTALL_DIR}/lib" CACHE INTERNAL "" )

find_library( LIBE57FORMAT_LIBRARY_RELEASE
				NAMES E57Format.lib libE57Format.a E57Format.so libE57Format.so
				PATHS "${LIBE57FORMAT_LIB_DIR}"
				NO_DEFAULT_PATH
)

find_library( LIBE57FORMAT_LIBRARY_DEBUG
				NAMES E57Format-d.lib libE57Format-d.a E57Format-d.so libE57Format.so
				PATHS "${LIBE57FORMAT_LIB_DIR}"
				NO_DEFAULT_PATH
)

if ( NOT LIBE57FORMAT_LIBRARY_RELEASE AND NOT LIBE57FORMAT_LIBRARY_DEBUG )
	message( SEND_ERROR "Cannot find the libeE57Format library in ${LIBE57FORMAT_LIB_DIR}" )
endif()

# Find Xerces
if (WIN32)
	set( Xerces_INCLUDE_DIR "${XERCES_ROOT}/include" CACHE PATH "Xerces include directory" )
	
	# ensure we can find at least one of the libs
	set( XERCES_LIB_DIR "${XERCES_ROOT}/lib" CACHE INTERNAL "" )
	
	find_library( Xerces_LIBRARY_RELEASE
					NAMES xerces-c_3.lib libxerces-c_3.a xerces-c_3.so libxerces-c_3.so
					PATHS "${XERCES_LIB_DIR}"
					NO_DEFAULT_PATH
	)

	find_library( Xerces_LIBRARY_DEBUG
					NAMES xerces-c_3D.lib libxerces-c_3D.a xerces-c_3D.so libxerces-c_3D.so
					PATHS "${XERCES_LIB_DIR}"
					NO_DEFAULT_PATH
	)
	if ( NOT Xerces_LIBRARY_RELEASE AND NOT Xerces_LIBRARY_DEBUG )
		message( SEND_ERROR "Cannot find the xerces-c library in ${XERCES_LIB_DIR}" )
	else()	
		file( GLOB XERCES_DLL_FILES ${XERCES_ROOT}/bin/xerces-c_3_2.dll )
		cloudViewer_install_files( "${XERCES_DLL_FILES}" "${ACloudViewer_DEST_FOLDER}" )
	endif()
else ()
	include(FindXercesC)
	find_package(XercesC REQUIRED)

	set( Xerces_INCLUDE_DIR ${XercesC_INCLUDE_DIR} CACHE PATH "Xerces include directory" )
	set( Xerces_LIBRARY_RELEASE ${XercesC_LIBRARY} CACHE FILEPATH "Xerces (release) library file" )
	if( CMAKE_CONFIGURATION_TYPES )
		set( Xerces_LIBRARY_DEBUG ${XercesC_LIBRARY} CACHE FILEPATH "Xerces (debug) library file" )
	endif()
endif()

if ( NOT Xerces_INCLUDE_DIR )
	message( SEND_ERROR "No Xerces include dir specified (Xerces_INCLUDE_DIR)" )
else()
	include_directories( ${Xerces_INCLUDE_DIR} )
endif()

# link project with libE57Format libraries
function( target_link_LIBE57FORMAT ) # 1 argument: ARGV0 = project name
	if( LIBE57FORMAT_INSTALL_DIR )
		
		target_link_libraries( ${ARGV0} optimized ${LIBE57FORMAT_LIBRARY_RELEASE})
		if(LIBE57FORMAT_LIBRARY_DEBUG)			
			target_link_libraries( ${ARGV0} debug ${LIBE57FORMAT_LIBRARY_DEBUG})
		endif()
		#Xerces
		if ( CMAKE_CONFIGURATION_TYPES )
			if (Xerces_LIBRARY_DEBUG AND Xerces_LIBRARY_RELEASE)
				target_link_libraries( ${ARGV0} debug ${Xerces_LIBRARY_DEBUG} optimized ${Xerces_LIBRARY_RELEASE} )
			else()
				message( FATAL_ERROR "Unable to find Xerces library. Please set Xerces_LIBRARY_DEBUG and Xerces_LIBRARY_RELEASE to point to the release and debug library files." )
			endif()
		else()
			if (Xerces_LIBRARY_RELEASE)
				target_link_libraries( ${ARGV0} optimized ${Xerces_LIBRARY_RELEASE} )
			endif()
			if(Xerces_LIBRARY_DEBUG)
				target_link_libraries( ${ARGV0} debug ${Xerces_LIBRARY_DEBUG})
			endif()
		endif()

		set_property( TARGET ${ARGV0} APPEND PROPERTY COMPILE_DEFINITIONS CV_E57_SUPPORT )
	else()
		message( SEND_ERROR "No libE57Format install dir specified (LIBE57FORMAT_INSTALL_DIR)" )
	endif()
endfunction()