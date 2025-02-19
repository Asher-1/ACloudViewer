# MeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

option( PLUGIN_IO_QMESH "Install the MeshIO plugin" OFF )

if ( PLUGIN_IO_QMESH )
	project( QMESH_IO_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} TYPE io )
	
	target_include_directories( ${PROJECT_NAME} PUBLIC
								${ASSIMP_INCLUDE_DIR})

	if( CMAKE_VERSION VERSION_LESS 3.13 )
		link_directories( ${ASSIMP_LIB_DIR} )
	else()
		target_link_directories( ${PROJECT_NAME} PUBLIC
								${ASSIMP_LIB_DIR})
	endif()

	add_subdirectory( include )
	add_subdirectory( src )
	
	# Turn off the assimp things we don't want
	set( BUILD_DOCS FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( BUILD_SHARED_LIBS FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_ANDROID_JNIIOSYSTEM FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_ASSIMP_TOOLS FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_COVERALLS FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_NONFREE_C4D_IMPORTER FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_SAMPLES FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_TESTS FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_INSTALL_PDB FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_NO_EXPORT TRUE CACHE INTERNAL "override ASSIMP flags" FORCE )
	
	# Turn off all importers and exporters, then enable only the ones we want
	set( ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT FALSE CACHE INTERNAL "override ASSIMP flags" FORCE )
	
	set( ASSIMP_BUILD_COLLADA_IMPORTER TRUE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_GLTF_IMPORTER TRUE CACHE INTERNAL "override ASSIMP flags" FORCE )
	set( ASSIMP_BUILD_IFC_IMPORTER TRUE CACHE INTERNAL "override ASSIMP flags" FORCE )
	
	# See: https://github.com/asmaloney/MeshIO/issues/1
	if ( AMD64 )
		message( WARNING "The AMD64 option may cause link errors. If you run into link errors regarding 'inflate_fast', try turning this off." )
	endif()

	target_link_libraries( ${PROJECT_NAME} ${ZLIB_LIBRARIES} )
	target_link_libraries( ${PROJECT_NAME} ${ASSIMP_LIBRARIES} )
endif()
