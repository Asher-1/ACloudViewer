function( export_VTK_dlls ) # 1 argument: ARGV0 = destination directory
	#export VTK dlls (if any)
	if( WIN32 AND VTK_DIR )

		# first of all check if files are in ${VTK_DIR} or ${VTK_DIR}/lib/cmake/vtk-version
		# (not sure why but it happens on my win7 system)
		get_filename_component(last_dir ${VTK_DIR} NAME) # get the last line of ${VTK_DIR}
		while( NOT (last_dir STREQUAL "lib") )
			get_filename_component(VTK_DIR ${VTK_DIR} PATH)
			get_filename_component(last_dir ${VTK_DIR} NAME)
		endwhile(condition)
		
		set(VTK_BINARY_DIR "")
		if (last_dir STREQUAL "lib")
			set(FOUNT_VTK_BINARY_DIR ON)
			get_filename_component(VTK_DIR ${VTK_DIR} PATH) #trim VTK_DIR path if needed
			set(VTK_BINARY_DIR "${VTK_DIR}/bin")
			message( STATUS "Found vtk binary dir: ${VTK_BINARY_DIR}" )
		else ()
			message("Cannot find vtk binary dir!")
		endif()
		
		if ( NOT (VTK_BINARY_DIR STREQUAL "") )
			#release DLLs
			file( GLOB vtk_release_dlls ${VTK_BINARY_DIR}/*.dll  )
			copy_files("${vtk_release_dlls}" "${ARGV0}") #mind the quotes!

			#debug DLLs
			if( CMAKE_CONFIGURATION_TYPES )
				file( GLOB vtk_debug_dlls ${VTK_BINARY_DIR}/*-gd.dll  )
				foreach( filename ${vtk_debug_dlls} )
					install( FILES ${filename} CONFIGURATIONS Debug DESTINATION ${ARGV0}_debug )
				endforeach()
			endif()
		endif()
		
		UNSET(VTK_BINARY_DIR)
	endif()

endfunction()
