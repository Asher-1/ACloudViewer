function( export_PCL_dlls ) # 1 argument: ARGV0 = destination directory

	#export PCL dlls (if any)
	if( WIN32 AND PCL_DIR )

		# first of all check if files are in ${PCL_DIR} or ${PCL_DIR}/cmake
		# (not sure why but it happens on my win7 system)
		get_filename_component(last_dir ${PCL_DIR} NAME) # get the last line of ${PCL_DIR}
		if (last_dir STREQUAL "cmake")
			get_filename_component(PCL_DIR ${PCL_DIR} PATH) #trim PCL_DIR path if needed
		endif()

		file( GLOB pcl_all_dlls ${PCL_DIR}/bin/*${PCL_RELEASE_SUFFIX}.dll  )
		file( GLOB pcl_debug_dlls ${PCL_DIR}/bin/*${PCL_DEBUG_SUFFIX}.dll  )
		set (pcl_release_dlls "")
		foreach( filename ${pcl_all_dlls} )
			if( NOT "${filename}" IN_LIST pcl_debug_dlls )
				list( APPEND pcl_release_dlls ${filename})
			endif()
		endforeach()

		#release DLLs
		cloudViewer_install_files("${pcl_release_dlls}" "${ARGV0}") #mind the quotes!

		#debug DLLs
		if( CMAKE_CONFIGURATION_TYPES )
			file( GLOB pcl_debug_dlls ${PCL_DIR}/bin/*${PCL_DEBUG_SUFFIX}.dll  )
			foreach( filename ${pcl_debug_dlls} )
				install( FILES ${filename} CONFIGURATIONS Debug DESTINATION ${ARGV0}_debug )
			endforeach()
		endif()

		UNSET(pcl_release_dlls)

	endif()

endfunction()
