function( export_OpenCV_dlls ) # 1 argument: ARGV0 = destination directory
	#export OpenCV dlls (if any)
	if( WIN32 AND OpenCV_DIR )

		# first of all check if files are in ${OpenCV_DIR} or ${OpenCV_DIR}/lib
		# (not sure why but it happens on my win7 system)
		get_filename_component(last_dir ${OpenCV_DIR} NAME) # get the last line of ${OpenCV_DIR}
		if (last_dir STREQUAL "lib")
			get_filename_component(OpenCV_DIR ${OpenCV_DIR} PATH) # trim OpenCV_DIR path if needed
		endif()
		
		file( GLOB opencv_all_dlls ${OpenCV_DIR}/bin/*.dll  )
		file( GLOB opencv_debug_dlls ${OpenCV_DIR}/bin/*d.dll  )
		set (opencv_release_dlls "")
		foreach( filename ${opencv_all_dlls} )
			if( NOT "${filename}" IN_LIST opencv_debug_dlls )
				list( APPEND opencv_release_dlls ${filename})
			endif()
		endforeach()

		#release DLLs
		cloudViewer_install_files("${opencv_release_dlls}" "${ARGV0}") #mind the quotes!

		#debug DLLs
		if( CMAKE_CONFIGURATION_TYPES )
			file( GLOB opencv_debug_dlls ${OpenCV_DIR}/bin/*d.dll  )
			foreach( filename ${opencv_debug_dlls} )
				install( FILES ${filename} CONFIGURATIONS Debug DESTINATION ${ARGV0}_debug )
			endforeach()
		endif()
		
		UNSET(opencv_release_dlls)
	endif()

endfunction()
