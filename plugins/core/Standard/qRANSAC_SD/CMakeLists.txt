option( PLUGIN_STANDARD_QRANSAC_SD "Install qRansac_SD (Shape Detection) plugin" OFF )

if (CV_RANSAC_SUPPORT)
	message("set CV_RANSAC_SUPPORT = ON")
else ()
	SET(PLUGIN_STANDARD_QRANSAC_SD OFF)
	message("Please set CV_RANSAC_SUPPORT = ON, otherwise this plugin is disabled!")
endif()

if( PLUGIN_STANDARD_QRANSAC_SD )
	project( QRANSAC_SD_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )

	#add_subdirectory( extern/RANSAC_SD EXCLUDE_FROM_ALL )
	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )
	
	target_include_directories(${PROJECT_NAME} PRIVATE ${QRANSAC_INCLUDE_DIRS})
	target_link_libraries( ${PROJECT_NAME} ${QRANSAC_LIBRARIES} )

	if (OPENMP_FOUND AND NOT WIN32) #DGM: OpenMP doesn't work with Visual at least (the process loops infinitely)
		target_compile_definitions( ${PROJECT_NAME} PRIVATE DOPARALLEL )
	endif()
endif()
