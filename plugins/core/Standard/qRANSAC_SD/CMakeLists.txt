cmake_minimum_required(VERSION 3.0)

option( PLUGIN_STANDARD_QRANSAC_SD "Check to install qRansac_SD (Shape Detection) plugin" OFF )

# CLOUDVIEWER  'Ransac Shape Detection' plugin
if (PLUGIN_STANDARD_QRANSAC_SD)
	project( QRANSAC_SD_PLUGIN )
	
	add_definitions( -DPOINTSWITHINDEX ) #Causes the sub clouds to be generated as partial clones of original cloud
	
	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

	#add_subdirectory (RANSAC_SD_orig)

	include( ../../../CMakePluginTpl.cmake )
	
	add_dependencies(${PROJECT_NAME} build_all_3rd_party_libs)
	target_link_libraries( ${PROJECT_NAME} ${3RDPARTY_LIBRARIES} )

if (OPENMP_FOUND AND NOT WIN32) #DGM: OpenMP doesn't work with Visual at least (the process loops infinitely)
	target_compile_definitions( ${PROJECT_NAME} PRIVATE DOPARALLEL )
endif()

endif()
