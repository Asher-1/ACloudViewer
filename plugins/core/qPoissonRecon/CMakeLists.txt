cmake_minimum_required(VERSION 3.0)

option( INSTALL_QPOISSON_RECON_PLUGIN "Check to install qPoissonRecon plugin" OFF )

# CLOUDVIEWER  'PoissonRecon' plugin
if (INSTALL_QPOISSON_RECON_PLUGIN)
	project( QPOISSON_RECON_PLUGIN )
	
	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )
		
	add_subdirectory (PoissonReconLib)

	include( ../../CMakePluginTpl.cmake )
	
	target_link_libraries( ${PROJECT_NAME} QPOISSON_RECON_LIB )
	include_directories( ${QPOISSON_RECON_LIB_SOURCE_DIR}/Src_CC_wrap )
	
	option( POISSON_RECON_WITH_OPEN_MP "Check to compile PoissonRecon plugin with OpenMP support" OFF )
	
	if ( POISSON_RECON_WITH_OPEN_MP )
		find_package(OpenMP)
		
		if (OPENMP_FOUND)
			set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
			set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
			set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS WITH_OPENMP )
		endif()
	endif ()

endif()