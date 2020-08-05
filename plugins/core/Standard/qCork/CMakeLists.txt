cmake_minimum_required(VERSION 3.0)

option( PLUGIN_STANDARD_QCORK "Check to install qCORK plugin" OFF )

# CloudViewer Mesh CSG operations plugin (based on the Cork library)
if (PLUGIN_STANDARD_QCORK)
    project( QCORK_PLUGIN )
	
	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )
    
    # Cork lib and MPIR are needed to build this plugin
    include( CorkSupport.cmake )
    
    include( ../../../CMakePluginTpl.cmake )
    
    target_link_cork( ${PROJECT_NAME} )
endif()
