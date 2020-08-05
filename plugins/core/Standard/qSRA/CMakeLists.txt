cmake_minimum_required(VERSION 3.0)

option( PLUGIN_STANDARD_QSRA "Check to install qSRA (Surface of Revolution Analysis) plugin" OFF )

# ErowCloudViewer 'Surface of Revolution Analysis' plugin
if (PLUGIN_STANDARD_QSRA)
	#DXF support should be enabled for this plugin! (not mandatory though)
	if(NOT OPTION_USE_DXF_LIB )
		message( WARNING "qSRA plugin works best with dxflib support! (enable OPTION_USE_DXF_LIB)" )
	endif()
	
	project( QSRA_PLUGIN )
	
	include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

	# we need includes from the main CC source dir
	include_directories( ${ErowCloudViewer_SOURCE_DIR} )
	
	#we need Color Scale Manager and Editor importer
	file( GLOB ECV_PLUGIN_CUSTOM_HEADER_LIST ${ErowCloudViewer_SOURCE_DIR}/ecvColorScale*.h )
	file( GLOB ECV_PLUGIN_CUSTOM_SOURCE_LIST ${ErowCloudViewer_SOURCE_DIR}/ecvColorScale*.cpp )
	#and the ccRenderToFileDlg class
	list( APPEND ECV_PLUGIN_CUSTOM_HEADER_LIST ${ErowCloudViewer_SOURCE_DIR}/ecvRenderToFileDlg.h )
	list( APPEND ECV_PLUGIN_CUSTOM_SOURCE_LIST ${ErowCloudViewer_SOURCE_DIR}/ecvRenderToFileDlg.cpp )

	#don't forget associated widgets(.ui)
	set( ECV_PLUGIN_CUSTOM_UI_LIST	${ErowCloudViewer_SOURCE_DIR}/ui_templates/colorScaleEditorDlg.ui
									${ErowCloudViewer_SOURCE_DIR}/ui_templates/renderToFileDialog.ui )

	include( ../../../CMakePluginTpl.cmake )
	
	#DXF support
	target_link_DXFLIB( ${PROJECT_NAME} )
endif()
