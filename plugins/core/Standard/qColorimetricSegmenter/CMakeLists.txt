cmake_minimum_required( VERSION 3.0 )

option( PLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER "Check to install colorimetric segmenter plugin" OFF )

if ( PLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER )

    project( QCOLORIMETRIC_SEGMENTER_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )

	include_directories( ${CloudCompare_SOURCE_DIR} )
	
	target_sources( ${PROJECT_NAME}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/qColorimetricSegmenter.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/qColorimetricSegmenter.h
			${CMAKE_CURRENT_SOURCE_DIR}/qColorimetricSegmenter.qrc
			${CMAKE_CURRENT_SOURCE_DIR}/HSV.h
			${CMAKE_CURRENT_SOURCE_DIR}/HSVDialog.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/HSVDialog.h
			${CMAKE_CURRENT_SOURCE_DIR}/HSVDialog.ui
			${CMAKE_CURRENT_SOURCE_DIR}/KmeansDlg.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/KmeansDlg.h
			${CMAKE_CURRENT_SOURCE_DIR}/KmeansDlg.ui
			${CMAKE_CURRENT_SOURCE_DIR}/QuantiDialog.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/QuantiDialog.h
			${CMAKE_CURRENT_SOURCE_DIR}/QuantiDialog.ui
			${CMAKE_CURRENT_SOURCE_DIR}/RgbDialog.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/RgbDialog.h
			${CMAKE_CURRENT_SOURCE_DIR}/RgbDialog.ui
			${CMAKE_CURRENT_SOURCE_DIR}/ScalarDialog.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/ScalarDialog.h
			${CMAKE_CURRENT_SOURCE_DIR}/ScalarDialog.ui
	)

endif()
