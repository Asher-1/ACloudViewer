# Put together our App bundle on macOS
if( APPLE )
	set_target_properties( ${PROJECT_NAME} PROPERTIES
		MACOSX_BUNDLE TRUE
		INSTALL_RPATH "@executable_path/../Frameworks"
		MACOSX_BUNDLE_NAME ${PROJECT_NAME}
		MACOSX_BUNDLE_GUI_IDENTIFIER com.asher.${PROJECT_NAME}
		MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/CloudViewer.plist
		MACOSX_BUNDLE_ICON_FILE cc_icon.icns
		MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
		MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_VERSION}"
		MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}" )

	set( ACloudViewer_MAC_BASE_DIR ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/${PROJECT_NAME}.app  CACHE INTERNAL "ACloudViewer bundle dir")
	set( ACloudViewer_MAC_FRAMEWORK_DIR ${ACloudViewer_MAC_BASE_DIR}/Contents/Frameworks CACHE INTERNAL "CloudViewer framework dir" )
	set( ACloudViewer_MAC_PLUGIN_DIR ${ACloudViewer_MAC_BASE_DIR}/Contents/cvPlugins CACHE INTERNAL "CloudViewer plugin dir" )
	
	# install icons
	install( FILES cc_icon.icns DESTINATION ${ACloudViewer_MAC_BASE_DIR}/Contents/Resources COMPONENT Runtime )
endif( APPLE )
