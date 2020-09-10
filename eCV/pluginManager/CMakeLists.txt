set(header_list
   ${header_list}
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvPluginInfoDlg.h
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvPluginUIManager.h
   PARENT_SCOPE
)

set(source_list
   ${source_list}
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvPluginInfoDlg.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvPluginUIManager.cpp
   PARENT_SCOPE
)

set( ui_list
	${ui_list}
	${CMAKE_CURRENT_SOURCE_DIR}/ui/ecvPluginInfoDlg.ui
	PARENT_SCOPE
)

set( qrc_list
	${qrc_list}
	${CMAKE_CURRENT_SOURCE_DIR}/pluginManager.qrc
	PARENT_SCOPE
)