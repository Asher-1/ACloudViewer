
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/CVPluginAPI.h
		${CMAKE_CURRENT_LIST_DIR}/ecvColorScaleSelector.h
		${CMAKE_CURRENT_LIST_DIR}/ecvColorScaleEditorDlg.h
		${CMAKE_CURRENT_LIST_DIR}/ecvColorScaleEditorWidget.h
		${CMAKE_CURRENT_LIST_DIR}/ecvCommandLineInterface.h
		${CMAKE_CURRENT_LIST_DIR}/ecvMainAppInterface.h
		${CMAKE_CURRENT_LIST_DIR}/ecvOverlayDialog.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPersistentSettings.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPickingHub.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPickingListener.h
		${CMAKE_CURRENT_LIST_DIR}/ecvQtHelpers.h
		${CMAKE_CURRENT_LIST_DIR}/ecvRenderToFileDlg.h
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_LIST_DIR}
)
