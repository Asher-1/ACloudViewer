
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/CVAppCommon.h
		${CMAKE_CURRENT_LIST_DIR}/CommonSettings.h
		${CMAKE_CURRENT_LIST_DIR}/ecvSettingManager.h
		${CMAKE_CURRENT_LIST_DIR}/ecvApplicationBase.h
		${CMAKE_CURRENT_LIST_DIR}/ecvCameraParamEditDlg.h
		${CMAKE_CURRENT_LIST_DIR}/ecvCustomViewpointButtonDlg.h
		${CMAKE_CURRENT_LIST_DIR}/ecvCustomViewpointsToolbar.h
		${CMAKE_CURRENT_LIST_DIR}/ecvDisplayOptionsDlg.h
		${CMAKE_CURRENT_LIST_DIR}/ecvOptions.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPickOneElementDlg.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPluginManager.h
		${CMAKE_CURRENT_LIST_DIR}/ecvTranslationManager.h
		${CMAKE_CURRENT_LIST_DIR}/ecvVersion.h
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_LIST_DIR}
)
