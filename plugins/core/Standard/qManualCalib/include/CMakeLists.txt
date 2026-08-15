
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/qManualCalib.h
		${CMAKE_CURRENT_LIST_DIR}/ManualSensorCalibDlg.h
		${CMAKE_CURRENT_LIST_DIR}/ManualAvmAdjustDlg.h
		${CMAKE_CURRENT_LIST_DIR}/BevBatchExport.h
		${CMAKE_CURRENT_LIST_DIR}/PcdBatchExport.h
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)
