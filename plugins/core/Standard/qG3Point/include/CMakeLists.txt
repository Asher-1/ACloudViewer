
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/G3PointAction.h
		${CMAKE_CURRENT_LIST_DIR}/G3Point.h
		${CMAKE_CURRENT_LIST_DIR}/G3PointDialog.h
		${CMAKE_CURRENT_LIST_DIR}/GrainsAsEllipsoids.h
		${CMAKE_CURRENT_LIST_DIR}/G3PointDisclaimer.h
		${CMAKE_CURRENT_LIST_DIR}/AnglesCustomPlot.h
		${CMAKE_CURRENT_LIST_DIR}/WolmanCustomPlot.h
		${CMAKE_CURRENT_LIST_DIR}/G3PointPlots.h
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)
