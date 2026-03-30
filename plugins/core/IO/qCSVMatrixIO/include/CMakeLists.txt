
target_sources( ${PROJECT_NAME}
    PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/qCSVMatrixIO.h
		${CMAKE_CURRENT_LIST_DIR}/CSVMatrixCommands.h
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
	    ${CMAKE_CURRENT_SOURCE_DIR}
)
