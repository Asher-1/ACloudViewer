
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/qPCLBase.cpp
		${CMAKE_CURRENT_LIST_DIR}/PclCommands.h
		${CMAKE_CURRENT_LIST_DIR}/PclCommands.cpp
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}
)
