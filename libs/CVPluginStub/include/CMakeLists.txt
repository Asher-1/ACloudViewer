
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/ecvDefaultPluginInterface.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPclPluginInterface.h
		${CMAKE_CURRENT_LIST_DIR}/ecvIOPluginInterface.h
		${CMAKE_CURRENT_LIST_DIR}/ecvPluginInterface.h
		${CMAKE_CURRENT_LIST_DIR}/ecvStdPluginInterface.h
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
	    ${CMAKE_CURRENT_LIST_DIR}
)
