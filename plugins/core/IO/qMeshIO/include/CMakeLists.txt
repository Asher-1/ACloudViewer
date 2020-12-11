# qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/qMeshIO.h
		${CMAKE_CURRENT_LIST_DIR}/IoUtils.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpIfc.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpGltf.h
		${CMAKE_CURRENT_LIST_DIR}/Assimp3DMax.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpBlender.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpCollada.h
		${CMAKE_CURRENT_LIST_DIR}/IoAbstractLoader.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpCommonFilter.h
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
	    ${CMAKE_CURRENT_SOURCE_DIR}
)
