# qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause


target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/qMeshIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/IoUtils.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpIfc.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpGltf.cpp
		${CMAKE_CURRENT_LIST_DIR}/Assimp3DMax.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpBlender.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpCollada.cpp
		${CMAKE_CURRENT_LIST_DIR}/IoAbstractLoader.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpCommonFilter.cpp
		
)
