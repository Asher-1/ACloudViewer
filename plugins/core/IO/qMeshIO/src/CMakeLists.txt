# qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause


target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/COLLADA.cpp
		${CMAKE_CURRENT_LIST_DIR}/glTF.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpCommonFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpAC3D.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpDirectX.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpBlender.cpp
		${CMAKE_CURRENT_LIST_DIR}/AssimpFBX.cpp
		${CMAKE_CURRENT_LIST_DIR}/Assimp3DMax.cpp
		${CMAKE_CURRENT_LIST_DIR}/IFC.cpp
		${CMAKE_CURRENT_LIST_DIR}/qMeshIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/mioAbstractLoader.cpp
		${CMAKE_CURRENT_LIST_DIR}/mioUtils.cpp
)
