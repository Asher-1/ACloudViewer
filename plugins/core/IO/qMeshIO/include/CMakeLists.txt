# qMeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/COLLADA.h
		${CMAKE_CURRENT_LIST_DIR}/glTF.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpCommonFilter.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpAC3D.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpDirectX.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpBlender.h
		${CMAKE_CURRENT_LIST_DIR}/Assimp3DMax.h
		${CMAKE_CURRENT_LIST_DIR}/AssimpFBX.h
		${CMAKE_CURRENT_LIST_DIR}/IFC.h
		${CMAKE_CURRENT_LIST_DIR}/qMeshIO.h
		${CMAKE_CURRENT_LIST_DIR}/mioAbstractLoader.h
		${CMAKE_CURRENT_LIST_DIR}/mioUtils.h
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
	    ${CMAKE_CURRENT_SOURCE_DIR}
)
