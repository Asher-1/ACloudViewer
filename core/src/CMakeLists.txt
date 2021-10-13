# SPDX-License-Identifier: MIT
# Copyright © Andy Maloney <asmaloney@gmail.com>

target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/AutoSegmentationTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/BoundingBox.cpp
		${CMAKE_CURRENT_LIST_DIR}/ChamferDistanceTransform.cpp
		${CMAKE_CURRENT_LIST_DIR}/CloudSamplingTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Console.cpp
		${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
		${CMAKE_CURRENT_LIST_DIR}/CVKdTree.cpp
		${CMAKE_CURRENT_LIST_DIR}/TrueKdTree.cpp
		${CMAKE_CURRENT_LIST_DIR}/CVLog.cpp
		${CMAKE_CURRENT_LIST_DIR}/CVMiscTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/CVShareable.cpp
		${CMAKE_CURRENT_LIST_DIR}/CVTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Delaunay2dMesh.cpp
		${CMAKE_CURRENT_LIST_DIR}/DgmOctree.cpp
		${CMAKE_CURRENT_LIST_DIR}/DgmOctreeReferenceCloud.cpp
		${CMAKE_CURRENT_LIST_DIR}/DistanceComputationTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Eigen.cpp
		${CMAKE_CURRENT_LIST_DIR}/ErrorFunction.cpp
		${CMAKE_CURRENT_LIST_DIR}/FastMarching.cpp
		${CMAKE_CURRENT_LIST_DIR}/FastMarchingForPropagation.cpp
		${CMAKE_CURRENT_LIST_DIR}/FileSystem.cpp
		${CMAKE_CURRENT_LIST_DIR}/GeometricalAnalysisTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Helper.cpp
		${CMAKE_CURRENT_LIST_DIR}/IJsonConvertible.cpp
		${CMAKE_CURRENT_LIST_DIR}/IntersectionTest.cpp
		${CMAKE_CURRENT_LIST_DIR}/LocalModel.cpp
		${CMAKE_CURRENT_LIST_DIR}/ManualSegmentationTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/MeshSamplingTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Neighbourhood.cpp
		${CMAKE_CURRENT_LIST_DIR}/NormalDistribution.cpp
		${CMAKE_CURRENT_LIST_DIR}/NormalizedProgress.cpp
		${CMAKE_CURRENT_LIST_DIR}/OrientedBoundingBox.cpp
		${CMAKE_CURRENT_LIST_DIR}/PointProjectionTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Polyline.cpp
		${CMAKE_CURRENT_LIST_DIR}/ReferenceCloud.cpp
		${CMAKE_CURRENT_LIST_DIR}/RegistrationTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/SaitoSquaredDistanceTransform.cpp
		${CMAKE_CURRENT_LIST_DIR}/ScalarField.cpp
		${CMAKE_CURRENT_LIST_DIR}/ScalarFieldTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/SimpleMesh.cpp
		${CMAKE_CURRENT_LIST_DIR}/StatisticalTestingTools.cpp
		${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
		${CMAKE_CURRENT_LIST_DIR}/WeibullDistribution.cpp
		${CMAKE_CURRENT_LIST_DIR}/Cpu.cpp
		${CMAKE_CURRENT_LIST_DIR}/CPUInfo.cpp
		${CMAKE_CURRENT_LIST_DIR}/Parallel.cpp
		${CMAKE_CURRENT_LIST_DIR}/MemoryInfo.cpp
		${CMAKE_CURRENT_LIST_DIR}/Universe.cpp
		${CMAKE_CURRENT_LIST_DIR}/StaticVector.cpp
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)
