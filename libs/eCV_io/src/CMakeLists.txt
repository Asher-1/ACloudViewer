
target_sources( ${PROJECT_NAME}
	PRIVATE

		${CMAKE_CURRENT_LIST_DIR}/AsciiFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/AsciiOpenDlg.cpp
		${CMAKE_CURRENT_LIST_DIR}/AsciiSaveDlg.cpp
		${CMAKE_CURRENT_LIST_DIR}/AutoIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/BinFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/DepthMapFileFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/DxfFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/ecvGlobalShiftManager.cpp
		${CMAKE_CURRENT_LIST_DIR}/ecvShiftAndScaleCloudDlg.cpp
		${CMAKE_CURRENT_LIST_DIR}/FeatureIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/FileIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/FileIOFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/IJsonConvertibleIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/ImageFileFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/ImageIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/LineSetIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/ObjFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/OctreeIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/OFFFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/PlyFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/PlyOpenDlg.cpp
		${CMAKE_CURRENT_LIST_DIR}/PTXFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/RasterGridFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/FileRply.c
		${CMAKE_CURRENT_LIST_DIR}/ShpDBFFields.cpp
		${CMAKE_CURRENT_LIST_DIR}/ShpFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/STLFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/VoxelGridIO.cpp
		${CMAKE_CURRENT_LIST_DIR}/VTKFilter.cpp
		${CMAKE_CURRENT_LIST_DIR}/PinholeCameraTrajectoryIO.cpp

)
