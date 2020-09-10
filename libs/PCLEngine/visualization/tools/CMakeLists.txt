PCL_ADD_EXECUTABLE_OPT_BUNDLE(pcl_viewer ${SUBSYS_NAME} pcd_viewer.cpp)
target_link_libraries(pcl_viewer pcl_common pcl_io pcl_kdtree pcl_visualization pcl_search)

PCL_ADD_EXECUTABLE_OPT_BUNDLE(pcl_pcd_image_viewer ${SUBSYS_NAME} image_viewer_test.cpp)
target_link_libraries(pcl_pcd_image_viewer pcl_common pcl_io pcl_kdtree pcl_visualization)

