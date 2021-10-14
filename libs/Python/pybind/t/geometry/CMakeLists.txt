target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/geometry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/image.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pointcloud.cpp
    ${CMAKE_CURRENT_LIST_DIR}/raycasting_scene.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tensormap.cpp
    ${CMAKE_CURRENT_LIST_DIR}/trianglemesh.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsdf_voxelgrid.cpp
)
