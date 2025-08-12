add_subdirectory(kernel)

file(GLOB_RECURSE T_GEOMETRY_HEADER "*.h")
add_library(tgeometry OBJECT $<TARGET_OBJECTS:tgeometry_kernel> ${T_GEOMETRY_HEADER})

target_sources(tgeometry PRIVATE
    Image.cpp
    LineSet.cpp
    BoundingVolume.cpp
    PointCloud.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    RaycastingScene.cpp
    TriangleMeshFactory.cpp
    VoxelBlockGrid.cpp
    VtkUtils.cpp
)

cloudViewer_show_and_abort_on_warning(tgeometry)
cloudViewer_set_global_properties(tgeometry)
cloudViewer_set_cloudViewer_lib_properties(tgeometry)
cloudViewer_link_3rdparty_libraries(tgeometry)

# Ensure VTK headers are visible when using system VTK
target_include_directories(tgeometry SYSTEM PRIVATE ${VTK_INCLUDE_DIRS})
message(STATUS "VTK_INCLUDE_DIRS: ${VTK_INCLUDE_DIR}")
if (NOT USE_SYSTEM_VTK)
    target_link_libraries(tgeometry PRIVATE 3rdparty_vtk)
else()
    target_link_libraries(tgeometry PRIVATE ${VTK_LIBRARIES})
endif()

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
