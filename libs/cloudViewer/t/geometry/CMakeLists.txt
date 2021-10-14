add_subdirectory(kernel)

file(GLOB_RECURSE T_GEOMETRY_HEADER "*.h")
add_library(tgeometry OBJECT $<TARGET_OBJECTS:tgeometry_kernel> ${T_GEOMETRY_HEADER})

target_sources(tgeometry PRIVATE
    Image.cpp
    PointCloud.cpp
    RaycastingScene.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TSDFVoxelGrid.cpp
)

cloudViewer_show_and_abort_on_warning(tgeometry)
cloudViewer_set_global_properties(tgeometry)
cloudViewer_set_cloudViewer_lib_properties(tgeometry)
cloudViewer_link_3rdparty_libraries(tgeometry)

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
