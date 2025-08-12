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

target_link_libraries(tgeometry PRIVATE 3rdparty_vtk)
# if (NOT USE_SYSTEM_VTK)
#     target_link_libraries(tgeometry PRIVATE 3rdparty_vtk)
# else()
#     find_package (VTK QUIET) # must quiet to avoid boost version issues
#      if (NOT VTK_FOUND)
#           message(FATAL_ERROR
#           "External VTK not found but ParaView requires VTK to function")
#      else ()
#           message(STATUS "Found System installed VTK: " ${VTK_VERSION})
#      endif ()
#     target_link_libraries(tgeometry PRIVATE ${VTK_LIBRARIES})
# endif()

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
