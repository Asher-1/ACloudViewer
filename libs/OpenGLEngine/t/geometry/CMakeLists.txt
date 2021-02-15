# Create object library
file(GLOB_RECURSE T_GEOMETRY_HEADER "*.h")

# Create object library
set(T_GEOMETRY_SRC
    kernel/PointCloud.cpp
    kernel/PointCloudCPU.cpp
    kernel/TSDFVoxelGrid.cpp
    kernel/TSDFVoxelGridCPU.cpp
    PointCloud.cpp
    Image.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TSDFVoxelGrid.cpp
)

if (BUILD_CUDA_MODULE)
    list(APPEND T_GEOMETRY_SRC
        kernel/PointCloudCUDA.cu
        kernel/TSDFVoxelGridCUDA.cu
        kernel/NPPImage.cpp
        )
endif()

if (WITH_IPPICV)
    list(APPEND T_GEOMETRY_SRC kernel/IPPImage.cpp)
endif()


add_library(tgeometry OBJECT ${T_GEOMETRY_HEADER} ${T_GEOMETRY_SRC})
cloudViewer_show_and_abort_on_warning(tgeometry)
cloudViewer_set_global_properties(tgeometry)
cloudViewer_set_cloudViewer_lib_properties(tgeometry)
cloudViewer_link_3rdparty_libraries(tgeometry)

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (WITH_IPPICV)
    target_compile_definitions(tgeometry PRIVATE WITH_IPPICV)
endif()
