file(GLOB_RECURSE ALL_HEADER_FILES "*.h")
set (SRC
    IoU.cpp
    Cloud.cpp
    contrib_nns.cpp
    GridSubsampling.cpp
)

if(BUILD_CUDA_MODULE)
    set (SRC
        ${SRC}
        IoU.cu
    )
endif()

add_library(ml_contrib OBJECT ${ALL_HEADER_FILES} ${SRC})
cloudViewer_set_global_properties(ml_contrib)
cloudViewer_link_3rdparty_libraries(ml_contrib)
cloudViewer_set_cloudViewer_lib_properties(ml_contrib)
cloudViewer_show_and_abort_on_warning(ml_contrib)

if(BUILD_CUDA_MODULE)
    target_include_directories(ml_contrib SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
