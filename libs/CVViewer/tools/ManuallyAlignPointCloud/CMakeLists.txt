include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR})
if (MSVC)
    include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc)
endif ()

file(GLOB HEADER_FILES "*.h")

add_executable(ManuallyAlignPointCloud ${HEADER_FILES})

target_sources(ManuallyAlignPointCloud PRIVATE
        ManuallyAlignPointCloud.cpp
        AlignmentSession.cpp
        VisualizerForAlignment.cpp
        )

target_link_libraries(ManuallyAlignPointCloud 3rdparty_jsoncpp 3rdparty_tinyfiledialogs)
target_link_libraries(ManuallyAlignPointCloud ECV_DB_LIB)
target_link_libraries(ManuallyAlignPointCloud ECV_IO_LIB)
target_link_libraries(ManuallyAlignPointCloud OPENGL_ENGINE_LIB)

# fix that You must build your code with position independent code if Qt was built with -reduce-relocations
cloudViewer_set_targets_independent(ManuallyAlignPointCloud)
set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
cloudViewer_show_and_abort_on_warning(ManuallyAlignPointCloud)
cloudViewer_set_global_properties(ManuallyAlignPointCloud)
