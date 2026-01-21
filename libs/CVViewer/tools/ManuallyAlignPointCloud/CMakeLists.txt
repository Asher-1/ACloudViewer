include_directories(${CloudViewer_SOURCE_DIR})

file(GLOB HEADER_FILES "*.h")

add_executable(ManuallyAlignPointCloud ${HEADER_FILES})

target_sources(ManuallyAlignPointCloud PRIVATE
        ManuallyAlignPointCloud.cpp
        AlignmentSession.cpp
        VisualizerForAlignment.cpp
        )

target_link_libraries(ManuallyAlignPointCloud 3rdparty_jsoncpp 3rdparty_tinyfiledialogs)
target_link_libraries(ManuallyAlignPointCloud CV_DB_LIB)
target_link_libraries(ManuallyAlignPointCloud CV_IO_LIB)
target_link_libraries(ManuallyAlignPointCloud CloudViewer)

# fix that You must build your code with position independent code if Qt was built with -reduce-relocations
cloudViewer_set_targets_independent(ManuallyAlignPointCloud)
set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
cloudViewer_show_and_abort_on_warning(ManuallyAlignPointCloud)
cloudViewer_set_global_properties(ManuallyAlignPointCloud)
