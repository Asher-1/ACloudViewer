cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

# calib3d/objdetect are enabled in opencv_build.cmake when PLUGIN_STANDARD_QMANUAL_CALIB=ON.
# Avoid check_include_file_cxx here: it can cache "not found" before ext_opencv installs headers.
set(MCALIB_BUILD_STATIC_DETECT_TOOLS OFF)
if(PLUGIN_STANDARD_QMANUAL_CALIB)
    set(MCALIB_BUILD_STATIC_DETECT_TOOLS ON)
endif()

function(add_mcalib_cli target)
    add_executable(${target} ${ARGN})
    set_target_properties(${target} PROPERTIES AUTOMOC OFF)
    target_compile_features(${target} PUBLIC cxx_std_17)
    # Only tools-local headers; library includes come from linked targets (PUBLIC/INTERFACE).
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -fPIC)
    endif()
endfunction()

function(link_mcalib_io_cli target)
    target_link_libraries(${target}
        MCALIB_CALIB_IO
        CVCoreLib
        3rdparty_threads
        3rdparty_opencv
        3rdparty_eigen3)
endfunction()

add_mcalib_cli(mcalib_rosbag2image mcalib_rosbag2image_main.cpp)
link_mcalib_io_cli(mcalib_rosbag2image)

add_mcalib_cli(mcalib_rosbag2pcd mcalib_rosbag2pcd_main.cpp)
link_mcalib_io_cli(mcalib_rosbag2pcd)

add_mcalib_cli(mcalib_rosbag_merge mcalib_rosbag_merge_main.cpp)
link_mcalib_io_cli(mcalib_rosbag_merge)

add_mcalib_cli(mcalib_rosbag_slice mcalib_rosbag_slice_main.cpp)
link_mcalib_io_cli(mcalib_rosbag_slice)

add_mcalib_cli(mcalib_extrinsic_compare mcalib_extrinsic_compare_main.cpp)
link_mcalib_io_cli(mcalib_extrinsic_compare)

if(MCALIB_BUILD_STATIC_DETECT_TOOLS)
    add_mcalib_cli(mcalib_static_aruco_detect mcalib_static_aruco_detect_main.cpp)
    link_mcalib_io_cli(mcalib_static_aruco_detect)

    add_mcalib_cli(mcalib_static_chessboard_detect
        mcalib_static_chessboard_detect_main.cpp)
    link_mcalib_io_cli(mcalib_static_chessboard_detect)
endif()

add_mcalib_cli(mcalib_export_bev
    mcalib_export_bev_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/BevBatchExport.cpp)
target_include_directories(mcalib_export_bev PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mcalib_export_bev
    MCALIB_CALIB_IO
    MCALIB_BEV_STITCH
    CVCoreLib
    3rdparty_threads
    3rdparty_opencv
    3rdparty_eigen3)
