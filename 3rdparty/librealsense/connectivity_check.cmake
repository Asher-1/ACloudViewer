include("${CMAKE_CURRENT_LIST_DIR}/cloudviewer_check_connectivity.cmake")
cloudviewer_check_realsense_connectivity(
    "https://librealsense.intel.com/Releases/connectivity_check"
    "${CMAKE_CURRENT_SOURCE_DIR}/connectivity_check")
set(INTERNET_CONNECTION ON)
