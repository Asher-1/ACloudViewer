include("${CMAKE_CURRENT_LIST_DIR}/cloudviewer_check_connectivity.cmake")
cloudviewer_check_realsense_connectivity(
    "${REALSENSE_FIRMWARE_URL}/Releases/connectivity_check"
    "${CMAKE_CURRENT_SOURCE_DIR}/connectivity_check")
set(INTERNET_CONNECTION ON)
