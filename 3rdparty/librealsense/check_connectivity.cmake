# The firmware library is a required link dependency in ACloudViewer. A single
# five-second probe must not silently turn firmware/TM2 support off upstream.
function(cloudviewer_check_realsense_connectivity url destination)
    foreach(attempt RANGE 1 5)
        message(STATUS "Checking RealSense firmware service (attempt ${attempt}/5)")
        file(DOWNLOAD "${url}" "${destination}"
             TIMEOUT 30 INACTIVITY_TIMEOUT 15 STATUS download_status)
        list(GET download_status 0 error_code)
        if(error_code EQUAL 0)
            return()
        endif()

        file(REMOVE "${destination}")
        message(STATUS "RealSense connectivity request failed: ${download_status}")
        if(attempt LESS 5)
            math(EXPR delay "${attempt} * 2")
            execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep "${delay}")
        endif()
    endforeach()
    message(FATAL_ERROR
        "RealSense firmware service unavailable after 5 attempts: ${url}; "
        "${download_status}. Required firmware/TM2 support was not disabled.")
endfunction()
