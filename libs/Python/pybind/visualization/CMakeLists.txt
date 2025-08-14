target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/renderoption.cpp
        ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
        ${CMAKE_CURRENT_LIST_DIR}/viewcontrol.cpp
        ${CMAKE_CURRENT_LIST_DIR}/visualization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/visualizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/rendering/material.cpp  # for RPC serialization
        )

if (BUILD_GUI)
    target_sources(pybind PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/o3dvisualizer.cpp
            )

    target_sources(pybind PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/gui/events.cpp
            ${CMAKE_CURRENT_LIST_DIR}/gui/gui.cpp
            )

    target_sources(pybind PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/rendering/rendering.cpp
            )
endif ()

if (BUILD_WEBRTC)
    target_sources(pybind PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/webrtc_server/webrtc_window_system.cpp
            )
endif ()
