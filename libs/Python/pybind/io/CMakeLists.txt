target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/class_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rpc.cpp
)

if (BUILD_AZURE_KINECT)
    target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/sensor.cpp
    )
endif()
