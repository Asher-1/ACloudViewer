target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/class_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/io.cpp
)

if (BUILD_AZURE_KINECT)
    target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/sensor.cpp
    )
endif()

if (BUILD_RPC_INTERFACE)
    target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/rpc.cpp
    )
endif()
