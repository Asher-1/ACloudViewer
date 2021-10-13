target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/pipelines.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/color_map/color_map.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/integration/integration.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/odometry/odometry.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/registration/feature.cpp
    ${CMAKE_CURRENT_LIST_DIR}/registration/global_optimization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/registration/registration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/registration/robust_kernels.cpp
)
