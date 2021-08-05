target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/pipelines.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/odometry/odometry.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/registration/registration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/registration/robust_kernel.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/slac/slac.cpp
)
