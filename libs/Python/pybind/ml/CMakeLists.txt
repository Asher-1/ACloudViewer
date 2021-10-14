target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ml.cpp
)

target_sources(pybind PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/contrib/contrib_nns.cpp
    ${CMAKE_CURRENT_LIST_DIR}/contrib/contrib_subsample.cpp
    ${CMAKE_CURRENT_LIST_DIR}/contrib/contrib.cpp
    ${CMAKE_CURRENT_LIST_DIR}/contrib/iou.cpp
)
