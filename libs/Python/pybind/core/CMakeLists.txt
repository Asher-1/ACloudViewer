target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/blob.cpp
        ${CMAKE_CURRENT_LIST_DIR}/core.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cuda_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/device.cpp
        ${CMAKE_CURRENT_LIST_DIR}/dtype.cpp
        ${CMAKE_CURRENT_LIST_DIR}/hashmap.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/linalg.cpp
        ${CMAKE_CURRENT_LIST_DIR}/scalar.cpp
        ${CMAKE_CURRENT_LIST_DIR}/size_vector.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensor_accessor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensor_converter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensor_function.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensorlist.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/nns/nearest_neighbor_search.cpp
        )
