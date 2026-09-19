target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/reconstruction.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/reconstruction_options.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/database/database.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/feature/feature.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/image/image.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/model/model.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/mvs/multi_views_stereo.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/gui/gui.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/sfm/structure_from_motion.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/sfm/mappers.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/vocab_tree/vocab_tree.cpp
        )

# Upstream pycolmap parity (src/pycolmap, Open3D-style bindings): the
# geometry/scene value types; further submodules land incrementally.
target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/geometry/geometry.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/scene/scene.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/sensor/sensor.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/estimators/estimators.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/estimators/covariance_bindings.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/util/util_bindings.cpp
        )
