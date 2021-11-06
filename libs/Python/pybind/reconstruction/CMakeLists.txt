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
        ${CMAKE_CURRENT_LIST_DIR}/vocab_tree/vocab_tree.cpp
        )

