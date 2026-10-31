# ABI parity with libColmapLib (see Reconstruction/CMakeLists.txt):
# the COLMAP subtree deliberately strips the fork-wide PCL-compatibility
# macro EIGEN_MAX_ALIGN_BYTES=0 so its TUs compile with compiler-default
# Eigen codegen. The binding TUs here include the same engine headers, so
# they must see identical preprocessor state; otherwise header-heavy types
# (Eigen-aligned members) get ODR-divergent layouts between the pybind
# extension and libColmapLib, which manifests as heap corruption
# ("free(): invalid pointer") whenever the engine mutates objects that the
# extension allocated.
# The ABI mirror (engine feature macros + EIGEN_MAX_ALIGN_BYTES strip) is
# applied in libs/Python/pybind/CMakeLists.txt, where the pybind target is
# defined: per-directory COMPILE_DEFINITIONS do not split a target whose
# sources span multiple directories, so the fix must live at the target's
# own directory level.
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

# Upstream pycolmap parity (src/pycolmap/estimators/ceres_bindings.cc,
# cost_functions.cc): the embedded-ceres binding surface.
target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/estimators/ceres_bindings.cpp
        )
target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/estimators/cost_functions_bindings.cpp
        )

# Upstream pycolmap parity (src/pycolmap/pipeline): global/hierarchical
# pipeline options, undistortion, scene clustering, view graph calibration,
# and dataset synthesis.
target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/pipeline/pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/pipeline/match_features.cpp
        )

target_sources(pybind PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/util/util_bindings.cpp
        )
