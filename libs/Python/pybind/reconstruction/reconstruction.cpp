// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/reconstruction.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/reconstruction/database/database.h"
#include "pybind/reconstruction/estimators/ceres_bindings.h"
#include "pybind/reconstruction/estimators/cost_functions_bindings.h"
#include "pybind/reconstruction/estimators/estimators.h"
#include "pybind/reconstruction/feature/feature.h"
#include "pybind/reconstruction/geometry/geometry.h"
#include "pybind/reconstruction/gui/gui.h"
#include "pybind/reconstruction/image/image.h"
#include "pybind/reconstruction/model/model.h"
#include "pybind/reconstruction/mvs/multi_views_stereo.h"
#include "pybind/reconstruction/pipeline/pipeline.h"
#include "pybind/reconstruction/reconstruction_options.h"
#include "pybind/reconstruction/scene/scene.h"
#include "pybind/reconstruction/sensor/sensor.h"
#include "pybind/reconstruction/sfm/structure_from_motion.h"
#include "pybind/reconstruction/util/util_bindings.h"
#include "pybind/reconstruction/vocab_tree/vocab_tree.h"

namespace cloudViewer {
namespace reconstruction {

void pybind_reconstruction(py::module& m) {
    py::module m_reconstruction = m.def_submodule("reconstruction");
    // The options surface registers first: every options class lives on
    // its functional submodule (upstream source-file placement), and other
    // binders take them as default-argument values at registration time.
    // Its def_submodule calls for not-yet-created submodules are harmless:
    // pybind returns the same module object when the owning binder creates
    // it again.
    options::pybind_reconstruction_options_early(m_reconstruction);
    // Upstream pycolmap parity (src/pycolmap): geometry first so the Rigid3d
    // / Sim3d value types exist before the scene classes reference them;
    // sensor next so Bitmap exists for the mvs bindings.
    geometry::pybind_geometry(m_reconstruction);
    sensor::pybind_sensor(m_reconstruction);
    scene::pybind_scene(m_reconstruction);
    estimators::pybind_estimators(m_reconstruction);
    database::pybind_database(m_reconstruction);
    feature::pybind_feature(m_reconstruction);
    image::pybind_image(m_reconstruction);
    model::pybind_model(m_reconstruction);
    mvs::pybind_multi_views_stereo(m_reconstruction);
    // Upstream pycolmap parity (src/pycolmap/estimators/ceres_bindings.cc
    // and cost_functions.cc): the embedded-ceres binding surface that
    // unlocks the solver configuration/results and the cost-function
    // factories without an external pyceres install. Registered after
    // scene/geometry so CameraModelId and Rigid3d value types exist.
    estimators::pybind_ceres(m_reconstruction);
    estimators::pybind_cost_functions(m_reconstruction);
    // Upstream pycolmap parity (src/pycolmap/pipeline): global/hierarchical
    // pipeline options, undistortion, scene clustering, view graph
    // calibration, and dataset synthesis. Registered after estimators so the
    // RotationEstimatorOptions / BundleAdjustmentBackend value types used by
    // the GlobalMapperOptions members exist.
    py::module m_pipeline = m_reconstruction.def_submodule("pipeline");
    sfm::pybind_structure_from_motion(m_reconstruction);
    // The pipeline binder relocates several option classes onto their
    // upstream source-file submodule placement (sfm/scene/image), so it
    // runs AFTER those submodule owners were created, and receives the
    // root to reach them.
    pipeline::pybind_pipeline(m_pipeline, m_reconstruction);
    vocab_tree::pybind_vocab_tree(m_reconstruction);
    gui::pybind_gui(m_reconstruction);
    util::pybind_util(m_reconstruction);
}

}  // namespace reconstruction
}  // namespace cloudViewer
